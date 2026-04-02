"""
Learning pipeline tests:
- FeedbackStore DB roundtrip and triplet loading
- ProjectionTrainer convergence
- ScanPipeline._apply_head blend ramp behavior
- reload_head from DB
- End-to-end /feedback -> /retrain -> /settings flow
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

os.environ["ENABLE_DEPTH"] = "0"
os.environ["ENABLE_OCR"] = "0"
os.environ["ENABLE_LEARNING"] = "1"

import torch
import torch.nn.functional as F

from visual_memory.tests.scripts.test_harness import (
    TestRunner,
    make_test_app,
    make_embedding,
    assert_status,
)
import visual_memory.api.pipelines as _pm

from visual_memory.database.store import DatabaseStore
from visual_memory.learning.feedback_store import FeedbackStore
from visual_memory.learning.projection_head import ProjectionHead
from visual_memory.learning.trainer import ProjectionTrainer
from visual_memory.api.routes.feedback import feedback_bp
from visual_memory.api.routes.retrain import retrain_bp
from visual_memory.api.routes.settings_route import settings_bp

_runner = TestRunner("learning_pipeline")


def test_feedback_store_roundtrip_and_triplets():
    with tempfile.TemporaryDirectory() as tmp:
        db = DatabaseStore(str(Path(tmp) / "learning.db"))
        store = FeedbackStore(db)
        a = make_embedding(1)
        p = make_embedding(2)
        n = make_embedding(3)
        store.record_positive(a, p, "wallet")
        store.record_negative(a, n, "wallet")
        counts = store.count()
        triplets = store.load_triplets()
        assert counts["positives"] == 1
        assert counts["negatives"] == 1
        assert counts["triplets"] == 1
        assert len(triplets) == 1


def test_trainer_convergence():
    head = ProjectionHead(dim=64)
    trainer = ProjectionTrainer(head, lr=1e-2)
    torch.manual_seed(0)
    anchor = F.normalize(torch.randn(1, 64), dim=1)
    positive = anchor.clone()
    negative = F.normalize(anchor + 0.01 * torch.randn(1, 64), dim=1)
    triplets = [(anchor, positive, negative)]
    first = trainer.train(triplets, epochs=1)
    last = trainer.train(triplets, epochs=10)
    assert last <= first


def test_apply_head_blending_ramp():
    pipeline = _pm._scan_pipeline
    pipeline._head_trained = True
    pipeline._enable_learning = True
    pipeline._head_weight = 1.0
    pipeline._head_ramp_at = 10
    pipeline._head_ramp_power = 1.0
    with torch.no_grad():
        pipeline._head.linear.weight.fill_(0.01)

    emb = make_embedding(42)
    pipeline.set_triplet_count(0)
    out0 = pipeline._apply_head(emb)
    assert torch.allclose(out0, emb, atol=1e-6)

    pipeline.set_triplet_count(5)
    out_mid = pipeline._apply_head(emb)
    sim_mid = F.cosine_similarity(emb, out_mid).item()
    assert sim_mid < 0.999999

    pipeline.set_triplet_count(10)
    out_full = pipeline._apply_head(emb)
    assert out_full.shape == emb.shape


def test_apply_head_blending_ramp_power_curve():
    pipeline = _pm._scan_pipeline
    pipeline._head_trained = True
    pipeline._enable_learning = True
    pipeline._head_weight = 1.0
    pipeline._head_ramp_at = 10
    pipeline._head_ramp_power = 2.0
    with torch.no_grad():
        pipeline._head.linear.weight.fill_(0.01)

    emb = make_embedding(99)
    pipeline.set_triplet_count(5)
    out_pow2 = pipeline._apply_head(emb)
    sim_pow2 = F.cosine_similarity(emb, out_pow2).item()

    pipeline._head_ramp_power = 1.0
    out_lin = pipeline._apply_head(emb)
    sim_lin = F.cosine_similarity(emb, out_lin).item()

    # power=2 should ramp more slowly than linear at midpoint, so output stays closer to input
    assert sim_pow2 > sim_lin


def test_reload_head_from_db():
    pipeline = _pm._scan_pipeline
    trained = ProjectionHead(dim=1536)
    with torch.no_grad():
        trained.linear.weight.fill_(0.01)
    _pm._database.save_projection_head(trained.state_dict())

    loaded = pipeline.reload_head()
    assert loaded is True
    w = pipeline._head.linear.weight.detach().cpu()
    assert torch.allclose(w, torch.full_like(w, 0.01))


def test_scan_cache_keeps_raw_embeddings():
    from visual_memory.pipelines.scan_mode.pipeline import ScanPipeline

    pipeline = ScanPipeline.__new__(ScanPipeline)
    pipeline._emb_cache = OrderedDict()
    pipeline._match_cache = OrderedDict()
    pipeline._crop_cache = OrderedDict()
    pipeline._scan_cache_meta = OrderedDict()

    anchor = make_embedding(123)
    raw_query = make_embedding(124)
    projected_query = make_embedding(125)

    pipeline._cache_feedback_match(
        "scan-raw",
        "wallet",
        anchor,
        raw_query,
        text_likelihood=0.12,
        ocr_ran=True,
        ocr_confidence=0.88,
        similarity_margin=0.14,
        margin_threshold=0.05,
    )

    cached = pipeline.get_cached_embeddings("scan-raw", "wallet")
    assert cached is not None
    cached_anchor, cached_query = cached
    assert torch.allclose(cached_anchor, anchor, atol=1e-6)
    assert torch.allclose(cached_query, raw_query, atol=1e-6)
    assert not torch.allclose(cached_query, projected_query, atol=1e-6)


client, db, cleanup = make_test_app([feedback_bp, retrain_bp, settings_bp])


def _seed_feedback_pair(scan_id: str, label: str, seed: int) -> None:
    import visual_memory.api.pipelines as _pm

    torch.manual_seed(seed)
    a = F.normalize(torch.randn(1, 1536), dim=1)
    q = F.normalize(torch.randn(1, 1536), dim=1)
    _pm._scan_pipeline._cached_embeddings[(scan_id, label)] = (a, q)


def test_feedback_retrain_settings_flow():
    import visual_memory.api.pipelines as _pm

    # Enable deterministic quick training for test
    resp = client.patch(
        "/settings",
        json={
            "min_feedback_for_training": 1,
            "projection_head_epochs": 1,
            "projection_head_weight": 0.5,
            "projection_head_ramp_at": 1,
        },
    )
    assert_status(resp, 200)

    _seed_feedback_pair("scan-a", "wallet", 10)
    _seed_feedback_pair("scan-b", "wallet", 11)

    resp = client.post("/feedback", json={"scan_id": "scan-a", "label": "wallet", "feedback": "correct"})
    assert_status(resp, 200)
    resp = client.post("/feedback", json={"scan_id": "scan-b", "label": "wallet", "feedback": "wrong"})
    assert_status(resp, 200)

    resp = client.post("/retrain")
    assert_status(resp, 200)
    data = resp.get_json()
    assert data.get("started") is True

    deadline = time.time() + 5.0
    status = {}
    while time.time() < deadline:
        r = client.get("/retrain/status")
        assert_status(r, 200)
        status = r.get_json() or {}
        if not status.get("running"):
            break
        time.sleep(0.05)

    assert status.get("running") is False
    assert status.get("error") is None
    last = status.get("last_result") or {}
    assert last.get("trained") is True
    assert last.get("triplets", 0) >= 1

    settings_resp = client.get("/settings")
    assert_status(settings_resp, 200)
    settings_data = settings_resp.get_json()
    assert settings_data.get("triplet_count", 0) >= 1
    assert settings_data.get("head_trained") is True
    assert settings_data.get("projection_head_weight") == 0.5

    # confirm persisted projection head exists in DB
    assert _pm._database.load_projection_head() is not None


for name, fn in [
    ("learning:feedback_store_roundtrip", test_feedback_store_roundtrip_and_triplets),
    ("learning:trainer_convergence", test_trainer_convergence),
    ("learning:apply_head_blend_ramp", test_apply_head_blending_ramp),
    ("learning:apply_head_blend_ramp_power_curve", test_apply_head_blending_ramp_power_curve),
    ("learning:reload_head", test_reload_head_from_db),
    ("learning:e2e_feedback_retrain_settings", test_feedback_retrain_settings_flow),
]:
    _runner.run(name, fn)

cleanup()
sys.exit(_runner.summary())
