"""CPU-only unit tests for the projection head triplet system. No model loading."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch

from visual_memory.learning.feedback_store import FeedbackStore
from visual_memory.learning.projection_head import ProjectionHead
from visual_memory.learning.trainer import ProjectionTrainer, triplet_loss
from visual_memory.tests.scripts.test_harness import TestRunner

os.environ["ENABLE_DEPTH"] = "0"


# Test 1: ProjectionHead is identity at init

def test_identity_at_init():
    head = ProjectionHead(dim=1536)
    head.eval()
    x = torch.randn(4, 1536)
    x_norm = torch.nn.functional.normalize(x, dim=1)
    with torch.no_grad():
        out = head(x_norm)
    sim = torch.nn.functional.cosine_similarity(x_norm, out).mean().item()
    assert abs(sim - 1.0) < 1e-5, f"cosine sim={sim:.6f}, expected ~1.0"


# Test 2: triplet_loss = 0 when anchor==positive, anchor!=negative

def test_triplet_loss_zero():
    a = torch.randn(1, 64)
    n = torch.randn(1, 64)
    loss = triplet_loss(a, a, n, margin=0.2).item()
    assert loss == 0.0, f"loss={loss:.4f}, expected 0.0"


# Test 3: triplet_loss > 0 when anchor==negative (bad ordering)

def test_triplet_loss_nonzero():
    a = torch.randn(1, 64)
    p = torch.randn(1, 64)
    loss = triplet_loss(a, p, a, margin=0.2).item()
    assert loss > 0.0, f"loss={loss:.4f}, expected >0.0"


# Test 4: FeedbackStore roundtrip

def test_feedback_store_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        from visual_memory.database.store import DatabaseStore
        db = DatabaseStore(str(Path(tmpdir) / "test.db"))
        store = FeedbackStore(db)
        anc = torch.randn(1, 1536)
        q = torch.randn(1, 1536)
        store.record_positive(anc, q, label="wallet")
        store.record_positive(anc, q, label="wallet")
        store.record_negative(anc, q, label="wallet")
        store.record_negative(anc, q, label="wallet")
        triplets = store.load_triplets()
        counts = store.count()
        assert len(triplets) == 4, f"triplets={len(triplets)}, expected 4"
        assert counts["positives"] == 2, f"positives={counts['positives']}, expected 2"
        assert counts["negatives"] == 2, f"negatives={counts['negatives']}, expected 2"


# Test 5: ProjectionTrainer.train_step returns float, loss decreases

def test_trainer_loss_decreases():
    head = ProjectionHead(dim=32)
    trainer = ProjectionTrainer(head, lr=1e-2)

    # anchor and positive are identical; negative is very close to anchor
    # -> pos_dist=0, neg_dist~0, initial loss > 0 (margin=0.2 kicks in)
    torch.manual_seed(0)
    a = torch.nn.functional.normalize(torch.randn(1, 32), dim=1)
    p = a.clone()
    n = torch.nn.functional.normalize(a + 0.01 * torch.randn(1, 32), dim=1)

    first_loss = trainer.train_step(a, p, n)
    assert isinstance(first_loss, float), f"train_step returned {type(first_loss)}, expected float"

    for _ in range(9):
        last_loss = trainer.train_step(a, p, n)

    assert last_loss < first_loss, f"loss did not decrease: first={first_loss:.4f}, last={last_loss:.4f}"


if __name__ == "__main__":
    runner = TestRunner("projection_head")
    for name, fn in [
        ("projection:identity_at_init", test_identity_at_init),
        ("projection:triplet_loss_zero", test_triplet_loss_zero),
        ("projection:triplet_loss_nonzero", test_triplet_loss_nonzero),
        ("projection:feedback_store_roundtrip", test_feedback_store_roundtrip),
        ("projection:trainer_loss_decreases", test_trainer_loss_decreases),
    ]:
        runner.run(name, fn)
    sys.exit(runner.summary())
