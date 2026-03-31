"""
Unit tests for embedding averaging in DatabaseStore.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from uuid import uuid4

import torch

from visual_memory.config import Settings
from visual_memory.database.store import DatabaseStore
from visual_memory.tests.scripts.test_harness import TestRunner, make_embedding

os.environ["ENABLE_DEPTH"] = "0"

_runner = TestRunner("embedding_averaging")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _make_db() -> tuple[DatabaseStore, Path]:
    run_dir = _repo_root() / "data" / "test_embedding_averaging" / uuid4().hex
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ.pop("DB_ENCRYPTION_KEY", None)
    return DatabaseStore(run_dir / "test.db"), run_dir


def _cleanup(run_dir: Path) -> None:
    shutil.rmtree(run_dir, ignore_errors=True)


def _expected_weight(count: int) -> float:
    settings = Settings()
    base = settings.embedding_average_weight_new - (4 * settings.embedding_average_ramp_factor)
    weight = base + (count * settings.embedding_average_ramp_factor)
    return max(0.0, min(weight, 1.0))


def test_single_teach_identity():
    db, run_dir = _make_db()
    try:
        emb = make_embedding(0)
        result = db.upsert_or_average_item(
            label="wallet",
            combined_embedding=emb,
            confidence=0.8,
            timestamp=1.0,
        )
        assert result["averaged"] is False
        assert db.get_embedding_count("wallet") == 1
        stored = db.get_all_items()[0]["combined_embedding"]
        assert torch.allclose(stored, emb)
    finally:
        _cleanup(run_dir)


def test_double_teach_averaging():
    db, run_dir = _make_db()
    try:
        emb1 = make_embedding(1)
        emb2 = make_embedding(2)
        db.upsert_or_average_item(label="wallet", combined_embedding=emb1, confidence=0.8, timestamp=1.0)
        result = db.upsert_or_average_item(label="wallet", combined_embedding=emb2, confidence=0.9, timestamp=2.0)
        assert result["averaged"] is True
        assert db.get_embedding_count("wallet") == 2
        stored = db.get_all_items()[0]["combined_embedding"]
        weight_new = _expected_weight(2)
        expected = (weight_new * emb2) + ((1.0 - weight_new) * emb1)
        assert torch.allclose(stored, expected)
    finally:
        _cleanup(run_dir)


def test_weight_ramp_converges():
    db, run_dir = _make_db()
    try:
        db.upsert_or_average_item(label="wallet", combined_embedding=make_embedding(0), confidence=0.8, timestamp=1.0)
        last_embedding = None
        for idx in range(1, 10):
            last_embedding = make_embedding(idx)
            db.upsert_or_average_item(label="wallet", combined_embedding=last_embedding, confidence=0.8, timestamp=1.0 + idx)
        assert db.get_embedding_count("wallet") == 10
        weight_new = _expected_weight(10)
        assert weight_new == 1.0
        stored = db.get_all_items()[0]["combined_embedding"]
        assert last_embedding is not None
        assert torch.allclose(stored, last_embedding)
    finally:
        _cleanup(run_dir)


for name, fn in [
    ("embedding_avg:single_teach_identity", test_single_teach_identity),
    ("embedding_avg:double_teach_averaging", test_double_teach_averaging),
    ("embedding_avg:weight_ramp_converges", test_weight_ramp_converges),
]:
    _runner.run(name, fn)

sys.exit(_runner.summary())
