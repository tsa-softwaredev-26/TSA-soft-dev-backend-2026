"""
Integration tests for POST /feedback, POST /retrain, GET /retrain/status.

Uses FeedbackStore(db) backed by a temp SQLite DB.
No model loading.
"""
from __future__ import annotations

import os
import sys
import time

os.environ["ENABLE_DEPTH"] = "0"

import torch
import torch.nn.functional as F

import visual_memory.api.pipelines as _pm
from visual_memory.tests.scripts.test_harness import (
    TestRunner, make_test_app, assert_status, assert_json_field,
)
from visual_memory.api.routes.feedback import feedback_bp
from visual_memory.api.routes.retrain import retrain_bp

_runner = TestRunner("feedback_retrain")


client, db, cleanup = make_test_app([feedback_bp, retrain_bp])


def _seed_scan_cache(scan_id: str, label: str) -> None:
    """Put (anchor, query) tensors into the stub's cache."""
    _pm._scan_pipeline.seed_embeddings(scan_id, label)


def test_feedback_correct():
    _seed_scan_cache("scan-001", "wallet")
    resp = client.post("/feedback", json={
        "scan_id": "scan-001",
        "label": "wallet",
        "feedback": "correct",
    })
    assert_status(resp, 200)
    data = resp.get_json()
    assert data.get("recorded") is True
    assert data.get("label") == "wallet"
    assert data.get("feedback") == "correct"
    assert "triplets" in data
    assert "min_for_training" in data


def test_feedback_wrong():
    _seed_scan_cache("scan-002", "keys")
    resp = client.post("/feedback", json={
        "scan_id": "scan-002",
        "label": "keys",
        "feedback": "wrong",
    })
    assert_status(resp, 200)
    data = resp.get_json()
    assert data.get("recorded") is True
    assert data.get("feedback") == "wrong"


def test_feedback_invalid_value():
    _seed_scan_cache("scan-003", "wallet")
    resp = client.post("/feedback", json={
        "scan_id": "scan-003",
        "label": "wallet",
        "feedback": "maybe",
    })
    assert_status(resp, 400)


def test_feedback_missing_scan_id():
    resp = client.post("/feedback", json={"label": "wallet", "feedback": "correct"})
    assert_status(resp, 400)


def test_feedback_missing_label():
    resp = client.post("/feedback", json={"scan_id": "x", "feedback": "correct"})
    assert_status(resp, 400)


def test_feedback_missing_feedback():
    resp = client.post("/feedback", json={"scan_id": "x", "label": "wallet"})
    assert_status(resp, 400)


def test_feedback_unknown_scan_id():
    resp = client.post("/feedback", json={
        "scan_id": "no-such-scan",
        "label": "wallet",
        "feedback": "correct",
    })
    assert_status(resp, 404)


def test_retrain_insufficient_data():
    # Fresh DB has 0 triplets, min_feedback_for_training defaults to 10
    resp = client.post("/retrain")
    assert_status(resp, 200)
    data = resp.get_json()
    assert data.get("started") is False
    assert data.get("reason") == "insufficient_data"
    assert "triplets" in data
    assert "min_required" in data


def test_retrain_status_idle():
    resp = client.get("/retrain/status")
    assert_status(resp, 200)
    data = resp.get_json()
    assert "running" in data
    assert "last_result" in data
    assert "error" in data


for name, fn in [
    ("feedback:correct", test_feedback_correct),
    ("feedback:wrong", test_feedback_wrong),
    ("feedback:invalid_value", test_feedback_invalid_value),
    ("feedback:missing_scan_id", test_feedback_missing_scan_id),
    ("feedback:missing_label", test_feedback_missing_label),
    ("feedback:missing_feedback_field", test_feedback_missing_feedback),
    ("feedback:unknown_scan_id", test_feedback_unknown_scan_id),
    ("retrain:insufficient_data", test_retrain_insufficient_data),
    ("retrain:status_idle", test_retrain_status_idle),
]:
    _runner.run(name, fn)

cleanup()
sys.exit(_runner.summary())
