"""
Integration tests for GET /health and debug endpoints.
"""
from __future__ import annotations

import os
import sys

os.environ["ENABLE_DEPTH"] = "0"

import visual_memory.api.pipelines as _pm
from visual_memory.tests.scripts.test_harness import (
    TestRunner, make_test_app, assert_status, seed_item,
)
from visual_memory.api.routes.health import health_bp
from visual_memory.api.routes.debug import debug_bp
from visual_memory.api.routes.items import items_bp

_runner = TestRunner("health_debug")


def _seed(db):
    seed_item(db, "wallet", emb_seed=1)
    seed_item(db, "keys", emb_seed=2)


client, db, cleanup = make_test_app([health_bp, debug_bp, items_bp], seed_fn=_seed)


def test_health_ok():
    resp = client.get("/health")
    assert_status(resp, 200)
    data = resp.get_json()
    assert data.get("status") == "ok"


def test_debug_wipe_items():
    resp = client.post("/debug/wipe", json={"target": "items", "confirm": True})
    assert_status(resp, 200)
    data = resp.get_json()
    assert "wiped" in data or "ok" in data or data.get("target") == "items"
    items = db.get_all_items()
    assert len(items) == 0


def test_debug_config_patch():
    resp = client.patch("/debug/config", json={"similarity_threshold": 0.5})
    assert_status(resp, 200)
    s = _pm._settings
    assert s.similarity_threshold == 0.5


for name, fn in [
    ("health:ok", test_health_ok),
    ("debug:wipe_items", test_debug_wipe_items),
    ("debug:config_patch", test_debug_config_patch),
]:
    _runner.run(name, fn)

cleanup()
sys.exit(_runner.summary())
