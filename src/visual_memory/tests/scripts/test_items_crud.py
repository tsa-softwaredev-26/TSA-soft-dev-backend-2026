"""
Integration tests for GET /items, DELETE /items/<label>, POST /items/<label>/rename.
"""
from __future__ import annotations

import os
import sys
import time

os.environ["ENABLE_DEPTH"] = "0"

import visual_memory.api.pipelines as _pm
from visual_memory.tests.scripts.test_harness import (
    TestRunner, make_test_app, assert_status, seed_item,
)
from visual_memory.api.routes.items import items_bp
from visual_memory.api.routes.scan import scan_bp

_runner = TestRunner("items_crud")


client, db, cleanup = make_test_app([items_bp, scan_bp])

seed_item(db, "wallet", emb_seed=1)
seed_item(db, "keys", emb_seed=2)
seed_item(db, "phone", emb_seed=3)


def test_list_items_all():
    resp = client.get("/items")
    assert_status(resp, 200)
    data = resp.get_json()
    assert "items" in data
    assert "count" in data
    assert data["count"] == len(data["items"])
    labels = {i["label"] for i in data["items"]}
    assert "wallet" in labels
    assert "keys" in labels


def test_list_items_empty():
    from visual_memory.database.store import DatabaseStore
    import tempfile
    from pathlib import Path
    tmp = tempfile.mkdtemp()
    empty_db_path = str(Path(tmp) / "empty.db")
    from visual_memory.config.settings import Settings
    empty_db = DatabaseStore(empty_db_path)
    orig = _pm._database
    _pm._database = empty_db
    resp = client.get("/items")
    _pm._database = orig
    data = resp.get_json()
    assert data["count"] == 0
    assert data["items"] == []


def test_list_items_filter_by_label():
    resp = client.get("/items?label=wallet")
    assert_status(resp, 200)
    data = resp.get_json()
    for item in data["items"]:
        assert item["label"] == "wallet"


def test_delete_existing_item():
    reload_called = []
    orig = _pm._scan_pipeline.reload_database
    _pm._scan_pipeline.reload_database = lambda: reload_called.append(1)
    resp = client.delete("/items/phone")
    _pm._scan_pipeline.reload_database = orig
    assert_status(resp, 200)
    data = resp.get_json()
    assert data.get("deleted") is True
    assert len(reload_called) >= 1


def test_delete_nonexistent_item():
    resp = client.delete("/items/doesnotexist_xyz")
    assert_status(resp, 404)


def test_rename_success():
    reload_called = []
    orig = _pm._scan_pipeline.reload_database
    _pm._scan_pipeline.reload_database = lambda: reload_called.append(1)
    resp = client.post("/items/wallet/rename", json={"new_label": "my wallet"})
    _pm._scan_pipeline.reload_database = orig
    assert_status(resp, 200)
    data = resp.get_json()
    assert data.get("renamed") is True
    assert data.get("old_label") == "wallet"
    assert data.get("new_label") == "my wallet"
    assert len(reload_called) >= 1


def test_rename_same_label():
    resp = client.post("/items/keys/rename", json={"new_label": "keys"})
    assert_status(resp, 400)


def test_rename_empty_label():
    resp = client.post("/items/keys/rename", json={"new_label": ""})
    assert_status(resp, 400)


def test_rename_missing_new_label():
    resp = client.post("/items/keys/rename", json={})
    assert_status(resp, 400)


def test_rename_invalid_json():
    resp = client.post(
        "/items/keys/rename",
        data='{"new_label":',
        content_type="application/json",
    )
    assert_status(resp, 400)


def test_rename_nonexistent_source():
    resp = client.post("/items/ghost_item_xyz/rename", json={"new_label": "something"})
    assert_status(resp, 404)


def test_rename_conflict_without_force():
    # Seed a second item
    seed_item(db, "laptop", emb_seed=10)
    seed_item(db, "tablet", emb_seed=11)
    resp = client.post("/items/laptop/rename", json={"new_label": "tablet"})
    # should return 409 since "tablet" already exists
    assert_status(resp, 409)
    data = resp.get_json()
    assert data.get("conflict") is True


def test_rename_conflict_with_force():
    resp = client.post("/items/laptop/rename", json={"new_label": "tablet", "force": True})
    assert_status(resp, 200)
    data = resp.get_json()
    assert data.get("renamed") is True


for name, fn in [
    ("items:list_all", test_list_items_all),
    ("items:list_empty", test_list_items_empty),
    ("items:filter_by_label", test_list_items_filter_by_label),
    ("items:delete_existing", test_delete_existing_item),
    ("items:delete_nonexistent", test_delete_nonexistent_item),
    ("items:rename_success", test_rename_success),
    ("items:rename_same_label", test_rename_same_label),
    ("items:rename_empty_label", test_rename_empty_label),
    ("items:rename_missing_field", test_rename_missing_new_label),
    ("items:rename_invalid_json", test_rename_invalid_json),
    ("items:rename_nonexistent_source", test_rename_nonexistent_source),
    ("items:rename_conflict_no_force", test_rename_conflict_without_force),
    ("items:rename_conflict_with_force", test_rename_conflict_with_force),
]:
    _runner.run(name, fn)

cleanup()
sys.exit(_runner.summary())
