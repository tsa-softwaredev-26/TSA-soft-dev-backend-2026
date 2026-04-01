"""
Integration tests for POST /sightings.
"""
from __future__ import annotations

import os
import sys
import time

os.environ["ENABLE_DEPTH"] = "0"

from visual_memory.tests.scripts.test_harness import (
    TestRunner, make_test_app, assert_status, seed_item,
)
from visual_memory.api.routes.sightings import sightings_bp

_runner = TestRunner("sightings")


def _seed(db):
    seed_item(db, "wallet", emb_seed=1)
    seed_item(db, "keys", emb_seed=2)


client, db, cleanup = make_test_app([sightings_bp], seed_fn=_seed)


def test_single_sighting():
    resp = client.post("/sightings", json={
        "room_name": "kitchen",
        "sightings": [{"label": "wallet", "direction": "to your left", "distance_ft": 3.0}],
    })
    assert_status(resp, 200)
    data = resp.get_json()
    assert data.get("saved") == 1
    assert "wallet" in data.get("labels", [])


def test_multiple_sightings():
    resp = client.post("/sightings", json={
        "room_name": "kitchen",
        "sightings": [
            {"label": "wallet"},
            {"label": "keys"},
        ],
    })
    assert_status(resp, 200)
    data = resp.get_json()
    assert data.get("saved") == 2


def test_room_normalization():
    resp = client.post("/sightings", json={
        "room_name": "In The Kitchen",
        "sightings": [{"label": "wallet"}],
    })
    assert_status(resp, 200)
    data = resp.get_json()
    assert data.get("room_name") == "kitchen"


def test_missing_sightings_field():
    resp = client.post("/sightings", json={"room_name": "kitchen"})
    assert_status(resp, 400)


def test_invalid_json_payload():
    resp = client.post(
        "/sightings",
        data='{"room_name":',
        content_type="application/json",
    )
    assert_status(resp, 400)


def test_empty_sightings_list():
    resp = client.post("/sightings", json={"sightings": []})
    assert_status(resp, 400)


def test_empty_label_skipped():
    resp = client.post("/sightings", json={
        "sightings": [{"label": ""}, {"label": "wallet"}],
    })
    assert_status(resp, 200)
    data = resp.get_json()
    # empty label is skipped, only wallet is saved
    assert data.get("saved") == 1
    assert "wallet" in data.get("labels", [])


def test_all_empty_labels():
    resp = client.post("/sightings", json={
        "sightings": [{"label": ""}, {"label": "   "}],
    })
    assert_status(resp, 400)


def test_no_room_name():
    resp = client.post("/sightings", json={
        "sightings": [{"label": "wallet"}],
    })
    assert_status(resp, 200)
    data = resp.get_json()
    assert data.get("room_name") is None


for name, fn in [
    ("sightings:single", test_single_sighting),
    ("sightings:multiple", test_multiple_sightings),
    ("sightings:room_normalization", test_room_normalization),
    ("sightings:missing_field", test_missing_sightings_field),
    ("sightings:invalid_json", test_invalid_json_payload),
    ("sightings:empty_list", test_empty_sightings_list),
    ("sightings:empty_label_skipped", test_empty_label_skipped),
    ("sightings:all_empty_labels", test_all_empty_labels),
    ("sightings:no_room_name", test_no_room_name),
]:
    _runner.run(name, fn)

cleanup()
sys.exit(_runner.summary())
