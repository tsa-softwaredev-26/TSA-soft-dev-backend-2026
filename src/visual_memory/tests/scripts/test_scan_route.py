"""
Integration tests for POST /scan and GET /crop.

Uses StubScanPipeline - no model loading.
"""
from __future__ import annotations

import io
import os
import sys
from PIL import Image

os.environ["ENABLE_DEPTH"] = "0"

import visual_memory.api.pipelines as _pm
from visual_memory.tests.scripts.test_harness import (
    TestRunner, make_test_app, assert_status, StubScanPipeline,
)
from visual_memory.api.routes.scan import scan_bp
from visual_memory.api.routes.crop import crop_bp

_runner = TestRunner("scan")


def _make_jpeg(dark: bool = False) -> bytes:
    color = (5, 5, 5) if dark else (128, 128, 128)
    img = Image.new("RGB", (64, 64), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


client, db, cleanup = make_test_app([scan_bp, crop_bp])

_VALID_DIRECTIONS = {
    "to your left", "slightly left", "ahead", "slightly right", "to your right"
}


def test_scan_missing_image():
    resp = client.post("/scan", data={}, content_type="multipart/form-data")
    assert_status(resp, 400)


def test_scan_no_matches():
    resp = client.post(
        "/scan",
        data={"image": (io.BytesIO(_make_jpeg()), "frame.jpg")},
        content_type="multipart/form-data",
    )
    assert_status(resp, 200)
    data = resp.get_json()
    assert "matches" in data
    assert data.get("count", 0) == 0 or isinstance(data.get("matches"), list)
    assert "scan_id" in data


def test_scan_returns_scan_id():
    resp = client.post(
        "/scan",
        data={"image": (io.BytesIO(_make_jpeg()), "frame.jpg")},
        content_type="multipart/form-data",
    )
    data = resp.get_json()
    assert "scan_id" in data
    assert isinstance(data["scan_id"], str)
    assert len(data["scan_id"]) > 0


def test_scan_with_match():
    # Patch the stub to return a match
    original_run = _pm._scan_pipeline.run
    def _fake_run(image, scan_id="", focal_length_px=0.0):
        return {
            "matches": [{
                "label": "wallet",
                "direction": "to your left",
                "similarity": 0.75,
                "narration": "Wallet 3 feet to your left.",
                "box": [10, 10, 100, 100],
                "distance_ft": 3.0,
            }],
            "count": 1,
            "scan_id": scan_id,
            "is_dark": False,
            "darkness_level": 15.0,
        }
    _pm._scan_pipeline.run = _fake_run
    resp = client.post(
        "/scan",
        data={"image": (io.BytesIO(_make_jpeg()), "frame.jpg")},
        content_type="multipart/form-data",
    )
    _pm._scan_pipeline.run = original_run
    data = resp.get_json()
    assert len(data.get("matches", [])) == 1
    match = data["matches"][0]
    assert match["label"] == "wallet"
    assert match["direction"] in _VALID_DIRECTIONS
    assert "similarity" in match
    assert "box" in match


def test_scan_invalid_focal_length():
    resp = client.post(
        "/scan",
        data={"image": (io.BytesIO(_make_jpeg()), "frame.jpg"), "focal_length_px": "not_a_float"},
        content_type="multipart/form-data",
    )
    assert_status(resp, 400)


def test_crop_missing_scan_id():
    resp = client.get("/crop?index=0")
    assert_status(resp, 400)


def test_crop_invalid_index():
    resp = client.get("/crop?scan_id=abc&index=notanint")
    assert_status(resp, 400)


def test_crop_unknown_scan_id():
    resp = client.get("/crop?scan_id=doesnotexist12345&index=0")
    assert_status(resp, 404)


def test_crop_returns_jpeg():
    # Seed a crop in the stub
    _pm._scan_pipeline.seed_crop("testscan", 0)
    resp = client.get("/crop?scan_id=testscan&index=0")
    assert_status(resp, 200)
    assert resp.content_type == "image/jpeg" or b"JFIF" in resp.data or len(resp.data) > 100


for name, fn in [
    ("scan:missing_image", test_scan_missing_image),
    ("scan:no_matches", test_scan_no_matches),
    ("scan:returns_scan_id", test_scan_returns_scan_id),
    ("scan:with_match", test_scan_with_match),
    ("scan:invalid_focal_length", test_scan_invalid_focal_length),
    ("crop:missing_scan_id", test_crop_missing_scan_id),
    ("crop:invalid_index", test_crop_invalid_index),
    ("crop:unknown_scan_id", test_crop_unknown_scan_id),
    ("crop:returns_jpeg", test_crop_returns_jpeg),
]:
    _runner.run(name, fn)

cleanup()
sys.exit(_runner.summary())
