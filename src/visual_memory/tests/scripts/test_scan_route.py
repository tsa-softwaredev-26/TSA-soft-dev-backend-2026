"""Integration tests for scan flow via POST /voice and GET /crop."""
from __future__ import annotations

import io
import os
import sys
from PIL import Image

os.environ["ENABLE_DEPTH"] = "0"

import visual_memory.api.pipelines as _pm
from visual_memory.tests.scripts.test_harness import (
    TestRunner, make_test_app, assert_status,
)
from visual_memory.api.routes.voice import voice_bp
from visual_memory.api.routes.crop import crop_bp

_runner = TestRunner("scan")


def _make_jpeg(dark: bool = False) -> bytes:
    color = (5, 5, 5) if dark else (128, 128, 128)
    img = Image.new("RGB", (64, 64), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


client, db, cleanup = make_test_app([voice_bp, crop_bp])


def test_scan_missing_image():
    resp = client.post("/voice", data={"request_type": "scan"}, content_type="multipart/form-data")
    assert_status(resp, 400)


def test_scan_no_matches():
    resp = client.post(
        "/voice",
        data={"request_type": "scan", "image": (io.BytesIO(_make_jpeg()), "frame.jpg")},
        content_type="multipart/form-data",
    )
    assert_status(resp, 200)
    data = resp.get_json().get("result", {})
    assert "matches" in data
    assert "scan_id" in data


def test_scan_with_match():
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
        "/voice",
        data={"request_type": "scan", "image": (io.BytesIO(_make_jpeg()), "frame.jpg")},
        content_type="multipart/form-data",
    )
    _pm._scan_pipeline.run = original_run
    data = resp.get_json().get("result", {})
    assert len(data.get("matches", [])) == 1


for name, fn in [
    ("scan:missing_image", test_scan_missing_image),
    ("scan:no_matches", test_scan_no_matches),
    ("scan:with_match", test_scan_with_match),
]:
    _runner.run(name, fn)

cleanup()
sys.exit(_runner.summary())
