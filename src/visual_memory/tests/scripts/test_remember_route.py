"""
Integration tests for POST /remember.

Uses StubRememberPipeline - no model loading.
"""
from __future__ import annotations

import io
import os
import sys

os.environ["ENABLE_DEPTH"] = "0"

from visual_memory.tests.scripts.test_harness import (
    TestRunner, make_test_app, assert_status, StubRememberPipeline,
)
import visual_memory.api.pipelines as _pm
from visual_memory.api.routes.remember import remember_bp
from visual_memory.api.routes.scan import scan_bp

_runner = TestRunner("remember")


def _make_image_bytes() -> bytes:
    from PIL import Image
    import io as _io
    img = Image.new("RGB", (64, 64), color=(100, 100, 100))
    buf = _io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


client, db, cleanup = make_test_app([remember_bp, scan_bp])


def test_single_image_success():
    resp = client.post(
        "/remember",
        data={"prompt": "wallet", "image": (io.BytesIO(_make_image_bytes()), "test.jpg")},
        content_type="multipart/form-data",
    )
    assert_status(resp, 200)
    data = resp.get_json()
    assert data.get("success") is True
    assert "result" in data
    assert data["result"]["label"] == "wallet"


def test_missing_image():
    resp = client.post(
        "/remember",
        data={"prompt": "wallet"},
        content_type="multipart/form-data",
    )
    assert_status(resp, 400)


def test_missing_prompt():
    resp = client.post(
        "/remember",
        data={"image": (io.BytesIO(_make_image_bytes()), "test.jpg")},
        content_type="multipart/form-data",
    )
    assert_status(resp, 400)


def test_response_schema_fields():
    resp = client.post(
        "/remember",
        data={"prompt": "wallet", "image": (io.BytesIO(_make_image_bytes()), "test.jpg")},
        content_type="multipart/form-data",
    )
    data = resp.get_json()
    result = data.get("result", {})
    for field in ("label", "confidence", "detection_quality", "blur_score", "is_blurry",
                  "second_pass", "second_pass_prompt", "box", "ocr_text", "ocr_confidence"):
        assert field in result, f"missing field in result: {field}"


def test_stub_failure_response():
    _pm._remember_pipeline = StubRememberPipeline(success=False)
    resp = client.post(
        "/remember",
        data={"prompt": "wallet", "image": (io.BytesIO(_make_image_bytes()), "test.jpg")},
        content_type="multipart/form-data",
    )
    data = resp.get_json()
    assert data.get("success") is False
    _pm._remember_pipeline = StubRememberPipeline(success=True)


def test_reload_database_called_on_success():
    called = []
    original = _pm._scan_pipeline.reload_database
    _pm._scan_pipeline.reload_database = lambda: called.append(1)
    client.post(
        "/remember",
        data={"prompt": "wallet", "image": (io.BytesIO(_make_image_bytes()), "test.jpg")},
        content_type="multipart/form-data",
    )
    _pm._scan_pipeline.reload_database = original
    assert len(called) >= 1, "reload_database not called after success"


def test_multi_image_success():
    img_bytes = _make_image_bytes()
    resp = client.post(
        "/remember",
        data={
            "prompt": "wallet",
            "images[]": [
                (io.BytesIO(img_bytes), "a.jpg"),
                (io.BytesIO(img_bytes), "b.jpg"),
            ],
        },
        content_type="multipart/form-data",
    )
    data = resp.get_json()
    assert data.get("success") is True
    assert "images_tried" in data
    assert "images_with_detection" in data


def test_multi_image_no_detection():
    _pm._remember_pipeline = StubRememberPipeline(success=False)
    img_bytes = _make_image_bytes()
    resp = client.post(
        "/remember",
        data={
            "prompt": "wallet",
            "images[]": [
                (io.BytesIO(img_bytes), "a.jpg"),
                (io.BytesIO(img_bytes), "b.jpg"),
            ],
        },
        content_type="multipart/form-data",
    )
    data = resp.get_json()
    assert data.get("success") is False
    assert data.get("images_tried") == 2
    assert data.get("images_with_detection") == 0
    _pm._remember_pipeline = StubRememberPipeline(success=True)


def test_stub_blurry_flag():
    blurry_result = {
        "label": "wallet",
        "confidence": 0.5,
        "detection_quality": "low",
        "detection_hint": "Image appears blurry",
        "blur_score": 30.0,
        "is_blurry": True,
        "second_pass": False,
        "second_pass_prompt": None,
        "box": [0, 0, 50, 50],
        "ocr_text": "",
        "ocr_confidence": 0.0,
    }
    _pm._remember_pipeline = StubRememberPipeline(success=True, result=blurry_result)
    resp = client.post(
        "/remember",
        data={"prompt": "wallet", "image": (io.BytesIO(_make_image_bytes()), "test.jpg")},
        content_type="multipart/form-data",
    )
    data = resp.get_json()
    assert data["result"]["is_blurry"] is True
    _pm._remember_pipeline = StubRememberPipeline(success=True)


for name, fn in [
    ("remember:single_image_success", test_single_image_success),
    ("remember:missing_image", test_missing_image),
    ("remember:missing_prompt", test_missing_prompt),
    ("remember:response_schema_fields", test_response_schema_fields),
    ("remember:stub_failure_response", test_stub_failure_response),
    ("remember:reload_database_called", test_reload_database_called_on_success),
    ("remember:multi_image_success", test_multi_image_success),
    ("remember:multi_image_no_detection", test_multi_image_no_detection),
    ("remember:blurry_flag", test_stub_blurry_flag),
]:
    _runner.run(name, fn)

cleanup()
sys.exit(_runner.summary())
