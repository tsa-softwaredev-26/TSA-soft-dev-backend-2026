"""
Integration tests for remember flow via POST /voice.
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
from visual_memory.api.routes.voice import voice_bp

_runner = TestRunner("remember")


def _make_image_bytes() -> bytes:
    from PIL import Image
    import io as _io
    img = Image.new("RGB", (64, 64), color=(100, 100, 100))
    buf = _io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


client, db, cleanup = make_test_app([voice_bp])


def test_single_image_success():
    resp = client.post(
        "/voice",
        data={"request_type": "remember", "prompt": "wallet", "image": (io.BytesIO(_make_image_bytes()), "test.jpg")},
        content_type="multipart/form-data",
    )
    assert_status(resp, 200)
    data = resp.get_json().get("result", {})
    assert data.get("success") is True
    assert "result" in data


def test_missing_image():
    resp = client.post(
        "/voice",
        data={"request_type": "remember", "prompt": "wallet"},
        content_type="multipart/form-data",
    )
    assert_status(resp, 400)


def test_missing_prompt():
    resp = client.post(
        "/voice",
        data={"request_type": "remember", "image": (io.BytesIO(_make_image_bytes()), "test.jpg")},
        content_type="multipart/form-data",
    )
    assert_status(resp, 400)


def test_stub_failure_response():
    _pm._remember_pipeline = StubRememberPipeline(success=False)
    resp = client.post(
        "/voice",
        data={"request_type": "remember", "prompt": "wallet", "image": (io.BytesIO(_make_image_bytes()), "test.jpg")},
        content_type="multipart/form-data",
    )
    data = resp.get_json().get("result", {})
    assert data.get("success") is False
    _pm._remember_pipeline = StubRememberPipeline(success=True)


for name, fn in [
    ("remember:single_image_success", test_single_image_success),
    ("remember:missing_image", test_missing_image),
    ("remember:missing_prompt", test_missing_prompt),
    ("remember:stub_failure_response", test_stub_failure_response),
]:
    _runner.run(name, fn)

cleanup()
sys.exit(_runner.summary())
