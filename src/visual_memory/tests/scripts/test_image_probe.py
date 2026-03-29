"""
Single-image end-to-end probe test.

Exercises the full API pipeline with one real image and label.
Works in both local mode (Flask test client with stubs) and remote mode.

Usage:
    # Local stub mode (default):
    python -m visual_memory.tests.scripts.test_image_probe

    # With a real image:
    python -m visual_memory.tests.scripts.test_image_probe --image path/to/photo.jpg --label wallet

    # Against a live server:
    TEST_BASE_URL=http://localhost:5050 python -m visual_memory.tests.scripts.test_image_probe \
        --image path/to/photo.jpg --label wallet
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time

os.environ["ENABLE_DEPTH"] = "0"

from visual_memory.tests.scripts.test_harness import (
    TestRunner, make_test_app, assert_status, StubRememberPipeline, TEST_BASE_URL,
)
import visual_memory.api.pipelines as _pm
from visual_memory.api.routes.remember import remember_bp
from visual_memory.api.routes.scan import scan_bp
from visual_memory.api.routes.feedback import feedback_bp
from visual_memory.api.routes.items import items_bp
from visual_memory.api.routes.sightings import sightings_bp
from visual_memory.api.routes.find import find_bp
from visual_memory.api.routes.debug import debug_bp

_runner = TestRunner("image_probe")


def _make_jpeg_bytes(path: str = None) -> bytes:
    from PIL import Image as _Image
    if path:
        img = _Image.open(path).convert("RGB")
    else:
        img = _Image.new("RGB", (64, 64), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def run_probe(client, db, label: str, image_bytes: bytes, is_remote: bool = False) -> None:
    scan_id = None

    def step_remember():
        resp = client.post(
            "/remember",
            data={"prompt": label, "image": (io.BytesIO(image_bytes), "probe.jpg")},
            content_type="multipart/form-data",
        )
        assert_status(resp, 200)
        data = resp.get_json()
        assert data.get("success") is True or "result" in data, f"remember failed: {data}"

    def step_scan():
        nonlocal scan_id
        resp = client.post(
            "/scan",
            data={"image": (io.BytesIO(image_bytes), "probe.jpg")},
            content_type="multipart/form-data",
        )
        assert_status(resp, 200)
        data = resp.get_json()
        assert "scan_id" in data
        scan_id = data["scan_id"]

    def step_sightings():
        resp = client.post("/sightings", json={
            "room_name": "test_room",
            "sightings": [{"label": label}],
        })
        assert_status(resp, 200)

    def step_rename():
        new_name = f"{label}_probe_renamed"
        resp = client.post(f"/items/{label}/rename", json={"new_label": new_name})
        # Ignore 404 if item wasn't saved (stub mode may not persist)
        if resp.status_code not in (200, 404):
            assert_status(resp, 200)

    def step_list_items():
        resp = client.get("/items")
        assert_status(resp, 200)
        data = resp.get_json()
        assert "items" in data

    def step_wipe():
        resp = client.post("/debug/wipe", json={"target": "items"})
        if resp.status_code not in (200, 404, 405):
            assert_status(resp, 200)

    for step_name, step_fn in [
        (f"probe:{label}:remember", step_remember),
        (f"probe:{label}:scan", step_scan),
        (f"probe:{label}:sightings", step_sightings),
        (f"probe:{label}:rename", step_rename),
        (f"probe:{label}:list_items", step_list_items),
        (f"probe:{label}:wipe", step_wipe),
    ]:
        _runner.run(step_name, step_fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="Path to input image")
    parser.add_argument("--label", default="probe_item", help="Ground truth label")
    args = parser.parse_args()

    label = args.label
    image_bytes = _make_jpeg_bytes(args.image)
    is_remote = bool(TEST_BASE_URL)

    all_bps = [remember_bp, scan_bp, feedback_bp, items_bp, sightings_bp, find_bp, debug_bp]
    client, db, cleanup = make_test_app(all_bps)

    run_probe(client, db, label, image_bytes, is_remote=is_remote)

    cleanup()
    sys.exit(_runner.summary())


if __name__ == "__main__":
    main()
