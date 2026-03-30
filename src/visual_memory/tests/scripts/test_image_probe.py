"""Single-image end-to-end probe test through /voice."""
from __future__ import annotations

import argparse
import io
import os
import sys
import urllib.request

os.environ["ENABLE_DEPTH"] = "0"

from visual_memory.tests.scripts.test_harness import TestRunner, make_test_app, assert_status, TEST_BASE_URL
from visual_memory.api.routes.feedback import feedback_bp
from visual_memory.api.routes.items import items_bp
from visual_memory.api.routes.sightings import sightings_bp
from visual_memory.api.routes.find import find_bp
from visual_memory.api.routes.debug import debug_bp
from visual_memory.api.routes.voice import voice_bp

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


def run_probe(client, label: str, image_bytes: bytes) -> None:
    def step_remember():
        resp = client.post(
            "/voice",
            data={"request_type": "remember", "prompt": label, "image": (io.BytesIO(image_bytes), "probe.jpg")},
            content_type="multipart/form-data",
        )
        assert_status(resp, 200)

    def step_scan():
        resp = client.post(
            "/voice",
            data={"request_type": "scan", "image": (io.BytesIO(image_bytes), "probe.jpg")},
            content_type="multipart/form-data",
        )
        assert_status(resp, 200)

    def step_sightings():
        resp = client.post("/sightings", json={"room_name": "test_room", "sightings": [{"label": label}]})
        assert_status(resp, 200)

    for step_name, step_fn in [
        (f"probe:{label}:remember", step_remember),
        (f"probe:{label}:scan", step_scan),
        (f"probe:{label}:sightings", step_sightings),
    ]:
        _runner.run(step_name, step_fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None)
    parser.add_argument("--label", default="probe_item")
    args = parser.parse_args()

    label = args.label
    image_bytes = _make_jpeg_bytes(args.image)

    if TEST_BASE_URL:
        def step_remote_health():
            req = urllib.request.Request(f"{TEST_BASE_URL}/health")
            with urllib.request.urlopen(req, timeout=10) as resp:
                assert resp.status == 200
        _runner.run("probe:remote:health", step_remote_health)

    all_bps = [voice_bp, feedback_bp, items_bp, sightings_bp, find_bp, debug_bp]
    client, db, cleanup = make_test_app(all_bps)

    run_probe(client, label, image_bytes)

    cleanup()
    sys.exit(_runner.summary())


if __name__ == "__main__":
    main()
