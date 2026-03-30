"""Cross-endpoint overload stress test."""
from __future__ import annotations

import io
import os
import sys
import time

from PIL import Image

os.environ["ENABLE_DEPTH"] = "0"

from visual_memory.api.routes.ask import ask_bp
from visual_memory.api.routes.feedback import feedback_bp
from visual_memory.api.routes.find import find_bp
from visual_memory.api.routes.item_ask import item_ask_bp
from visual_memory.api.routes.items import items_bp
from visual_memory.api.routes.remember import remember_bp
from visual_memory.api.routes.retrain import retrain_bp
from visual_memory.api.routes.scan import scan_bp
from visual_memory.api.routes.settings_route import settings_bp
from visual_memory.api.routes.sightings import sightings_bp
from visual_memory.api.routes.transcribe import transcribe_bp
from visual_memory.tests.scripts.stress_artifacts import (
    append_recommendations,
    percentile_ms,
    record_failure_modes,
    write_json_report,
)
from visual_memory.tests.scripts.test_harness import TestRunner, make_test_app, seed_item

_runner = TestRunner("stress_endpoints_concurrency")


def _seed(db):
    seed_item(db, "wallet", room_name="kitchen", emb_seed=1)
    seed_item(db, "keys", room_name="office", emb_seed=2)


client, db, cleanup = make_test_app(
    [remember_bp, scan_bp, ask_bp, find_bp, item_ask_bp, feedback_bp, retrain_bp, settings_bp, items_bp, sightings_bp, transcribe_bp],
    seed_fn=_seed,
)


def _img_bytes() -> bytes:
    img = Image.new("RGB", (64, 64), color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_endpoint_mix_overload():
    failures: list[dict] = []
    lat: list[float] = []
    img = _img_bytes()

    def _call(i: int):
        t0 = time.monotonic()
        lane = i % 6
        if lane == 0:
            r = client.post("/ask", json={"query": "wallet"})
            expected = (200, 400)
        elif lane == 1:
            r = client.get("/find?label=wallet")
            expected = (200,)
        elif lane == 2:
            r = client.patch("/settings", json={"projection_head_weight": 0.5})
            expected = (200,)
        elif lane == 3:
            r = client.post("/sightings", json={"room_name": "kitchen", "sightings": [{"label": "wallet"}]})
            expected = (200,)
        elif lane == 4:
            r = client.post("/scan", data={"image": (io.BytesIO(img), "f.jpg")}, content_type="multipart/form-data")
            expected = (200,)
        else:
            r = client.post("/item/ask", json={"scan_id": "s", "label": "wallet", "query": "read this"})
            expected = (200,)
        return r.status_code, expected, (time.monotonic() - t0) * 1000.0

    results = [_call(i) for i in range(240)]

    for idx, (status, expected, ms) in enumerate(results):
        lat.append(ms)
        if status not in expected:
            failures.append({"case": "endpoint_mix", "index": idx, "status": status, "expected": expected})

    report = {
        "suite": "stress_endpoints_concurrency",
        "requests": len(results),
        "p50_ms": percentile_ms(lat, 50),
        "p95_ms": percentile_ms(lat, 95),
        "failure_count": len(failures),
    }
    write_json_report("stress_overload_report.json", report)
    if failures:
        record_failure_modes(failures)
    append_recommendations(
        [
            "[stress_endpoints_concurrency]",
            f"- p95 latency: {report['p95_ms']}ms for {report['requests']} mixed requests",
            "- If failures cluster by endpoint, split worker pools by read/write routes.",
        ]
    )
    assert not failures, f"endpoint overload failures: {failures[:3]}"


for name, fn in [
    ("stress_endpoints_concurrency:mix_overload", test_endpoint_mix_overload),
]:
    _runner.run(name, fn)

cleanup()
sys.exit(_runner.summary())
