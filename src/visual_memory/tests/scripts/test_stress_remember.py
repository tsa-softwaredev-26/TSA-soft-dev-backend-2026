"""Stress tests for /remember burst and payload quality."""
from __future__ import annotations

import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

os.environ["ENABLE_DEPTH"] = "0"

from visual_memory.api.routes.remember import remember_bp
from visual_memory.api.routes.scan import scan_bp
from visual_memory.tests.scripts.stress_artifacts import (
    append_recommendations,
    percentile_ms,
    record_failure_modes,
    write_json_report,
)
from visual_memory.tests.scripts.test_harness import TestRunner, make_test_app

_runner = TestRunner("stress_remember")
client, db, cleanup = make_test_app([remember_bp, scan_bp])


def _img_bytes(color: tuple[int, int, int]) -> bytes:
    img = Image.new("RGB", (64, 64), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _remember_ok(color: tuple[int, int, int]) -> tuple[int, float]:
    t0 = time.monotonic()
    resp = client.post(
        "/remember",
        data={"prompt": "wallet", "image": (io.BytesIO(_img_bytes(color)), "sample.jpg")},
        content_type="multipart/form-data",
    )
    return resp.status_code, (time.monotonic() - t0) * 1000.0


def _remember_missing_prompt() -> int:
    resp = client.post(
        "/remember",
        data={"image": (io.BytesIO(_img_bytes((0, 0, 0))), "sample.jpg")},
        content_type="multipart/form-data",
    )
    return resp.status_code


def test_remember_burst_and_invalid_mix():
    latencies: list[float] = []
    failures: list[dict] = []
    colors = [(255, 255, 255), (0, 0, 0), (0, 255, 0), (255, 0, 0), (0, 0, 255)]
    with ThreadPoolExecutor(max_workers=10) as ex:
        ok_results = list(ex.map(_remember_ok, colors * 20))  # 100 valid requests
    for i, (status, ms) in enumerate(ok_results):
        latencies.append(ms)
        if status not in (200, 500):
            failures.append({"case": "remember_valid_lane", "index": i, "status": status})

    invalid_statuses = [_remember_missing_prompt() for _ in range(30)]
    if any(s != 400 for s in invalid_statuses):
        failures.append({"case": "remember_invalid_lane", "bad_statuses": invalid_statuses})

    report = {
        "suite": "stress_remember",
        "valid_requests": len(ok_results),
        "invalid_requests": len(invalid_statuses),
        "p50_ms": percentile_ms(latencies, 50),
        "p95_ms": percentile_ms(latencies, 95),
        "failure_count": len(failures),
    }
    write_json_report("stress_remember_report.json", report)
    if failures:
        record_failure_modes(failures)
    append_recommendations(
        [
            "[stress_remember]",
            f"- p95 latency: {report['p95_ms']}ms, failures: {report['failure_count']}",
            "- If failures spike, isolate multipart parsing and temp file handling paths.",
        ]
    )
    assert not failures, f"stress remember failures: {failures[:2]}"


for name, fn in [
    ("stress_remember:burst_and_invalid_mix", test_remember_burst_and_invalid_mix),
]:
    _runner.run(name, fn)

cleanup()
sys.exit(_runner.summary())
