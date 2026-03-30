"""Stress tests for /scan and dedup behavior."""
from __future__ import annotations

import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

os.environ["ENABLE_DEPTH"] = "0"

import visual_memory.api.pipelines as _pm
from visual_memory.api.routes.scan import scan_bp
from visual_memory.tests.scripts.stress_artifacts import (
    append_recommendations,
    percentile_ms,
    record_failure_modes,
    write_json_report,
)
from visual_memory.tests.scripts.test_harness import TestRunner, assert_status, make_test_app
from visual_memory.utils.similarity_utils import deduplicate_matches

_runner = TestRunner("stress_scan")
client, db, cleanup = make_test_app([scan_bp])


def _jpeg_bytes(color: tuple[int, int, int], size: tuple[int, int] = (64, 64)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _post_scan(payload: bytes) -> tuple[int, float]:
    t0 = time.monotonic()
    resp = client.post(
        "/scan",
        data={"image": (io.BytesIO(payload), "frame.jpg")},
        content_type="multipart/form-data",
    )
    dt = (time.monotonic() - t0) * 1000.0
    return resp.status_code, dt


def test_dedup_bomb_compaction():
    base = {
        "label": "wallet",
        "direction": "to your left",
        "similarity": 0.9,
        "narration": "Wallet to your left.",
        "box": [10, 10, 120, 120],
        "distance_ft": 3.0,
    }
    dup = [dict(base) for _ in range(250)]
    # Add a second distinct same-label box and another label.
    dup.append({**base, "box": [300, 300, 360, 360], "similarity": 0.7})
    dup.append(
        {
            "label": "keys",
            "direction": "ahead",
            "similarity": 0.8,
            "narration": "Keys ahead.",
            "box": [420, 40, 520, 120],
            "distance_ft": 4.2,
        }
    )
    deduped = deduplicate_matches(dup, iou_threshold=0.5)
    labels = [m["label"] for m in deduped]
    assert len(deduped) <= 3, f"expected heavy compaction, got {len(deduped)}"
    assert "wallet" in labels and "keys" in labels


def test_scan_overload_parallel():
    original_run = _pm._scan_pipeline.run

    def _fake_run(image, scan_id: str = "", focal_length_px: float = 0.0):
        # Simulate large candidate pool compacted to no matches.
        return {
            "matches": [],
            "count": 0,
            "scan_id": scan_id,
            "is_dark": False,
            "darkness_level": 5.0,
            "candidate_count": 1000,
        }

    _pm._scan_pipeline.run = _fake_run
    failures: list[dict] = []
    latencies: list[float] = []
    try:
        payloads = [
            _jpeg_bytes((255, 255, 255)),
            _jpeg_bytes((0, 0, 0)),
            _jpeg_bytes((255, 0, 0)),
            _jpeg_bytes((0, 255, 0)),
            _jpeg_bytes((0, 0, 255)),
        ]
        jobs = payloads * 30  # 150 requests
        with ThreadPoolExecutor(max_workers=12) as ex:
            results = list(ex.map(_post_scan, jobs))
        for i, (status, latency_ms) in enumerate(results):
            latencies.append(latency_ms)
            if status != 200:
                failures.append({"case": "scan_overload_parallel", "index": i, "status": status})
        assert not failures, f"unexpected statuses: {failures[:3]}"
    finally:
        _pm._scan_pipeline.run = original_run

    report = {
        "suite": "stress_scan",
        "requests": len(latencies),
        "p50_ms": percentile_ms(latencies, 50),
        "p95_ms": percentile_ms(latencies, 95),
        "max_ms": round(max(latencies), 2) if latencies else 0.0,
        "failure_count": len(failures),
    }
    write_json_report("stress_scan_report.json", report)
    if failures:
        record_failure_modes(failures)
    append_recommendations(
        [
            "[stress_scan]",
            f"- p95 latency: {report['p95_ms']}ms over {report['requests']} requests",
            "- If p95 regresses, lower per-request candidate fan-out or batch in smaller chunks.",
        ]
    )


for name, fn in [
    ("stress_scan:dedup_bomb_compaction", test_dedup_bomb_compaction),
    ("stress_scan:overload_parallel", test_scan_overload_parallel),
]:
    _runner.run(name, fn)

cleanup()
sys.exit(_runner.summary())
