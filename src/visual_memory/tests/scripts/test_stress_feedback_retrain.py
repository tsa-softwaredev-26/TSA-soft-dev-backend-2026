"""Stress tests for /feedback and /retrain concurrency."""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

os.environ["ENABLE_DEPTH"] = "0"

import visual_memory.api.pipelines as _pm
from visual_memory.api.routes.feedback import feedback_bp
from visual_memory.api.routes.retrain import retrain_bp
from visual_memory.api.routes.settings_route import settings_bp
from visual_memory.tests.scripts.stress_artifacts import (
    append_recommendations,
    percentile_ms,
    record_failure_modes,
    write_json_report,
)
from visual_memory.tests.scripts.test_harness import TestRunner, make_test_app

_runner = TestRunner("stress_feedback_retrain")
client, db, cleanup = make_test_app([feedback_bp, retrain_bp, settings_bp])


def _seed(scan_id: str, label: str, seed: int):
    _pm._scan_pipeline.seed_embeddings(scan_id, label, seed=seed)


def test_feedback_retrain_contention():
    failures: list[dict] = []
    for i in range(120):
        _seed(f"scan-{i}", "wallet", i)
    for i in range(120):
        fb = "correct" if i % 2 == 0 else "wrong"
        resp = client.post("/feedback", json={"scan_id": f"scan-{i}", "label": "wallet", "feedback": fb})
        if resp.status_code != 200:
            failures.append({"case": "feedback_seed", "index": i, "status": resp.status_code})

    resp = client.patch("/settings", json={"min_feedback_for_training": 1, "projection_head_epochs": 1})
    if resp.status_code != 200:
        failures.append({"case": "settings_patch", "status": resp.status_code})

    def _start_retrain():
        t0 = time.monotonic()
        r = client.post("/retrain")
        return r.status_code, r.get_json() or {}, (time.monotonic() - t0) * 1000.0

    with ThreadPoolExecutor(max_workers=8) as ex:
        starts = list(ex.map(lambda _: _start_retrain(), range(24)))

    latencies = [ms for _, _, ms in starts]
    started = sum(1 for status, data, _ in starts if status == 200 and data.get("started") is True)
    already = sum(1 for status, data, _ in starts if status == 200 and data.get("reason") == "already_running")
    if started < 1:
        failures.append({"case": "retrain_start", "detail": "no retrain started"})
    if already < 1:
        failures.append({"case": "retrain_contention", "detail": "no already_running responses"})

    deadline = time.time() + 6.0
    last = {}
    while time.time() < deadline:
        status_resp = client.get("/retrain/status")
        if status_resp.status_code != 200:
            failures.append({"case": "status_poll", "status": status_resp.status_code})
            break
        last = status_resp.get_json() or {}
        if not last.get("running"):
            break
        time.sleep(0.05)

    report = {
        "suite": "stress_feedback_retrain",
        "feedback_count": 120,
        "retrain_attempts": len(starts),
        "started": started,
        "already_running": already,
        "p95_ms": percentile_ms(latencies, 95),
        "status_final": last,
        "failure_count": len(failures),
    }
    write_json_report("stress_learning_report.json", report)
    if failures:
        record_failure_modes(failures)
    append_recommendations(
        [
            "[stress_feedback_retrain]",
            f"- retrain attempts={report['retrain_attempts']} started={started} already_running={already}",
            "- If contention behavior regresses, audit retrain lock boundaries and status transitions.",
        ]
    )
    assert not failures, f"stress feedback/retrain failures: {failures[:3]}"


for name, fn in [
    ("stress_feedback_retrain:contention", test_feedback_retrain_contention),
]:
    _runner.run(name, fn)

cleanup()
sys.exit(_runner.summary())
