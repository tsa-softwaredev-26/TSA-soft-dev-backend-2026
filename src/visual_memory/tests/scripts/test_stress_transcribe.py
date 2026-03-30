"""Stress tests for /voice transcription throughput and validation."""
from __future__ import annotations

import os
import sys
import time

os.environ["ENABLE_DEPTH"] = "0"

from visual_memory.api.routes.sightings import sightings_bp
from visual_memory.api.routes.voice import voice_bp
from visual_memory.engine.model_registry import registry
from visual_memory.tests.scripts.stress_artifacts import (
    append_recommendations,
    percentile_ms,
    record_failure_modes,
    write_json_report,
)
from visual_memory.tests.scripts.test_harness import StubSpeechRecognizer, TestRunner, make_test_app, seed_item

_runner = TestRunner("stress_voice")


def _seed(db):
    seed_item(db, "wallet", room_name="kitchen", emb_seed=1)


client, db, cleanup = make_test_app([voice_bp, sightings_bp], seed_fn=_seed)


def _install_stub():
    original = registry._whisper_recognizer
    stub = StubSpeechRecognizer()
    registry._whisper_recognizer = stub
    return original


def _restore_stub(original):
    registry._whisper_recognizer = original


def _valid_audio() -> bytes:
    return b"OggS" + b"x" * 256


def _invalid_audio() -> bytes:
    return b"RIFF" + b"x" * 40


def _post_voice(data: bytes, context: int = 1):
    t0 = time.monotonic()
    resp = client.post(f"/voice?context={context}", data=data, content_type="audio/ogg")
    return resp.status_code, (time.monotonic() - t0) * 1000.0


def test_voice_load_mix():
    import visual_memory.api.routes.transcribe as _tr

    original_stub = _install_stub()
    old_loader = _tr.load_audio_bytes
    failures: list[dict] = []
    latencies: list[float] = []
    try:
        _tr.load_audio_bytes = lambda b, target_sr=16000: (__import__("numpy").zeros(4000, dtype="float32"), 16000)
        jobs = [(_valid_audio(), 1)] * 100 + [(_valid_audio(), 0)] * 50
        results = [_post_voice(*x) for x in jobs]
        for i, (status, ms) in enumerate(results):
            latencies.append(ms)
            if status != 200:
                failures.append({"case": "voice_valid_lane", "index": i, "status": status})

        _tr.load_audio_bytes = old_loader
        bad_statuses = []
        for _ in range(40):
            resp = client.post("/voice", data=_invalid_audio(), content_type="audio/wav")
            bad_statuses.append(resp.status_code)
        if any(s != 400 for s in bad_statuses):
            failures.append({"case": "voice_invalid_lane", "statuses": bad_statuses})
    finally:
        _tr.load_audio_bytes = old_loader
        _restore_stub(original_stub)

    report = {
        "suite": "stress_voice",
        "valid_requests": 150,
        "invalid_requests": 40,
        "p50_ms": percentile_ms(latencies, 50),
        "p95_ms": percentile_ms(latencies, 95),
        "failure_count": len(failures),
    }
    write_json_report("stress_voice_report.json", report)
    if failures:
        record_failure_modes(failures)
    append_recommendations(
        [
            "[stress_voice]",
            f"- p95 latency: {report['p95_ms']}ms",
            "- Keep voice format validation strict for non-ogg payloads.",
        ]
    )
    assert not failures, f"stress voice failures: {failures[:3]}"


for name, fn in [
    ("stress_voice:load_mix", test_voice_load_mix),
]:
    _runner.run(name, fn)

cleanup()
sys.exit(_runner.summary())
