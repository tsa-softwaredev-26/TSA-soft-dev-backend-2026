"""
Integration tests for POST /voice transcription flow.
"""
from __future__ import annotations

import os
import sys

os.environ["ENABLE_DEPTH"] = "0"

from visual_memory.tests.scripts.test_harness import (
    TestRunner,
    make_test_app,
    assert_status,
    seed_item,
    StubSpeechRecognizer,
)
from visual_memory.api.routes.voice import voice_bp
from visual_memory.api.routes.sightings import sightings_bp
from visual_memory.engine.model_registry import registry

_runner = TestRunner("voice")


def _seed(db):
    seed_item(db, "wallet", room_name="kitchen", emb_seed=1)
    seed_item(db, "keys", room_name="bedroom", emb_seed=2)


client, db, cleanup = make_test_app([voice_bp, sightings_bp], seed_fn=_seed)


def _install_stub_recognizer():
    original = registry._whisper_recognizer
    stub = StubSpeechRecognizer()
    stub.set_result("hello world")
    registry._whisper_recognizer = stub
    return stub, original


def _restore_stub_recognizer(original):
    registry._whisper_recognizer = original


def test_voice_missing_audio():
    resp = client.post("/voice", data=b"", content_type="audio/webm")
    assert_status(resp, 400)
    data = resp.get_json()
    assert data.get("error") == "missing audio data"


def test_voice_invalid_format():
    wav_like = b"RIFF" + b"x" * 40
    resp = client.post("/voice", data=wav_like, content_type="audio/wav")
    assert_status(resp, 400)
    data = resp.get_json()
    assert data.get("error") == "invalid audio format"
    assert data.get("format_detected") == "wav"


def test_voice_success_without_context():
    stub, original = _install_stub_recognizer()
    try:
        import visual_memory.api.routes.transcribe as _tr

        old_loader = _tr.load_audio_bytes
        _tr.load_audio_bytes = lambda b, target_sr=16000: (__import__("numpy").zeros(16000, dtype="float32"), 16000)
        resp = client.post("/voice?context=0", data=b"OggS" + b"x" * 32, content_type="audio/ogg")
        _tr.load_audio_bytes = old_loader
    finally:
        _restore_stub_recognizer(original)
    assert_status(resp, 200)
    data = resp.get_json()
    assert isinstance(data.get("transcription"), str)
    assert data.get("transcription_meta", {}).get("context_used") is False


def test_voice_success_with_context():
    stub, original = _install_stub_recognizer()
    try:
        import visual_memory.api.routes.transcribe as _tr

        old_loader = _tr.load_audio_bytes
        _tr.load_audio_bytes = lambda b, target_sr=16000: (__import__("numpy").zeros(8000, dtype="float32"), 16000)
        resp = client.post("/voice?context=1", data=b"OggS" + b"x" * 64, content_type="audio/ogg")
        _tr.load_audio_bytes = old_loader
    finally:
        _restore_stub_recognizer(original)
    assert_status(resp, 200)
    data = resp.get_json()
    assert data.get("transcription_meta", {}).get("context_used") is True
    assert data.get("transcription_meta", {}).get("context_policy")
    assert data.get("transcription_meta", {}).get("context_state_id")


def test_voice_model_failure():
    stub, original = _install_stub_recognizer()
    stub.set_error(RuntimeError("forced test failure"))
    try:
        import visual_memory.api.routes.transcribe as _tr

        old_loader = _tr.load_audio_bytes
        _tr.load_audio_bytes = lambda b, target_sr=16000: (__import__("numpy").zeros(4000, dtype="float32"), 16000)
        resp = client.post("/voice", data=b"OggS" + b"x" * 64, content_type="audio/ogg")
        _tr.load_audio_bytes = old_loader
    finally:
        _restore_stub_recognizer(original)
    assert_status(resp, 500)
    data = resp.get_json()
    assert data.get("error") == "transcription failed"


for name, fn in [
    ("voice:missing_audio", test_voice_missing_audio),
    ("voice:invalid_format", test_voice_invalid_format),
    ("voice:success_no_context", test_voice_success_without_context),
    ("voice:success_with_context", test_voice_success_with_context),
    ("voice:model_failure", test_voice_model_failure),
]:
    _runner.run(name, fn)

cleanup()
sys.exit(_runner.summary())
