"""
Integration tests for the stateful WebSocket voice session.

Uses flask_socketio.test_client() - no real model loading, no real audio decoding.
Whisper and audio decoding are stubbed the same way as test_transcribe_route.py.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ["ENABLE_DEPTH"] = "0"

from visual_memory.tests.scripts.test_harness import (
    TestRunner,
    StubSpeechRecognizer,
    StubScanPipeline,
    StubRememberPipeline,
    seed_item,
)
from visual_memory.engine.model_registry import registry

_runner = TestRunner("voice_ws")


def _make_ws_test_app(seed_fn=None, empty_db: bool = False, ws_headers: dict | None = None):
    """Create a Flask app with SocketIO + stubs, return (sio_client, db, cleanup)."""
    import numpy as np
    import shutil
    from flask import Flask
    from flask_socketio import SocketIO
    from visual_memory.database.store import DatabaseStore
    from visual_memory.config.settings import Settings
    from visual_memory.config.user_settings import UserSettings
    from visual_memory.learning.feedback_store import FeedbackStore
    import visual_memory.api.pipelines as _pm
    from visual_memory.api.routes.voice_ws import register_events
    from visual_memory.api.voice_session import _sessions

    tmp_dir = tempfile.mkdtemp()
    db_path = str(Path(tmp_dir) / "test.db")
    db = DatabaseStore(db_path)

    settings = Settings()
    settings.db_path = db_path
    settings.enable_depth = False
    settings.enable_ocr = False

    scan_stub = StubScanPipeline()
    remember_stub = StubRememberPipeline()
    feedback_stub = FeedbackStore(db)
    user_settings = UserSettings()

    _pm._settings = settings
    _pm._database = db
    _pm._scan_pipeline = scan_stub
    _pm._remember_pipeline = remember_stub
    _pm._feedback_store = feedback_stub
    _pm._user_settings = user_settings

    if seed_fn is not None and not empty_db:
        seed_fn(db)

    # Stub Whisper
    stub_recognizer = StubSpeechRecognizer()
    stub_recognizer.set_result("where is my wallet")
    original_recognizer = registry._whisper_recognizer
    registry._whisper_recognizer = stub_recognizer

    # Stub audio decoder so no real audio is decoded
    import visual_memory.api.routes.transcribe as _tr
    original_loader = _tr.load_audio_bytes
    _tr.load_audio_bytes = lambda b, target_sr=16000: (
        np.zeros(16000, dtype="float32"), 16000
    )

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
    sio = SocketIO(app, async_mode="threading")
    register_events(sio)

    client = sio.test_client(app, headers=ws_headers)

    def cleanup():
        if client.is_connected():
            client.disconnect()
        _tr.load_audio_bytes = original_loader
        registry._whisper_recognizer = original_recognizer
        _sessions.clear()
        _pm._database = None
        _pm._scan_pipeline = None
        _pm._remember_pipeline = None
        _pm._feedback_store = None
        _pm._user_settings = None
        _pm._settings = Settings()
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return client, sio, db, stub_recognizer, cleanup


def _seed_items(db):
    seed_item(db, "wallet", room_name="kitchen", emb_seed=1)
    seed_item(db, "keys", room_name="bedroom", emb_seed=2)


def _get_events(client, event_type: str) -> list[dict]:
    """Return all received events of a given type."""
    return [e["args"][0] for e in client.get_received() if e["name"] == event_type]


def _last_session_state(events: list[dict]) -> dict:
    states = [e for e in events if e["name"] == "session_state"]
    assert states, "missing session_state event"
    return states[-1]["args"][0]


def _ogg_audio() -> str:
    """Return base64-encoded minimal OGG-magic-byte audio payload."""
    import base64
    return base64.b64encode(b"OggS" + b"\x00" * 60).decode()


def _client_sid(client) -> str:
    from visual_memory.api.voice_session import _sessions
    if len(_sessions) == 1:
        return next(iter(_sessions.keys()))
    raise RuntimeError(f"Expected exactly one active voice session, got {len(_sessions)}")


# Tests

def test_connect_returning_user():
    """Returning user (items in DB) receives tts 'Ready.'"""
    client, sio, db, stub, cleanup = _make_ws_test_app(seed_fn=_seed_items)
    try:
        client.connect()
        received = client.get_received()
        tts_events = [e for e in received if e["name"] == "tts"]
        assert tts_events, "no tts event on connect"
        narration = tts_events[0]["args"][0].get("narration", "")
        assert "ready" in narration.lower(), f"unexpected narration: {narration!r}"
        session_state = _last_session_state(received)
        assert session_state.get("current_mode") == "idle"
    finally:
        cleanup()


def test_connect_empty_db_triggers_onboarding():
    """First-time user (empty DB) receives welcome onboarding TTS."""
    client, sio, db, stub, cleanup = _make_ws_test_app(empty_db=True)
    try:
        client.connect()
        received = client.get_received()
        tts_events = [e for e in received if e["name"] == "tts"]
        assert tts_events, "no tts event on connect"
        narration = tts_events[0]["args"][0].get("narration", "")
        assert "welcome" in narration.lower(), f"unexpected onboarding narration: {narration!r}"
        assert tts_events[0]["args"][0].get("next_state") == "onboarding_teach"
        session_state = _last_session_state(received)
        assert session_state.get("current_mode") == "onboarding_teach"
    finally:
        cleanup()


def test_connect_bad_api_key_rejected():
    """Connection with wrong key is rejected when API_KEY is set."""
    orig = os.environ.get("API_KEY", "")
    os.environ["API_KEY"] = "a" * 32
    try:
        import visual_memory.api.routes.voice_ws as _vws
        orig_key = _vws._API_KEY
        _vws._API_KEY = "a" * 32

        client, sio, db, stub, cleanup = _make_ws_test_app(
            seed_fn=_seed_items,
            ws_headers={"X-API-Key": "wrong" * 8},
        )
        try:
            # Connection should be rejected when bad key is provided.
            assert not client.is_connected()
        finally:
            _vws._API_KEY = orig_key
            cleanup()
    finally:
        if orig:
            os.environ["API_KEY"] = orig
        else:
            os.environ.pop("API_KEY", None)


def test_audio_find_intent():
    """Audio event with find intent returns action_result + tts."""
    client, sio, db, stub, cleanup = _make_ws_test_app(seed_fn=_seed_items)
    try:
        client.connect()
        client.get_received()  # consume connect tts

        stub.set_result("where is my wallet")
        client.emit("audio", {"audio": _ogg_audio()})
        received = client.get_received()

        transcription_events = [e for e in received if e["name"] == "transcription"]
        assert transcription_events, "no transcription event"

        tts_events = [e for e in received if e["name"] == "tts"]
        assert tts_events, "no tts event after find command"
        narration = tts_events[-1]["args"][0].get("narration", "")
        assert narration, "tts narration is empty"
    finally:
        cleanup()


def test_audio_scan_no_image_requests_image():
    """Scan intent without image emits control:request_image."""
    client, sio, db, stub, cleanup = _make_ws_test_app(seed_fn=_seed_items)
    try:
        client.connect()
        client.get_received()

        stub.set_result("scan")
        client.emit("audio", {"audio": _ogg_audio()})
        received = client.get_received()

        control_events = [e for e in received if e["name"] == "control"]
        assert control_events, "no control event"
        assert control_events[0]["args"][0].get("action") == "request_image"
    finally:
        cleanup()


def test_audio_scan_with_image_returns_results():
    """Scan intent with image runs pipeline and returns action_result."""
    import base64
    from PIL import Image
    import io

    client, sio, db, stub, cleanup = _make_ws_test_app(seed_fn=_seed_items)
    try:
        client.connect()
        client.get_received()

        # Configure scan stub to return a match
        from visual_memory.tests.scripts.test_harness import StubScanPipeline
        import visual_memory.api.pipelines as _pm
        scan_stub: StubScanPipeline = _pm._scan_pipeline
        scan_stub.run = lambda image, scan_id="", focal_length_px=0.0: {
            "matches": [{"label": "wallet", "narration": "Wallet to your left.", "similarity": 0.8}],
            "count": 1,
            "scan_id": scan_id,
            "is_dark": False,
            "darkness_level": 120.0,
        }

        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        stub.set_result("scan")
        client.emit("audio", {"audio": _ogg_audio(), "image": img_b64})
        received = client.get_received()

        action_events = [e for e in received if e["name"] == "action_result"]
        assert action_events, "no action_result event"
        assert action_events[0]["args"][0].get("type") == "scan"

        tts_events = [e for e in received if e["name"] == "tts"]
        assert tts_events, "no tts after scan"
    finally:
        cleanup()


def test_audio_remember_no_image_requests_image():
    """Teach intent without image emits control:request_image."""
    client, sio, db, stub, cleanup = _make_ws_test_app(seed_fn=_seed_items)
    try:
        client.connect()
        client.get_received()

        stub.set_result("teach my glasses")
        client.emit("audio", {"audio": _ogg_audio()})
        received = client.get_received()

        control_events = [e for e in received if e["name"] == "control"]
        assert control_events, "no control event for remember"
        assert control_events[0]["args"][0].get("action") == "request_image"
    finally:
        cleanup()


def test_awaiting_location_transitions_to_idle():
    """After scan->image->remember success, sending room name returns idle state."""
    import base64
    from PIL import Image
    import io
    import visual_memory.api.pipelines as _pm
    from visual_memory.api.voice_session import _sessions

    client, sio, db, stub, cleanup = _make_ws_test_app(seed_fn=_seed_items)
    try:
        client.connect()
        client.get_received()

        # Fake the session into awaiting_location via remember
        sid = _client_sid(client)
        if sid in _sessions:
            session = _sessions[sid]
            session.state = "awaiting_location"
            session.context["label"] = "wallet"

        stub.set_result("the kitchen")
        client.emit("audio", {"audio": _ogg_audio()})
        received = client.get_received()

        tts_events = [e for e in received if e["name"] == "tts"]
        assert tts_events, "no tts after location"
        # Should transition to idle
        last_state = tts_events[-1]["args"][0].get("next_state")
        assert last_state == "idle", f"expected idle, got {last_state!r}"
    finally:
        cleanup()


def test_navigate_advances_item_index():
    """Navigate event advances item_index and returns tts for next item."""
    from visual_memory.api.voice_session import _sessions

    client, sio, db, stub, cleanup = _make_ws_test_app(seed_fn=_seed_items)
    try:
        client.connect()
        client.get_received()

        # Seed scan results into the session
        sid = _client_sid(client)
        if sid in _sessions:
            session = _sessions[sid]
            session.state = "focused_on_item"
            session.context["scan_matches"] = [
                {"label": "wallet", "narration": "Wallet to your left."},
                {"label": "keys", "narration": "Keys straight ahead."},
            ]
            session.context["item_index"] = 0
            session.context["current_label"] = "wallet"

        client.emit("navigate", {"direction": "next"})
        received = client.get_received()

        tts_events = [e for e in received if e["name"] == "tts"]
        assert tts_events, "no tts after navigate"
        narration = tts_events[0]["args"][0].get("narration", "")
        assert "keys" in narration.lower() or "ahead" in narration.lower(), (
            f"expected keys narration, got: {narration!r}"
        )

        action_events = [e for e in received if e["name"] == "action_result"]
        assert action_events, "no action_result on navigate"
        assert action_events[0]["args"][0].get("type") == "item_focus"
    finally:
        cleanup()


def test_navigate_empty_session_returns_tts():
    """Navigate with no cached scan results returns a tts error message."""
    client, sio, db, stub, cleanup = _make_ws_test_app(seed_fn=_seed_items)
    try:
        client.connect()
        client.get_received()

        client.emit("navigate", {"direction": "next"})
        received = client.get_received()

        tts_events = [e for e in received if e["name"] == "tts"]
        assert tts_events, "expected tts for empty navigate"
        narration = tts_events[0]["args"][0].get("narration", "")
        assert "no scan" in narration.lower() or "navigate" in narration.lower()
    finally:
        cleanup()


def test_home_back_returns_idle_and_clears_scan_context():
    from visual_memory.api.voice_session import _sessions

    client, sio, db, stub, cleanup = _make_ws_test_app(seed_fn=_seed_items)
    try:
        client.connect()
        client.get_received()

        sid = _client_sid(client)
        session = _sessions[sid]
        session.state = "focused_on_item"
        session.context["scan_matches"] = [{"label": "wallet", "narration": "Wallet to your left."}]
        session.context["scan_id"] = "scan-123"
        session.context["item_index"] = 0
        session.context["current_label"] = "wallet"

        stub.set_result("go back")
        client.emit("audio", {"audio": _ogg_audio()})
        received = client.get_received()

        action = [e for e in received if e["name"] == "action_result"]
        assert action and action[-1]["args"][0].get("type") == "navigate_back"
        tts = [e for e in received if e["name"] == "tts"]
        assert tts and tts[-1]["args"][0].get("next_state") == "idle"
        state_event = _last_session_state(received)
        assert state_event.get("current_mode") == "idle"
        context = state_event.get("context") or {}
        assert context.get("scan_id", "") == ""
        assert context.get("label", "") == ""
    finally:
        cleanup()


def test_settings_command_returns_open_settings_and_idle():
    client, sio, db, stub, cleanup = _make_ws_test_app(seed_fn=_seed_items)
    try:
        client.connect()
        client.get_received()

        stub.set_result("settings")
        client.emit("audio", {"audio": _ogg_audio()})
        received = client.get_received()

        action = [e for e in received if e["name"] == "action_result"]
        assert action and action[-1]["args"][0].get("type") == "open_settings"
        tts = [e for e in received if e["name"] == "tts"]
        assert tts and "opening settings" in tts[-1]["args"][0].get("narration", "").lower()
        state_event = _last_session_state(received)
        assert state_event.get("current_mode") == "idle"
    finally:
        cleanup()


def test_disconnect_clears_session():
    """Disconnect removes the session from _sessions."""
    from visual_memory.api.voice_session import _sessions

    client, sio, db, stub, cleanup = _make_ws_test_app(seed_fn=_seed_items)
    try:
        client.connect()
        sid = _client_sid(client)
        assert sid in _sessions, "session not created on connect"
        client.disconnect()
        assert sid not in _sessions, "session not removed on disconnect"
    finally:
        cleanup()


def test_bad_audio_returns_error():
    """Empty audio payload returns error event (missing audio data)."""
    client, sio, db, stub, cleanup = _make_ws_test_app(seed_fn=_seed_items)
    try:
        client.connect()
        client.get_received()

        client.emit("audio", {"audio": ""})
        received = client.get_received()

        error_events = [e for e in received if e["name"] == "error"]
        assert error_events, "expected error event for empty audio"
        code = error_events[0]["args"][0].get("code", "")
        assert code == "transcription_failed"
    finally:
        cleanup()


def test_onboarding_happy_path_end_to_end():
    """Empty DB flow completes teach->scan->navigate->ask onboarding sequence."""
    import base64
    import io
    from PIL import Image
    import visual_memory.api.pipelines as _pm

    client, sio, db, stub, cleanup = _make_ws_test_app(empty_db=True)
    try:
        client.connect()
        connect_tts = _get_events(client, "tts")
        assert connect_tts and connect_tts[0].get("next_state") == "onboarding_teach"

        remember_stub: StubRememberPipeline = _pm._remember_pipeline
        remember_stub._success = True
        scan_stub: StubScanPipeline = _pm._scan_pipeline
        scan_stub.run = lambda image, scan_id="", focal_length_px=0.0: {
            "matches": [
                {"label": "wallet", "narration": "Wallet to your left.", "similarity": 0.8},
                {"label": "keys", "narration": "Keys ahead.", "similarity": 0.79},
            ],
            "count": 2,
            "scan_id": scan_id,
            "is_dark": False,
            "darkness_level": 100.0,
        }

        img = Image.new("RGB", (64, 64), color=(140, 140, 140))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        stub.set_result("teach my wallet")
        client.emit("audio", {"audio": _ogg_audio(), "image": img_b64})
        tts_events = _get_events(client, "tts")
        assert tts_events and tts_events[-1].get("next_state") == "awaiting_location"

        stub.set_result("kitchen")
        client.emit("audio", {"audio": _ogg_audio()})
        tts_events = _get_events(client, "tts")
        assert tts_events and tts_events[-1].get("next_state") == "onboarding_teach"

        stub.set_result("teach my keys")
        client.emit("audio", {"audio": _ogg_audio(), "image": img_b64})
        tts_events = _get_events(client, "tts")
        assert tts_events and tts_events[-1].get("next_state") == "awaiting_location"

        stub.set_result("bedroom")
        client.emit("audio", {"audio": _ogg_audio()})
        tts_events = _get_events(client, "tts")
        assert tts_events and tts_events[-1].get("next_state") == "onboarding_await_scan"

        stub.set_result("scan")
        client.emit("audio", {"audio": _ogg_audio(), "image": img_b64})
        tts_events = _get_events(client, "tts")
        assert tts_events and tts_events[-1].get("next_state") == "focused_on_item"

        client.emit("navigate", {"direction": "next"})
        tts_events = _get_events(client, "tts")
        assert tts_events, "missing navigate tts"
        assert "now ask me where you left your" in tts_events[-1].get("narration", "").lower()

        stub.set_result("where is my wallet")
        client.emit("audio", {"audio": _ogg_audio()})
        tts_events = _get_events(client, "tts")
        assert tts_events and tts_events[-1].get("next_state") == "idle"
        assert "you are all set" in tts_events[-1].get("narration", "").lower()
    finally:
        cleanup()


def test_hints_non_repeat_and_trigger_order():
    """Hints fire in configured order and are not repeated once emitted."""
    from visual_memory.api.voice_session import _sessions
    import visual_memory.api.pipelines as _pm

    client, sio, db, stub, cleanup = _make_ws_test_app(seed_fn=_seed_items)
    try:
        client.connect()
        client.get_received()

        sid = _client_sid(client)
        session = _sessions[sid]
        session.state = "focused_on_item"
        session.context["scan_matches"] = [{"label": "wallet", "narration": "Wallet to your left."}]
        session.context["item_index"] = 0
        session.context["current_label"] = "wallet"
        session.context["scan_id"] = "scan-1"
        session.context["scan_count"] = 0
        session.context["teach_count"] = 0
        session.context["hints_given"] = set()

        scan_stub: StubScanPipeline = _pm._scan_pipeline
        scan_stub.run = lambda image, scan_id="", focal_length_px=0.0: {
            "matches": [{"label": "wallet", "narration": "Wallet to your left.", "similarity": 0.8}],
            "count": 1,
            "scan_id": scan_id,
            "is_dark": False,
            "darkness_level": 100.0,
        }

        def _expect_hint_after_scan(scan_count: int, expected_hint: str | None):
            stub.set_result("scan")
            client.emit("audio", {"audio": _ogg_audio(), "image": _tiny_jpeg_b64()})
            tts = _get_events(client, "tts")
            assert tts, f"missing tts after scan_count {scan_count}"
            narration = tts[-1].get("narration", "").lower()
            if expected_hint is None:
                assert "swipe right to browse each detected item one at a time." not in narration
                assert "if a detection was wrong, just say wrong." not in narration
                assert "double tap to save a location while browsing items." not in narration
            else:
                assert expected_hint in narration, f"expected hint {expected_hint!r} in {narration!r}"
            assert narration.count("swipe right to browse each detected item one at a time.") <= 1
            assert narration.count("if a detection was wrong, just say wrong.") <= 1
            assert narration.count("double tap to save a location while browsing items.") <= 1

        _expect_hint_after_scan(1, "swipe right to browse each detected item one at a time.")
        _expect_hint_after_scan(2, "if a detection was wrong, just say wrong.")
        _expect_hint_after_scan(3, None)
        _expect_hint_after_scan(4, None)
        _expect_hint_after_scan(5, "double tap to save a location while browsing items.")
        _expect_hint_after_scan(6, None)

        # Trigger teach-based hints in order
        remember_stub: StubRememberPipeline = _pm._remember_pipeline
        remember_stub._success = True

        def _expect_hint_after_teach(room: str, expected_hint: str | None):
            stub.set_result("teach my book")
            client.emit("audio", {"audio": _ogg_audio(), "image": _tiny_jpeg_b64()})
            tts = _get_events(client, "tts")
            assert tts and tts[-1].get("next_state") == "awaiting_location"
            stub.set_result(room)
            client.emit("audio", {"audio": _ogg_audio()})
            tts = _get_events(client, "tts")
            assert tts, "missing tts after location save"
            narration = tts[-1].get("narration", "").lower()
            if expected_hint is None:
                assert "you can ask questions about your items, like what is in a receipt." not in narration
                assert "try saying export this when looking at a document." not in narration
            else:
                assert expected_hint in narration, f"expected hint {expected_hint!r} in {narration!r}"

        _expect_hint_after_teach("office", None)
        _expect_hint_after_teach("hallway", "you can ask questions about your items, like what is in a receipt.")
        _expect_hint_after_teach("garage", "try saying export this when looking at a document.")

        # No repeats after all hints are delivered
        stub.set_result("scan")
        client.emit("audio", {"audio": _ogg_audio(), "image": _tiny_jpeg_b64()})
        narration = _get_events(client, "tts")[-1].get("narration", "").lower()
        assert "swipe right to browse each detected item one at a time." not in narration
        assert "if a detection was wrong, just say wrong." not in narration
        assert "double tap to save a location while browsing items." not in narration
        assert "you can ask questions about your items, like what is in a receipt." not in narration
        assert "try saying export this when looking at a document." not in narration
    finally:
        cleanup()


def _tiny_jpeg_b64() -> str:
    import base64
    import io
    from PIL import Image
    img = Image.new("RGB", (32, 32), color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def test_audio_invalid_image_payload_returns_bad_payload():
    """Malformed image payload is rejected before dispatch."""
    client, sio, db, stub, cleanup = _make_ws_test_app(seed_fn=_seed_items)
    try:
        client.connect()
        client.get_received()
        stub.set_result("scan")
        client.emit("audio", {"audio": _ogg_audio(), "image": "not-base64!!!"})
        errors = _get_events(client, "error")
        assert errors, "expected bad_payload error"
        assert errors[-1].get("code") == "bad_payload"
        assert "invalid image payload" in errors[-1].get("message", "").lower()
    finally:
        cleanup()


def test_audio_oversize_image_payload_returns_bad_payload():
    """Image bytes over 50MB are rejected with bad_payload."""
    import base64
    client, sio, db, stub, cleanup = _make_ws_test_app(seed_fn=_seed_items)
    try:
        client.connect()
        client.get_received()
        stub.set_result("scan")
        over = base64.b64encode(b"\x89PNG" + b"\x00" * (50 * 1024 * 1024 + 1)).decode()
        client.emit("audio", {"audio": _ogg_audio(), "image": over})
        errors = _get_events(client, "error")
        assert errors, "expected bad_payload for oversize image"
        assert errors[-1].get("code") == "bad_payload"
        assert "50mb limit" in errors[-1].get("message", "").lower()
    finally:
        cleanup()


for name, fn in [
    ("ws:connect_returning_user", test_connect_returning_user),
    ("ws:connect_onboarding", test_connect_empty_db_triggers_onboarding),
    ("ws:connect_bad_key", test_connect_bad_api_key_rejected),
    ("ws:audio_find", test_audio_find_intent),
    ("ws:audio_scan_no_image", test_audio_scan_no_image_requests_image),
    ("ws:audio_scan_with_image", test_audio_scan_with_image_returns_results),
    ("ws:audio_remember_no_image", test_audio_remember_no_image_requests_image),
    ("ws:awaiting_location", test_awaiting_location_transitions_to_idle),
    ("ws:navigate_advance", test_navigate_advances_item_index),
    ("ws:navigate_empty", test_navigate_empty_session_returns_tts),
    ("ws:home_back_clears_context", test_home_back_returns_idle_and_clears_scan_context),
    ("ws:settings_action", test_settings_command_returns_open_settings_and_idle),
    ("ws:disconnect_clears_session", test_disconnect_clears_session),
    ("ws:bad_audio_error", test_bad_audio_returns_error),
    ("ws:onboarding_happy_path", test_onboarding_happy_path_end_to_end),
    ("ws:hints_order_non_repeat", test_hints_non_repeat_and_trigger_order),
    ("ws:invalid_image_payload", test_audio_invalid_image_payload_returns_bad_payload),
    ("ws:oversize_image_payload", test_audio_oversize_image_payload_returns_bad_payload),
]:
    _runner.run(name, fn)

sys.exit(_runner.summary())
