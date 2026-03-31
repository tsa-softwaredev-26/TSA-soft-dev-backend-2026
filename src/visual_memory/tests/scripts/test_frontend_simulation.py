"""
Frontend-like WebSocket simulations with client-side state mirroring.
"""
from __future__ import annotations

import base64
import io
import os
import shutil
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

os.environ["ENABLE_DEPTH"] = "0"

from PIL import Image

from visual_memory.engine.model_registry import registry
from visual_memory.tests.scripts.test_harness import (
    StubRememberPipeline,
    StubScanPipeline,
    StubSpeechRecognizer,
    TestRunner,
    seed_item,
)

_runner = TestRunner("frontend_simulation")


def _make_ws_test_app(seed_fn=None, empty_db: bool = False):
    import numpy as np
    from flask import Flask
    from flask_socketio import SocketIO

    import visual_memory.api.pipelines as _pm
    import visual_memory.api.routes.transcribe as _tr
    from visual_memory.api.routes.voice_ws import register_events
    from visual_memory.api.voice_session import _sessions
    from visual_memory.config.settings import Settings
    from visual_memory.config.user_settings import UserSettings
    from visual_memory.database.store import DatabaseStore
    from visual_memory.learning.feedback_store import FeedbackStore

    tmp_dir = Path(".test-artifacts") / f"frontend-sim-{uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(tmp_dir / "test.db")
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

    stub_recognizer = StubSpeechRecognizer()
    stub_recognizer.set_result("scan")
    original_recognizer = registry._whisper_recognizer
    registry._whisper_recognizer = stub_recognizer

    original_loader = _tr.load_audio_bytes
    _tr.load_audio_bytes = lambda b, target_sr=16000: (np.zeros(16000, dtype="float32"), 16000)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
    sio = SocketIO(app, async_mode="threading")
    register_events(sio)
    client = sio.test_client(app)

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
        shutil.rmtree(str(tmp_dir), ignore_errors=True)

    return client, db, stub_recognizer, cleanup


def _seed_items(db):
    seed_item(db, "wallet", room_name="kitchen", emb_seed=1)
    seed_item(db, "keys", room_name="bedroom", emb_seed=2)


def _ogg_audio() -> str:
    return base64.b64encode(b"OggS" + b"\x00" * 60).decode()


def _tiny_jpeg_b64() -> str:
    img = Image.new("RGB", (48, 48), color=(130, 130, 130))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


class FrontendSimulator:
    def __init__(self, client, recognizer: StubSpeechRecognizer):
        self.client = client
        self.recognizer = recognizer
        self.current_mode = "idle"
        self.context: dict[str, Any] = {}
        self.scan_id = ""
        self.events: list[dict] = []

    def _consume(self) -> list[dict]:
        received = self.client.get_received()
        self.events.extend(received)
        for event in received:
            payload = event["args"][0] if event.get("args") else {}
            if event["name"] == "session_state":
                self.current_mode = payload.get("current_mode", self.current_mode)
                self.context = dict(payload.get("context") or {})
                if self.context.get("scan_id"):
                    self.scan_id = self.context.get("scan_id", "")
            elif event["name"] == "tts":
                next_state = payload.get("next_state")
                if next_state:
                    self.current_mode = next_state
            elif event["name"] == "action_result" and payload.get("type") == "scan":
                data = payload.get("data") or {}
                self.scan_id = data.get("scan_id", "") or self.scan_id
        return received

    def connect(self) -> list[dict]:
        self.client.connect()
        return self._consume()

    def speak(self, text: str, image_b64: str | None = None) -> list[dict]:
        self.recognizer.set_result(text)
        payload = {"audio": _ogg_audio()}
        if image_b64 is not None:
            payload["image"] = image_b64
        self.client.emit("audio", payload)
        return self._consume()

    def navigate(self, direction: str) -> list[dict]:
        self.client.emit("navigate", {"direction": direction})
        return self._consume()


def test_onboarding_home_idle_and_scan_state_tracking():
    import visual_memory.api.pipelines as _pm

    client, db, stub, cleanup = _make_ws_test_app(empty_db=True)
    sim = FrontendSimulator(client, stub)
    try:
        events = sim.connect()
        tts = [e for e in events if e["name"] == "tts"]
        assert tts and tts[-1]["args"][0].get("next_state") == "onboarding_teach"
        assert sim.current_mode == "onboarding_teach"

        remember_stub: StubRememberPipeline = _pm._remember_pipeline
        remember_stub._success = True
        scan_stub: StubScanPipeline = _pm._scan_pipeline
        scan_stub.run = lambda image, scan_id="", focal_length_px=0.0: {
            "matches": [
                {"label": "wallet", "narration": "Wallet to your left.", "similarity": 0.8},
                {"label": "keys", "narration": "Keys ahead.", "similarity": 0.79},
            ],
            "count": 2,
            "scan_id": "scan-onboard-1",
            "is_dark": False,
            "darkness_level": 100.0,
        }

        sim.speak("teach my wallet", image_b64=_tiny_jpeg_b64())
        assert sim.current_mode == "awaiting_location"
        sim.speak("kitchen")
        assert sim.current_mode == "onboarding_teach"

        sim.speak("teach my keys", image_b64=_tiny_jpeg_b64())
        assert sim.current_mode == "awaiting_location"
        sim.speak("office")
        assert sim.current_mode == "onboarding_await_scan"

        sim.speak("scan", image_b64=_tiny_jpeg_b64())
        assert sim.current_mode == "focused_on_item"
        assert sim.scan_id == "scan-onboard-1"

        sim.speak("home")
        assert sim.current_mode == "idle"
        assert sim.context.get("scan_id", "") == ""
        assert sim.context.get("label", "") == ""
    finally:
        cleanup()


def test_scan_navigation_settings_and_back_behavior():
    import visual_memory.api.pipelines as _pm

    client, db, stub, cleanup = _make_ws_test_app(seed_fn=_seed_items)
    sim = FrontendSimulator(client, stub)
    try:
        sim.connect()
        assert sim.current_mode == "idle"

        scan_stub: StubScanPipeline = _pm._scan_pipeline
        scan_stub.run = lambda image, scan_id="", focal_length_px=0.0: {
            "matches": [
                {"label": "wallet", "narration": "Wallet to your left.", "similarity": 0.8},
                {"label": "keys", "narration": "Keys ahead.", "similarity": 0.79},
            ],
            "count": 2,
            "scan_id": "scan-nav-1",
            "is_dark": False,
            "darkness_level": 100.0,
        }

        sim.speak("scan", image_b64=_tiny_jpeg_b64())
        assert sim.current_mode == "focused_on_item"
        assert sim.context.get("label") == "wallet"
        assert sim.context.get("scan_id") == "scan-nav-1"

        sim.navigate("next")
        assert sim.current_mode == "focused_on_item"
        assert sim.context.get("label") == "keys"

        events = sim.speak("settings")
        action = [e for e in events if e["name"] == "action_result"]
        assert action and action[-1]["args"][0].get("type") == "open_settings"
        assert sim.current_mode == "idle"

        sim.speak("scan", image_b64=_tiny_jpeg_b64())
        assert sim.current_mode == "focused_on_item"
        sim.speak("back")
        assert sim.current_mode == "idle"
    finally:
        cleanup()


def test_hints_and_context_driven_item_query_routing():
    import visual_memory.api.pipelines as _pm

    client, db, stub, cleanup = _make_ws_test_app(seed_fn=_seed_items)
    sim = FrontendSimulator(client, stub)
    try:
        sim.connect()
        scan_stub: StubScanPipeline = _pm._scan_pipeline
        scan_stub.run = lambda image, scan_id="", focal_length_px=0.0: {
            "matches": [{"label": "wallet", "narration": "Wallet to your left.", "similarity": 0.8}],
            "count": 1,
            "scan_id": "scan-hints-1",
            "is_dark": False,
            "darkness_level": 100.0,
        }
        remember_stub: StubRememberPipeline = _pm._remember_pipeline
        remember_stub._success = True

        sim.speak("teach my book", image_b64=_tiny_jpeg_b64())
        sim.speak("hallway")
        teach2_events = sim.speak("teach my receipt", image_b64=_tiny_jpeg_b64())
        teach2_events += sim.speak("office")
        teach_narrations = " ".join(
            (e["args"][0].get("narration", "").lower() for e in teach2_events if e["name"] == "tts")
        )
        assert "you can ask questions about your items, like what is in a receipt." in teach_narrations

        scan1 = sim.speak("scan", image_b64=_tiny_jpeg_b64())
        scan_text = " ".join((e["args"][0].get("narration", "").lower() for e in scan1 if e["name"] == "tts"))
        assert "swipe right to browse each detected item one at a time." in scan_text

        scan2 = sim.speak("scan", image_b64=_tiny_jpeg_b64())
        scan2_text = " ".join((e["args"][0].get("narration", "").lower() for e in scan2 if e["name"] == "tts"))
        assert "if a detection was wrong, just say wrong." in scan2_text

        assert sim.current_mode == "focused_on_item"
        assert sim.context.get("scan_id") == "scan-hints-1"
        assert sim.context.get("label") == "wallet"

        query_events = sim.speak("what does this say")
        actions = [e for e in query_events if e["name"] == "action_result"]
        assert actions and actions[-1]["args"][0].get("type") == "item_ask"
        assert sim.current_mode == "focused_on_item"
    finally:
        cleanup()


def main() -> int:
    for name, fn in [
        ("frontend_sim:onboarding_home_idle_scan_state", test_onboarding_home_idle_and_scan_state_tracking),
        ("frontend_sim:scan_navigation_settings_back", test_scan_navigation_settings_and_back_behavior),
        ("frontend_sim:hints_and_context_routing", test_hints_and_context_driven_item_query_routing),
    ]:
        _runner.run(name, fn)
    return _runner.summary()


if __name__ == "__main__":
    sys.exit(main())
