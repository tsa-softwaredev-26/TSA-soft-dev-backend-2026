"""
Integration tests for /voice command classification and state-aware routing.
"""
from __future__ import annotations

import io
import os
import sys

from PIL import Image

os.environ["ENABLE_DEPTH"] = "0"

import visual_memory.api.pipelines as _pm
from visual_memory.api.routes.voice import (
    classify_command,
    classify_item_intent,
    extract_teach_label,
    voice_bp,
)
from visual_memory.tests.scripts.test_harness import (
    TestRunner,
    assert_status,
    make_test_app,
    seed_item,
)

_runner = TestRunner("voice_router")


def _seed(db):
    seed_item(db, "wallet", ocr_text="RFID Blocking", room_name="kitchen", emb_seed=1)
    seed_item(db, "receipt", ocr_text="Office Chair $299", room_name="office", emb_seed=2)


client, db, cleanup = make_test_app([voice_bp], seed_fn=_seed)


def _make_jpeg() -> bytes:
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_extract_teach_label_patterns():
    assert extract_teach_label("remember this as house keys") == "house keys"
    assert extract_teach_label("this is my laptop") == "laptop"
    assert extract_teach_label("save this as tax return") == "tax return"
    assert extract_teach_label("where is my wallet") is None


def test_classify_item_intent():
    assert classify_item_intent("what does this say") == "read_ocr"
    assert classify_item_intent("export this") == "export_ocr"
    assert classify_item_intent("describe this") == "describe"
    assert classify_item_intent("rename this to work wallet") == "rename"
    assert classify_item_intent("where is this") == "find"
    assert classify_item_intent("nonsense command") == "read_ocr"


def test_classify_command_idle():
    assert classify_command("scan", {"current_mode": "idle"})["command"] == "scan"
    assert classify_command("start scan", {"current_mode": "idle"})["command"] == "scan"
    assert classify_command("scan now", {"current_mode": "idle"})["command"] == "scan"
    assert classify_command("find", {"current_mode": "idle"}) == {"command": "find", "query": ""}
    assert classify_command("ask", {"current_mode": "idle"}) == {"command": "ask"}
    assert classify_command("settings", {"current_mode": "idle"}) == {"command": "open_settings"}
    assert classify_command("open settings", {"current_mode": "idle"}) == {"command": "open_settings"}
    assert classify_command("remember this as gym bag", {"current_mode": "idle"}) == {
        "command": "remember",
        "label": "gym bag",
    }
    assert classify_command("where is my wallet", {"current_mode": "idle"})["command"] == "find"
    blocked_item = classify_command("what is written on this paper", {"current_mode": "idle"})
    assert blocked_item.get("error") == "command_unavailable"
    assert blocked_item.get("requested_command") == "item_ask"
    assert classify_command("tell me about my keys", {"current_mode": "idle"}) == {
        "command": "ask",
        "query_type": "general",
    }
    assert classify_command("", {"current_mode": "idle"}) == {"command": "ask", "query_type": "unknown"}


def test_classify_command_state_aware():
    waiting = classify_command("kitchen", {"current_mode": "awaiting_location"})
    assert waiting == {"command": "set_location", "location": "kitchen"}
    focused = classify_command("copy this", {"current_mode": "focused_on_item"})
    assert focused == {"command": "item_ask", "intent": "export_ocr"}
    blocked = classify_command("settings", {"current_mode": "focused_on_item"})
    assert blocked.get("error") == "command_unavailable"
    assert blocked.get("requested_command") == "open_settings"
    assert "describe/read" in blocked.get("available_commands", [])
    nav = classify_command("next", {"current_mode": "focused_on_item"})
    assert nav == {"command": "navigate_next"}
    blocked_nav = classify_command("next", {"current_mode": "idle"})
    assert blocked_nav.get("error") == "command_unavailable"


def test_voice_idle_find_request_type():
    resp = client.post("/voice", json={"text": "find wallet", "state": {"current_mode": "idle"}})
    assert_status(resp, 200)
    data = resp.get_json()
    assert data.get("request_type") == "find"
    assert data.get("classification", {}).get("command") == "find"
    assert data.get("result", {}).get("found") is True


def test_voice_focused_item_routes_item_ask():
    resp = client.post(
        "/voice",
        json={
            "text": "what does this say",
            "state": {"current_mode": "focused_on_item", "context": {"scan_id": "s1", "label": "wallet"}},
        },
    )
    assert_status(resp, 200)
    data = resp.get_json()
    assert data.get("request_type") == "item_ask"
    assert data.get("classification", {}).get("intent") == "read_ocr"
    assert data.get("result", {}).get("action") == "read_ocr"


def test_voice_command_wrong_state_returns_error_payload():
    resp = client.post("/voice", json={"text": "settings", "state": {"current_mode": "focused_on_item"}})
    assert_status(resp, 200)
    data = resp.get_json()
    assert data.get("request_type") == "error"
    assert data.get("result", {}).get("error") == "command_unavailable"
    assert data.get("result", {}).get("current_mode") == "focused_on_item"
    assert "describe/read" in data.get("result", {}).get("available_commands", [])


def test_voice_idle_command_variants_from_dataset():
    for utterance, expected in [
        ("scan", "scan"),
        ("start scan", "scan"),
        ("scan now", "scan"),
        ("find", "find"),
        ("ask", "ask"),
        ("settings", "open_settings"),
        ("open settings", "open_settings"),
    ]:
        out = classify_command(utterance, {"current_mode": "idle"})
        assert out.get("command") == expected


def test_voice_awaiting_location_sets_location():
    resp = client.post(
        "/voice",
        json={
            "text": "bedroom",
            "state": {"current_mode": "awaiting_location", "context": {"label": "wallet"}},
        },
    )
    assert_status(resp, 200)
    data = resp.get_json()
    assert data.get("request_type") == "set_location"
    assert data.get("next_state") == "idle"
    assert data.get("result", {}).get("saved") is True


def test_voice_teach_phrase_routes_remember():
    resp = client.post(
        "/voice",
        data={
            "text": "remember this as blue mug",
            "image": (io.BytesIO(_make_jpeg()), "frame.jpg"),
            "state": '{"current_mode":"idle"}',
        },
        content_type="multipart/form-data",
    )
    assert_status(resp, 200)
    data = resp.get_json()
    assert data.get("request_type") == "remember"
    assert data.get("classification", {}).get("label") == "blue mug"
    assert data.get("next_state") == "awaiting_location"


def test_voice_gibberish_defaults_to_ask():
    resp = client.post("/voice", json={"text": "blargle zorp", "state": {"current_mode": "idle"}})
    assert_status(resp, 200)
    data = resp.get_json()
    assert data.get("request_type") == "ask"
    assert data.get("classification", {}).get("query_type") == "unknown"


def test_voice_invalid_json_returns_400():
    resp = client.post(
        "/voice",
        data='{"text":',
        content_type="application/json",
    )
    assert_status(resp, 400)


def test_voice_invalid_base64_audio_returns_400():
    resp = client.post(
        "/voice",
        json={
            "audio": "not-base64!!!",
            "state": {"current_mode": "idle"},
        },
    )
    assert_status(resp, 400)
    data = resp.get_json()
    assert data.get("error") == "invalid audio payload"


for name, fn in [
    ("voice_router:teach_label_patterns", test_extract_teach_label_patterns),
    ("voice_router:item_intent_classify", test_classify_item_intent),
    ("voice_router:command_idle_classify", test_classify_command_idle),
    ("voice_router:command_state_classify", test_classify_command_state_aware),
    ("voice_router:idle_find_route", test_voice_idle_find_request_type),
    ("voice_router:focused_item_route", test_voice_focused_item_routes_item_ask),
    ("voice_router:wrong_state_error_payload", test_voice_command_wrong_state_returns_error_payload),
    ("voice_router:dataset_command_coverage", test_voice_idle_command_variants_from_dataset),
    ("voice_router:awaiting_location_route", test_voice_awaiting_location_sets_location),
    ("voice_router:teach_phrase_route", test_voice_teach_phrase_routes_remember),
    ("voice_router:gibberish_defaults_to_ask", test_voice_gibberish_defaults_to_ask),
    ("voice_router:invalid_json", test_voice_invalid_json_returns_400),
    ("voice_router:invalid_audio_payload", test_voice_invalid_base64_audio_returns_400),
]:
    _runner.run(name, fn)

cleanup()
_pm._database = None
_pm._scan_pipeline = None
_pm._remember_pipeline = None
_pm._feedback_store = None
_pm._user_settings = None
sys.exit(_runner.summary())
