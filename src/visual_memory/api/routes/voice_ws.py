"""Stateful WebSocket voice session event handlers."""
from __future__ import annotations

import base64
import io
import os
import re
import time

from flask import request
from flask_socketio import SocketIO, emit

from visual_memory.api.pipelines import (
    get_database,
    get_feedback_store,
    get_scan_pipeline,
)
from visual_memory.api.narration import (
    ONBOARDING_AWAIT_SCAN_PROMPT,
    ONBOARDING_TEACH_LISTENING_PROMPT,
)
from visual_memory.api.routes.transcribe import transcribe_audio_bytes
from visual_memory.api.voice_session import VoiceSession, clear_session, get_session
from visual_memory.utils import get_logger
from visual_memory.utils.logger import LogTag
from visual_memory.utils.voice_state_context import build_state_contract

_log = get_logger(__name__)
_API_KEY = os.environ.get("API_KEY", "")
_ROOM_PREFIX = re.compile(r"^(in (the|my) |the |my )", re.IGNORECASE)
_MAX_WS_MEDIA_BYTES = 50 * 1024 * 1024

# Hint content and when to fire them (counter_key, threshold count)
_HINTS: dict[str, str] = {
    "navigate": "Swipe right to browse each detected item one at a time.",
    "feedback": "If a detection was wrong, just say wrong.",
    "ask": "You can ask questions about your items, like what is in a receipt.",
    "export": "Try saying export this when looking at a document.",
    "double_tap": "Double tap to save a location while browsing items.",
}
_HINT_TRIGGERS: dict[str, tuple[str, int]] = {
    "navigate":   ("scan_count", 1),
    "feedback":   ("scan_count", 2),
    "ask":        ("teach_count", 2),
    "export":     ("teach_count", 3),
    "double_tap": ("scan_count", 5),
}


# Helpers

def _session_state_payload(session: VoiceSession) -> dict:
    return {
        "current_mode": session.state,
        "context": {
            "scan_id": session.context.get("scan_id", ""),
            "label": session.context.get("current_label", ""),
            "item_index": session.context.get("item_index"),
            "onboarding_phase": session.context.get("onboarding_phase", ""),
        },
    }


def _emit_session_state(session: VoiceSession) -> None:
    emit("session_state", _session_state_payload(session))


def _clear_focus_context(session: VoiceSession) -> None:
    for key in ("scan_matches", "scan_id", "item_index", "current_label"):
        session.context.pop(key, None)


def _listening_prompt(session: VoiceSession) -> str:
    """Return a short listening hint appropriate for the current session state.

    Shown or spoken while the user is recording after pressing the Spaitra button.
    Keeps the user oriented without requiring them to remember available commands.
    """
    state = session.state
    if state == "onboarding_teach":
        return ONBOARDING_TEACH_LISTENING_PROMPT
    if state == "onboarding_await_scan":
        return ONBOARDING_AWAIT_SCAN_PROMPT
    if state == "awaiting_location":
        return "Which room is this?"
    if state == "awaiting_image":
        return "Hold your phone up."
    if state == "awaiting_confirmation":
        return "Say yes to confirm, or no to cancel."
    if state == "focused_on_item":
        label = session.context.get("current_label", "")
        if label:
            return f"Focused on {label}. Ask about it, or say wrong if it was incorrect."
        return "Ask about the item, or say wrong if the detection was incorrect."
    return "Scan, teach something new, or ask me anything."


def _increment(session: VoiceSession, key: str) -> None:
    session.context[key] = session.context.get(key, 0) + 1


def _get_hint(session: VoiceSession) -> str | None:
    """Return the next due hint narration, or None if none are ready."""
    given: set[str] = session.context.setdefault("hints_given", set())
    for hint_type, (counter_key, threshold) in _HINT_TRIGGERS.items():
        if hint_type in given:
            continue
        if session.context.get(counter_key, 0) >= threshold:
            given.add(hint_type)
            return _HINTS[hint_type]
    return None


def _decode_audio(value) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if not isinstance(value, str):
        return b""
    encoded = value.strip()
    if "," in encoded and encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    try:
        return base64.b64decode(encoded, validate=True)
    except Exception:
        return b""


def _decode_image(value) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if not isinstance(value, str):
        return None
    encoded = value.strip()
    if not encoded:
        return None
    if "," in encoded and encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    try:
        return base64.b64decode(encoded, validate=True)
    except Exception:
        return None


class _BytesFileStorage:
    """Minimal FileStorage-compatible wrapper around raw image bytes.

    Allows the WS handler to pass image bytes into process_remember_request
    and process_scan_request without requiring an actual Flask request context.
    """

    def __init__(self, data: bytes, filename: str = "upload.jpg"):
        self.stream = io.BytesIO(data)
        self.filename = filename

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            f.write(self.stream.getvalue())


def _normalize_room_name(text: str) -> str:
    return _ROOM_PREFIX.sub("", (text or "").strip()).strip().lower()


def _extract_label(command_text: str, state_context: dict | None = None) -> str:
    """Extract the item label from a teach/remember command via Ollama or regex."""
    try:
        from visual_memory.utils.ollama_utils import extract_item_intent
        label = extract_item_intent(command_text, state_context=state_context)
        if label:
            return label.strip().lower()
    except Exception:
        pass
    # Regex fallback: strip common teach prefixes
    cleaned = re.sub(
        r"^(teach|remember|save|learn|this is|it's|it is)\s+(me\s+)?(my\s+|a\s+|the\s+)?",
        "",
        command_text.strip(),
        flags=re.IGNORECASE,
    )
    return cleaned.strip().lower() or command_text.strip().lower()


def _extract_find_query(command_text: str, state_context: dict | None = None) -> str:
    try:
        from visual_memory.utils.ollama_utils import extract_search_term
        term = extract_search_term(command_text, state_context=state_context)
        if term:
            return term.strip()
    except Exception:
        pass
    return command_text.strip()


def _normalize_command_type(query: str, has_image: bool) -> str:
    q = (query or "").strip().lower()
    if not q:
        return "ask"
    if any(t in q for t in ("settings", "preferences")):
        return "open_settings"
    if any(t in q for t in ("go back", "back", "home", "cancel")):
        return "navigate_back"
    if has_image and any(t in q for t in ("remember", "save", "learn", "teach", "this is")):
        return "remember"
    if any(t in q for t in ("scan", "what is around", "what do you see", "what's around")):
        return "scan"
    if any(t in q for t in ("teach", "remember", "save", "learn", "this is")):
        return "remember"
    if any(t in q for t in ("where", "find", "last seen", "location of", "where did i")):
        return "find"
    return "ask"


# Pipeline helpers that bypass Flask request context

def _run_scan(image_bytes: bytes, focal_length: float | None) -> tuple[dict, int]:
    from visual_memory.api.routes.scan import process_scan_request
    image_file = _BytesFileStorage(image_bytes, "frame.jpg")
    focal_raw = str(focal_length) if focal_length is not None else ""
    return process_scan_request(image_file=image_file, focal_length_raw=focal_raw)


def _run_remember(image_bytes: bytes, label: str) -> tuple[dict, int]:
    from visual_memory.api.routes.remember import process_remember_request
    image_file = _BytesFileStorage(image_bytes, "teach.jpg")
    return process_remember_request(prompt=label, image_file=image_file, image_files=[])


def _run_find(command_text: str, state_context: dict | None = None) -> tuple[dict, int]:
    from visual_memory.api.routes.find import _format_sighting, _ocr_content_match, build_narration
    from visual_memory.api.pipelines import get_settings
    search_label = _extract_find_query(command_text, state_context=state_context)
    if not search_label:
        return {"error": "could not determine what to find"}, 400
    db = get_database()
    row = db.get_last_sighting(search_label)
    resolved_label = search_label
    matched_by = "exact"
    if row is None:
        settings = get_settings()
        ocr_match = _ocr_content_match(search_label, settings.text_similarity_threshold)
        if ocr_match:
            row = db.get_last_sighting(ocr_match)
            resolved_label = ocr_match
            matched_by = "ocr"
    if row is None:
        return {
            "label": search_label,
            "found": False,
            "narration": f"I could not find {search_label} in your memory.",
            "sightings": [],
        }, 200
    sighting = _format_sighting(row)
    return {
        "label": search_label,
        "found": True,
        "matched_label": resolved_label,
        "matched_by": matched_by,
        "last_sighting": sighting,
        "sightings": [sighting],
        "narration": build_narration(resolved_label, sighting),
    }, 200


# Sub-handlers called from _dispatch

def _handle_scan(session: VoiceSession, image_bytes: bytes, focal_length: float | None) -> None:
    onboarding_phase = session.context.get("onboarding_phase", "")
    result, status = _run_scan(image_bytes, focal_length)
    if status != 200:
        emit("error", {"code": "scan_failed", "message": result.get("error", "Scan failed.")})
        return

    matches = result.get("matches", [])
    scan_id = result.get("scan_id", "")
    _increment(session, "scan_count")

    emit("action_result", {"type": "scan", "data": result})

    if not matches:
        if onboarding_phase == "await_scan":
            narration = "I did not spot either item. Make sure they are in view and try again."
            session.state = "onboarding_await_scan"
        else:
            narration = "I did not find anything familiar."
            session.state = "idle"
        emit("tts", {"narration": narration, "next_state": session.state})
        return

    # Cache matches for navigate and item queries
    session.context["scan_matches"] = matches
    session.context["scan_id"] = scan_id
    session.context["item_index"] = 0
    session.context["current_label"] = matches[0].get("label", "")
    session.state = "focused_on_item"

    narration_parts = [m.get("narration") or m.get("label", "") for m in matches]
    combined = ". ".join(p for p in narration_parts if p)

    if onboarding_phase == "await_scan":
        # Onboarding scan succeeded - move to swipe intro
        ask_label = session.context.get("onboarding_item_1", "") or (matches[0].get("label", "") if matches else "")
        session.context["onboarding_phase"] = "ask"
        session.context["onboarding_ask_label"] = ask_label
        emit("tts", {
            "narration": f"{combined}. Now swipe right to browse each item.",
            "next_state": "focused_on_item",
        })
        return

    hint = _get_hint(session)
    if hint:
        combined = f"{combined}. {hint}"
    emit("tts", {"narration": combined, "next_state": "focused_on_item"})


def _handle_remember(session: VoiceSession, image_bytes: bytes, label: str) -> None:
    onboarding_phase = session.context.get("onboarding_phase", "")
    result, status = _run_remember(image_bytes, label)
    emit("action_result", {"type": "remember", "data": result})

    if status != 200:
        emit("error", {"code": "remember_failed", "message": result.get("error", "Remember failed.")})
        return

    success = isinstance(result, dict) and result.get("success", False)
    result_data = result.get("result") if isinstance(result, dict) else None

    if not success:
        is_dark = isinstance(result, dict) and result.get("is_dark", False)
        is_blurry = isinstance(result_data, dict) and result_data.get("is_blurry", False)
        if is_dark:
            narration = "Image is too dark. Increase lighting and try again."
        elif is_blurry:
            narration = "Hold steady and try again."
        else:
            narration = "Could not detect the item. Try a different angle or better lighting."
        emit("tts", {"narration": narration, "next_state": session.state})
        return

    _increment(session, "teach_count")
    session.context["label"] = label
    session.state = "awaiting_location"

    if onboarding_phase == "teach_1":
        session.context["onboarding_item_1"] = label
        emit("tts", {
            "narration": f"{label.capitalize()} saved. Which room is this?",
            "next_state": "awaiting_location",
        })
    elif onboarding_phase == "teach_2":
        session.context["onboarding_item_2"] = label
        emit("tts", {
            "narration": f"{label.capitalize()} saved. Which room is this?",
            "next_state": "awaiting_location",
        })
    else:
        emit("tts", {
            "narration": f"{label.capitalize()} saved. Which room is this?",
            "next_state": "awaiting_location",
        })


def _handle_location(session: VoiceSession, command_text: str) -> None:
    """Handle the awaiting_location state: save sighting and advance state."""
    onboarding_phase = session.context.get("onboarding_phase", "")
    label = session.context.get("label", "")
    location = _normalize_room_name(command_text)

    if not location:
        emit("tts", {"narration": "What room is this?", "next_state": "awaiting_location"})
        return

    if label:
        try:
            db = get_database()
            if db.get_items_metadata(label=label):
                db.add_sighting(label=label, room_name=location)
        except Exception as exc:
            _log.warning({"event": "ws_sighting_failed", "tag": LogTag.API, "error": str(exc)})

    emit("action_result", {"type": "set_location", "data": {"label": label, "location": location}})

    if onboarding_phase == "teach_1":
        # First item done - now teach second
        session.state = "onboarding_teach"
        session.context["onboarding_phase"] = "teach_2"
        session.context.pop("label", None)
        emit("tts", {
            "narration": f"Got it, {label} in the {location}. Now teach me the second item. Say its name.",
            "next_state": "onboarding_teach",
        })
    elif onboarding_phase == "teach_2":
        # Second item done - ready to scan
        session.state = "onboarding_await_scan"
        session.context["onboarding_phase"] = "await_scan"
        session.context.pop("label", None)
        emit("tts", {
            "narration": (
                f"Got it, {label} in the {location}. "
                "Great. Place both items somewhere in the room, then say scan."
            ),
            "next_state": "onboarding_await_scan",
        })
    else:
        session.state = "idle"
        session.context.pop("label", None)
        hint = _get_hint(session)
        narration = f"Got it, {label} in the {location}." if label else f"Got it, {location}."
        if hint:
            narration = f"{narration} {hint}"
        emit("tts", {"narration": narration, "next_state": "idle"})


def _handle_confirmation(session: VoiceSession, command_text: str) -> None:
    q = command_text.strip().lower()
    confirmed = any(t in q for t in ("yes", "yeah", "yep", "confirm", "do it", "correct", "sure"))
    denied = any(t in q for t in ("no", "nope", "cancel", "stop", "never mind"))
    pending = session.context.pop("pending_confirmation", None)
    session.state = "idle"

    if confirmed and not denied and pending:
        action = pending.get("action")
        if action == "delete_item":
            lbl = pending.get("label", "")
            try:
                deleted = get_database().delete_items_by_label(lbl)
                get_scan_pipeline().reload_database()
                if deleted > 0:
                    emit("tts", {"narration": f"{lbl.capitalize()} deleted.", "next_state": "idle"})
                else:
                    emit("tts", {"narration": "I could not find that item to delete.", "next_state": "idle"})
            except Exception:
                emit("tts", {"narration": "Could not delete the item.", "next_state": "idle"})
        else:
            emit("tts", {"narration": "Done.", "next_state": "idle"})
    else:
        emit("tts", {"narration": "Canceled.", "next_state": "idle"})


def _handle_item_query(session: VoiceSession, command_text: str) -> None:
    from visual_memory.api.routes.item_ask import process_item_ask_request
    scan_id = session.context.get("scan_id", "")
    label = session.context.get("current_label", "")
    state_ctx = build_state_contract(mode=session.state, context=session.context)
    result, status = process_item_ask_request(
        scan_id=scan_id, label=label, query=command_text, state_context=state_ctx
    )
    narration = result.get("narration", "") if isinstance(result, dict) else ""
    emit("action_result", {"type": "item_ask", "data": result})
    emit("tts", {"narration": narration or "Done.", "next_state": session.state})


def _handle_feedback(session: VoiceSession) -> None:
    scan_id = session.context.get("scan_id", "")
    label = session.context.get("current_label", "")
    if not scan_id or not label:
        emit("tts", {"narration": "No active scan to give feedback on.", "next_state": session.state})
        return
    try:
        embeddings = get_scan_pipeline().get_cached_embeddings(scan_id, label)
        if isinstance(embeddings, (tuple, list)) and len(embeddings) == 2:
            anchor_emb, query_emb = embeddings
            get_feedback_store().record_negative(anchor_emb, query_emb, label)
        else:
            emit("error", {"code": "cache_expired", "message": "Feedback expired. Please rescan."})
            return
    except Exception as exc:
        _log.warning({"event": "ws_feedback_failed", "tag": LogTag.API, "error": str(exc)})
    emit("tts", {"narration": "Got it, noted as incorrect.", "next_state": session.state})


def _handle_navigate_back(session: VoiceSession) -> None:
    _clear_focus_context(session)
    session.state = "idle"
    emit("action_result", {"type": "navigate_back", "data": {"action": "navigate_back"}})
    emit("tts", {"narration": "Going back.", "next_state": "idle"})


def _handle_open_settings(session: VoiceSession) -> None:
    _clear_focus_context(session)
    session.state = "idle"
    emit("action_result", {"type": "open_settings", "data": {"action": "open_settings"}})
    emit("tts", {"narration": "Opening settings.", "next_state": "idle"})


# Main dispatch

def _dispatch(
    session: VoiceSession,
    command_text: str,
    image_bytes: bytes | None,
    focal_length: float | None,
) -> None:
    """Route the command to the right handler based on session state and intent."""
    state = session.state

    _log.info({
        "event": "ws_dispatch",
        "tag": LogTag.API,
        "sid": session.sid,
        "state": state,
        "has_image": image_bytes is not None,
        "command_len": len(command_text),
    })

    # awaiting_image: run the pending action with the supplied image
    if state == "awaiting_image":
        if image_bytes is None:
            emit("control", {"action": "request_image"})
            emit("tts", {"narration": "Still waiting for an image.", "next_state": state})
            return
        pending = session.context.get("pending_action")
        if pending == "scan":
            _handle_scan(session, image_bytes, focal_length)
        elif pending == "remember":
            label = session.context.get("label", "")
            if not label:
                session.state = "idle"
                emit("tts", {"narration": "I lost the item name. Please start again.", "next_state": "idle"})
                return
            _handle_remember(session, image_bytes, label)
        else:
            session.state = "idle"
            emit("error", {"code": "bad_state", "message": "Unknown pending action."})
        return

    # awaiting_location: treat all input as a room name
    if state == "awaiting_location":
        _handle_location(session, command_text)
        return

    # awaiting_confirmation: yes/no on a pending action
    if state == "awaiting_confirmation":
        _handle_confirmation(session, command_text)
        return

    # focused_on_item: intercept item-specific queries and feedback; rest falls through
    if state == "focused_on_item":
        q = command_text.strip().lower()
        if "wrong" in q:
            _handle_feedback(session)
            return
        item_query_tokens = ("read", "what does", "text", "says", "export", "copy", "share",
                             "describe", "what is this", "looks like")
        if any(t in q for t in item_query_tokens):
            _handle_item_query(session, command_text)
            return
        # Not item-specific - fall through to normal command dispatch

    intent = _normalize_command_type(command_text, has_image=image_bytes is not None)

    # onboarding_await_scan: only accept scan and explicit navigation commands
    if state == "onboarding_await_scan":
        if intent == "navigate_back":
            _handle_navigate_back(session)
            return
        if intent == "open_settings":
            _handle_open_settings(session)
            return
        if intent != "scan":
            emit("tts", {"narration": ONBOARDING_AWAIT_SCAN_PROMPT, "next_state": state})
            return

    # Normal command dispatch (covers idle, onboarding_teach, focused_on_item fall-through)
    if intent == "navigate_back":
        _handle_navigate_back(session)
        return
    if intent == "open_settings":
        _handle_open_settings(session)
        return

    if intent == "scan":
        if image_bytes is None:
            session.context["pending_action"] = "scan"
            session.state = "awaiting_image"
            emit("control", {"action": "request_image", "context": "scan"})
            emit("tts", {"narration": "Hold your phone up.", "next_state": "awaiting_image"})
        else:
            _handle_scan(session, image_bytes, focal_length)

    elif intent == "remember":
        state_ctx = build_state_contract(mode=session.state, context=session.context)
        label = _extract_label(command_text, state_context=state_ctx)
        session.context["label"] = label
        if image_bytes is None:
            session.context["pending_action"] = "remember"
            session.state = "awaiting_image"
            emit("control", {"action": "request_image"})
            emit("tts", {
                "narration": f"Point your phone at your {label}.",
                "next_state": "awaiting_image",
            })
        else:
            _handle_remember(session, image_bytes, label)

    elif intent == "find":
        state_ctx = build_state_contract(mode=session.state, context=session.context)
        result, status = _run_find(command_text, state_context=state_ctx)
        narration = result.get("narration", "") if isinstance(result, dict) else ""
        emit("action_result", {"type": "find", "data": result})
        # Onboarding ask-demo completion
        if session.context.get("onboarding_phase") == "ask_prompted":
            session.context["onboarding_phase"] = ""
            session.state = "idle"
            hint = _get_hint(session)
            completion = "You are all set. Say scan, teach, or ask anything."
            if hint:
                completion = f"{completion} {hint}"
            emit("tts", {"narration": f"{narration} {completion}".strip(), "next_state": "idle"})
        else:
            session.state = "idle"
            emit("tts", {"narration": narration or "Nothing found.", "next_state": "idle"})

    else:
        from visual_memory.api.routes.ask import process_ask_query
        state_ctx = build_state_contract(mode=session.state, context=session.context)
        result, status = process_ask_query(command_text, state_context=state_ctx)
        narration = result.get("narration", "") if isinstance(result, dict) else ""
        emit("action_result", {"type": "ask", "data": result})
        if session.context.get("onboarding_phase") == "ask_prompted":
            session.context["onboarding_phase"] = ""
            session.state = "idle"
            hint = _get_hint(session)
            completion = "You are all set. Say scan, teach, or ask anything."
            if hint:
                completion = f"{completion} {hint}"
            emit("tts", {"narration": f"{narration} {completion}".strip(), "next_state": "idle"})
        else:
            session.state = "idle"
            emit("tts", {"narration": narration or "Not sure about that.", "next_state": "idle"})


# Event registration

def register_events(sio: SocketIO) -> None:
    """Attach all WebSocket event handlers to the SocketIO instance."""

    @sio.on("connect")
    def on_connect():
        if _API_KEY:
            key = request.headers.get("X-API-Key") or request.args.get("key", "")
            if key != _API_KEY:
                _log.warning({"event": "ws_auth_rejected", "tag": LogTag.API, "sid": request.sid})
                return False  # reject connection

        session = get_session(request.sid)
        _log.info({"event": "ws_connect", "tag": LogTag.API, "sid": request.sid})

        try:
            item_count = len(get_database().get_items_metadata())
        except Exception:
            item_count = -1

        if item_count == 0:
            session.state = "onboarding_teach"
            session.context["onboarding_phase"] = "teach_1"
            emit("tts", {
                "narration": (
                    "Welcome to Spaitra. I remember where your things are. "
                    "Grab two items near you and let's start. "
                    "Say teach, then the name of the first item."
                ),
                "next_state": "onboarding_teach",
            })
        else:
            session.state = "idle"
            emit("tts", {"narration": "Ready.", "next_state": "idle"})
        _emit_session_state(session)

    @sio.on("disconnect")
    def on_disconnect():
        _log.info({"event": "ws_disconnect", "tag": LogTag.API, "sid": request.sid})
        clear_session(request.sid)

    @sio.on("audio")
    def on_audio(data):
        if not isinstance(data, dict):
            emit("error", {"code": "bad_payload", "message": "Expected JSON object."})
            return

        session = get_session(request.sid)
        t0 = time.monotonic()

        audio_bytes = _decode_audio(data.get("audio", b""))
        raw_image = data.get("image")
        image_bytes = _decode_image(raw_image)
        if raw_image not in (None, "") and image_bytes is None:
            emit("error", {"code": "bad_payload", "message": "Invalid image payload."})
            return
        if len(audio_bytes) > _MAX_WS_MEDIA_BYTES:
            emit("error", {"code": "bad_payload", "message": "Audio payload exceeds 50MB limit."})
            return
        if image_bytes is not None and len(image_bytes) > _MAX_WS_MEDIA_BYTES:
            emit("error", {"code": "bad_payload", "message": "Image payload exceeds 50MB limit."})
            return
        focal_length = data.get("focal_length_px")
        if isinstance(focal_length, str):
            try:
                focal_length = float(focal_length)
            except ValueError:
                focal_length = None

        transcription, t_status = transcribe_audio_bytes(
            audio_bytes,
            use_context=True,
            voice_mode=session.state,
            voice_context={
                **session.context,
                "state_contract": build_state_contract(mode=session.state, context=session.context),
            },
        )
        if t_status != 200:
            emit("error", {
                "code": "transcription_failed",
                "message": transcription.get("error", "Transcription failed."),
            })
            return

        command_text = (transcription.get("text") or "").strip()
        emit("transcription", {
            "text": command_text,
            "context_used": transcription.get("context_used", False),
            "context_state_id": transcription.get("context_state_id"),
            "context_policy": transcription.get("context_policy"),
        })

        _dispatch(session, command_text, image_bytes, focal_length)
        _emit_session_state(session)

        _log.info({
            "event": "ws_audio_handled",
            "tag": LogTag.API,
            "sid": session.sid,
            "state_after": session.state,
            "duration_ms": round((time.monotonic() - t0) * 1000),
        })

    @sio.on("navigate")
    def on_navigate(data):
        if not isinstance(data, dict):
            emit("error", {"code": "bad_payload", "message": "Expected JSON object."})
            return

        session = get_session(request.sid)
        matches = session.context.get("scan_matches", [])

        if not matches:
            emit("tts", {"narration": "No scan results to navigate.", "next_state": session.state})
            return

        direction = (data.get("direction") or "next").lower()
        current = session.context.get("item_index", 0)

        if direction in ("next", "right"):
            new_index = min(current + 1, len(matches) - 1)
        elif direction in ("prev", "left"):
            new_index = max(current - 1, 0)
        else:
            try:
                new_index = max(0, min(int(data.get("item_index", current)), len(matches) - 1))
            except (TypeError, ValueError):
                new_index = current

        session.context["item_index"] = new_index
        match = matches[new_index]
        session.context["current_label"] = match.get("label", "")
        narration = match.get("narration") or match.get("label") or "Item."

        emit("action_result", {"type": "item_focus", "data": match})

        # Onboarding: after first swipe cycle, prompt the ask demo
        if session.context.get("onboarding_phase") == "ask" and new_index >= len(matches) - 1:
            ask_label = session.context.get("onboarding_ask_label", "")
            session.context["onboarding_phase"] = "ask_prompted"
            emit("tts", {
                "narration": (
                    f"{narration}. Swipe left to go back. That is how scan works. "
                    f"Now ask me where you left your {ask_label or 'first item'}."
                ),
                "next_state": "focused_on_item",
            })
            _emit_session_state(session)
            return

        emit("tts", {"narration": narration, "next_state": "focused_on_item"})
        _emit_session_state(session)

        _log.info({
            "event": "ws_navigate",
            "tag": LogTag.API,
            "sid": session.sid,
            "direction": direction,
            "item_index": new_index,
            "label": match.get("label", ""),
        })

    @sio.on("chat_start")
    def on_chat_start(data):
        """User pressed the Spaitra button. Respond with a listening prompt.

        The frontend should start recording audio after receiving this event.
        When recording ends, send the audio via the audio event as normal.
        """
        session = get_session(request.sid)
        emit("listening", {
            "state": session.state,
            "prompt": _listening_prompt(session),
        })
        _log.info({
            "event": "ws_chat_start",
            "tag": LogTag.API,
            "sid": request.sid,
            "state": session.state,
        })

    @sio.on("chat_stop")
    def on_chat_stop(data):
        """User released the Spaitra button without sending audio (canceled)."""
        session = get_session(request.sid)
        emit("listening_stopped", {"state": session.state})
