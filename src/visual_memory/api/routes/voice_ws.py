"""Stateful WebSocket voice session event handlers."""
from __future__ import annotations

import base64
import io
import os
import re
import tempfile
import time
from pathlib import Path

from flask import request
from flask_socketio import SocketIO, emit
from PIL import Image

from visual_memory.api.pipelines import (
    get_database,
    get_feedback_store,
    reload_scan_database_if_loaded,
    get_scan_pipeline,
    get_user_settings,
)
from visual_memory.api.narration import (
    ONBOARDING_AWAIT_SCAN_PROMPT,
    ONBOARDING_TEACH_LISTENING_PROMPT,
)
from visual_memory.api.routes.transcribe import transcribe_audio_bytes
from visual_memory.api.voice_session import (
    VoiceSession,
    apply_state_transition,
    clear_focus,
    clear_session,
    focused_label,
    focused_scan_id,
    focused_scan_index,
    get_session,
    session_state_payload,
    set_focused_match,
)
from visual_memory.engine.model_registry import registry
from visual_memory.engine.vlm import get_vlm_pipeline
from visual_memory.utils import get_logger
from visual_memory.utils.logger import LogTag
from visual_memory.utils.voice_state_context import build_state_contract

_log = get_logger(__name__)
_API_KEY = os.environ.get("API_KEY", "")
_ROOM_PREFIX = re.compile(r"^(in (the|my) |the |my )", re.IGNORECASE)
_MAX_WS_MEDIA_BYTES = 50 * 1024 * 1024
_DESCRIBE_SCENE_PATTERNS = (
    "describe what i am looking at",
    "describe what i'm looking at",
    "describe what im looking at",
    "describe what i see",
    "describe this scene",
)

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

_SHORTCUT_SCAN = "scan"
_SHORTCUT_ACK_PHASES = frozenset({"started", "submitted", "canceled"})
_SHORTCUT_ERROR_CODES = frozenset({"not_allowed_in_mode", "bad_payload", "unauthorized"})


# Helpers

def _onboarding_phase_prompt(phase: str) -> str:
    if phase == "teach_1":
        return "First, teach me your first item. Say teach, then the item name."
    if phase == "teach_2":
        return "Now teach me your second item. Say teach, then the item name."
    if phase == "await_scan":
        return "Now scan your two taught items. Hold chat and say scan."
    if phase == "ask":
        return "Swipe right to browse both detected items, then ask where you left one."
    if phase == "ask_prompted":
        return "Now ask where you left one of those items."
    return "Let's finish onboarding first."

def _emit_session_state(session: VoiceSession) -> None:
    emit("session_state", session_state_payload(session))


def _clear_focus_context(session: VoiceSession) -> None:
    clear_focus(session)


def _emit_error(code: str, message: str) -> None:
    _log.warning({"event": "ws_error_emitted", "tag": LogTag.API, "error_code": code, "message": message})
    emit("error", {"code": code, "message": message})


def _emit_shortcut_ack(shortcut: str, phase: str) -> None:
    if phase not in _SHORTCUT_ACK_PHASES:
        return
    emit("shortcut_ack", {"shortcut": shortcut, "phase": phase})


def _emit_shortcut_listening(*, shortcut: str, prompt: str, state: str) -> None:
    emit("shortcut_listening", {"shortcut": shortcut, "prompt": prompt, "state": state})


def _emit_shortcut_error(shortcut: str, code: str, message: str) -> None:
    normalized = code if code in _SHORTCUT_ERROR_CODES else "bad_payload"
    emit("shortcut_error", {"shortcut": shortcut, "code": normalized, "message": message})


def _shortcut_allowed(session: VoiceSession, shortcut: str) -> tuple[bool, str]:
    if shortcut != _SHORTCUT_SCAN:
        return False, "Shortcut is not supported."
    if session.state in {"awaiting_location", "awaiting_confirmation", "onboarding_teach"}:
        return False, "Scan is not available right now."
    if session.state == "awaiting_image":
        if session.context.get("pending_action") != "scan":
            return False, "Scan is not available right now."
    return True, ""


def _emit_turn(
    session: VoiceSession,
    *,
    next_state: str | None = None,
    context_updates: dict | None = None,
    clear_keys: tuple[str, ...] = (),
    action_result: dict | None = None,
    control: dict | None = None,
    tts_narration: str | None = None,
) -> None:
    apply_state_transition(
        session,
        next_state=next_state,
        context_updates=context_updates,
        clear_keys=clear_keys,
    )
    if action_result is not None:
        emit("action_result", action_result)
    if control is not None:
        emit("control", control)
        _log.info({
            "event": "ws_control_request_image",
            "tag": LogTag.API,
            "action": control.get("action"),
            "context": control.get("context"),
        })
    if tts_narration is not None:
        emit("tts", {"narration": tts_narration, "next_state": session.state})
    _emit_session_state(session)


def _focused_context(session: VoiceSession, *, require_match: bool = True) -> tuple[dict | None, str | None]:
    matches = session.context.get("scan_matches", [])
    scan_id = focused_scan_id(session)
    label = focused_label(session)
    if not scan_id:
        return None, "no_scan_id"
    if (not isinstance(matches, list) or not matches) and not require_match and label:
        return {
            "scan_id": scan_id,
            "label": label,
            "item_index": focused_scan_index(session),
            "match": {"label": label},
        }, None
    if not isinstance(matches, list) or not matches:
        return None, "no_matches"
    idx = focused_scan_index(session)
    if idx < 0 or idx >= len(matches):
        return None, "index_out_of_bounds"
    match = matches[idx] if isinstance(matches[idx], dict) else {}
    if label and match.get("label") and label != match.get("label"):
        return None, "label_mismatch"
    return {
        "scan_id": scan_id,
        "label": str(match.get("label") or label or ""),
        "item_index": idx,
        "match": match,
    }, None


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


def _decode_audio(value) -> tuple[bytes, str | None]:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value), None
    if not isinstance(value, str):
        return b"", None
    encoded = value.strip()
    if "," in encoded and encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    try:
        return base64.b64decode(encoded, validate=True), None
    except Exception as exc:
        _log.warning({
            "event": "ws_audio_decode_failed",
            "tag": LogTag.API,
            "reason": "invalid_base64",
            "error": str(exc),
        })
        return b"", "invalid_audio_payload"


def _decode_image(value) -> tuple[bytes | None, str | None]:
    if value is None:
        return None, None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value), None
    if not isinstance(value, str):
        return None, "invalid_image_payload"
    encoded = value.strip()
    if not encoded:
        return None, None
    if "," in encoded and encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    try:
        return base64.b64decode(encoded, validate=True), None
    except Exception as exc:
        _log.warning({
            "event": "ws_image_decode_failed",
            "tag": LogTag.API,
            "reason": "invalid_base64",
            "error": str(exc),
        })
        return None, "invalid_image_payload"


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


def _extract_idle_describe_target(command_text: str) -> tuple[bool, str | None]:
    q = re.sub(r"\s+", " ", (command_text or "").strip().lower())
    if "describe" not in q:
        return False, None
    if any(p in q for p in _DESCRIBE_SCENE_PATTERNS):
        return True, None
    match = re.search(r"\bdescribe\s+(?:this|the|my|a|an)?\s*([a-z0-9][a-z0-9\s'\-]{0,40})\b", q)
    if not match:
        return False, None
    target = re.sub(r"\s+", " ", match.group(1)).strip(" ?.!,'\"")
    if not target or target in {"it", "this", "that", "scene"}:
        return True, None
    return True, target


def _describe_image_with_vlm(image: Image.Image, timeout_s: float) -> str:
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="ws_describe_", suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        image.save(tmp_path, format="JPEG", quality=90)
        return get_vlm_pipeline().describe(tmp_path, timeout=timeout_s)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _handle_idle_describe(session: VoiceSession, image_bytes: bytes, target: str | None) -> None:
    apply_state_transition(
        session,
        next_state="idle",
        clear_keys=("pending_action", "describe_target"),
    )

    perf_cfg = get_user_settings().get_performance_config()
    if not perf_cfg.vlm_enabled:
        _emit_turn(
            session,
            tts_narration="Visual describe is disabled in fast mode. Switch to balanced or accurate.",
        )
        return

    timeout_s = float(perf_cfg.vlm_timeout_seconds)
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        _emit_error("bad_payload", f"Could not decode image: {exc}")
        _emit_session_state(session)
        return

    if target is None:
        try:
            narration = _describe_image_with_vlm(image, timeout_s)
            _emit_turn(
                session,
                action_result={"type": "ask", "data": {"action": "describe_scene", "target": None}},
                tts_narration=narration,
            )
        except TimeoutError:
            _emit_turn(session, tts_narration="That took too long. Try again.")
        except Exception as exc:
            _log.warning({"event": "ws_describe_scene_failed", "error": str(exc)})
            _emit_turn(session, tts_narration="I could not describe this view right now.")
        return

    try:
        detection = registry.gdino_detector.detect(image, target)
    except Exception as exc:
        _log.warning({"event": "ws_describe_target_detect_failed", "target": target, "error": str(exc)})
        _emit_turn(session, tts_narration="I could not isolate that object right now.")
        return

    if not detection:
        _emit_turn(session, tts_narration=f"I could not find {target} in this view.")
        return

    box = detection.get("box")
    if not isinstance(box, list) or len(box) != 4:
        _emit_turn(session, tts_narration=f"I could not isolate {target} in this view.")
        return

    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(x1, image.width - 1))
    y1 = max(0, min(y1, image.height - 1))
    x2 = max(x1 + 1, min(x2, image.width))
    y2 = max(y1 + 1, min(y2, image.height))
    crop = image.crop((x1, y1, x2, y2))

    try:
        narration = _describe_image_with_vlm(crop, timeout_s)
        _emit_turn(
            session,
            action_result={"type": "ask", "data": {"action": "describe_scene", "target": target}},
            tts_narration=narration,
        )
    except TimeoutError:
        _emit_turn(session, tts_narration="That took too long. Try again.")
    except Exception as exc:
        _log.warning({"event": "ws_describe_target_vlm_failed", "target": target, "error": str(exc)})
        _emit_turn(session, tts_narration="I could not describe that object right now.")


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
        _emit_error("scan_failed", result.get("error", "Scan failed."))
        return

    matches = result.get("matches", [])
    scan_id = result.get("scan_id", "")
    _increment(session, "scan_count")

    if not matches:
        if onboarding_phase == "await_scan":
            narration = "I did not spot either item. Make sure they are in view and try again."
            next_state = "onboarding_await_scan"
        else:
            narration = "I did not find anything familiar."
            next_state = "idle"
        _emit_turn(
            session,
            next_state=next_state,
            action_result={"type": "scan", "data": result},
            tts_narration=narration,
        )
        return

    set_focused_match(session, matches, 0)
    apply_state_transition(session, context_updates={"scan_id": scan_id})

    narration_parts = [m.get("narration") or m.get("label", "") for m in matches]
    combined = ". ".join(p for p in narration_parts if p)

    if onboarding_phase == "await_scan":
        # Onboarding scan succeeded - move to swipe intro
        ask_label = session.context.get("onboarding_item_1", "") or (matches[0].get("label", "") if matches else "")
        _emit_turn(
            session,
            next_state="focused_on_item",
            context_updates={"onboarding_phase": "ask", "onboarding_ask_label": ask_label},
            action_result={"type": "scan", "data": result},
            tts_narration=f"{combined}. Now swipe right to browse each item.",
        )
        return

    hint = _get_hint(session)
    if hint:
        combined = f"{combined}. {hint}"
    _emit_turn(
        session,
        next_state="focused_on_item",
        action_result={"type": "scan", "data": result},
        tts_narration=combined,
    )


def _handle_remember(session: VoiceSession, image_bytes: bytes, label: str) -> None:
    onboarding_phase = session.context.get("onboarding_phase", "")
    result, status = _run_remember(image_bytes, label)

    if status != 200:
        _emit_error("remember_failed", result.get("error", "Remember failed."))
        _emit_turn(session, action_result={"type": "remember", "data": result})
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
        _emit_turn(
            session,
            action_result={"type": "remember", "data": result},
            tts_narration=narration,
        )
        return

    _increment(session, "teach_count")
    updates = {"label": label}
    if onboarding_phase == "teach_1":
        updates["onboarding_item_1"] = label
    elif onboarding_phase == "teach_2":
        updates["onboarding_item_2"] = label
    _emit_turn(
        session,
        next_state="awaiting_location",
        context_updates=updates,
        action_result={"type": "remember", "data": result},
        tts_narration=f"{label.capitalize()} saved. Which room is this?",
    )


def _handle_location(session: VoiceSession, command_text: str) -> None:
    """Handle the awaiting_location state: save sighting and advance state."""
    onboarding_phase = session.context.get("onboarding_phase", "")
    label = session.context.get("label", "")
    raw = (command_text or "").strip().lower()
    if any(t in raw for t in ("scan", "teach", "remember", "find", "where", "settings", "go back", "home")):
        _emit_turn(
            session,
            next_state="awaiting_location",
            tts_narration="Please tell me the room first, like kitchen or bedroom.",
        )
        return

    location = _normalize_room_name(command_text)

    if not location:
        _emit_turn(session, next_state="awaiting_location", tts_narration="What room is this?")
        return

    if label:
        try:
            db = get_database()
            if db.get_items_metadata(label=label):
                db.add_sighting(label=label, room_name=location)
        except Exception as exc:
            _log.warning({"event": "ws_sighting_failed", "tag": LogTag.API, "error": str(exc)})

    if onboarding_phase == "teach_1":
        _emit_turn(
            session,
            next_state="onboarding_teach",
            context_updates={"onboarding_phase": "teach_2"},
            clear_keys=("label",),
            action_result={"type": "set_location", "data": {"label": label, "location": location}},
            tts_narration=f"Got it, {label} in the {location}. Now teach me the second item. Say its name.",
        )
    elif onboarding_phase == "teach_2":
        _emit_turn(
            session,
            next_state="onboarding_await_scan",
            context_updates={"onboarding_phase": "await_scan"},
            clear_keys=("label",),
            action_result={"type": "set_location", "data": {"label": label, "location": location}},
            tts_narration=(
                f"Got it, {label} in the {location}. "
                "Great. Place both items somewhere in the room, then say scan."
            ),
        )
    else:
        hint = _get_hint(session)
        narration = f"Got it, {label} in the {location}." if label else f"Got it, {location}."
        if hint:
            narration = f"{narration} {hint}"
        _emit_turn(
            session,
            next_state="idle",
            clear_keys=("label",),
            action_result={"type": "set_location", "data": {"label": label, "location": location}},
            tts_narration=narration,
        )


def _handle_confirmation(session: VoiceSession, command_text: str) -> None:
    q = command_text.strip().lower()
    confirmed = any(t in q for t in ("yes", "yeah", "yep", "confirm", "do it", "correct", "sure"))
    denied = any(t in q for t in ("no", "nope", "cancel", "stop", "never mind"))
    pending = session.context.pop("pending_confirmation", None)
    apply_state_transition(session, next_state="idle")

    if confirmed and not denied and pending:
        action = pending.get("action")
        if action == "delete_item":
            lbl = pending.get("label", "")
            try:
                deleted = get_database().delete_items_by_label(lbl)
                reload_scan_database_if_loaded()
                if deleted > 0:
                    _emit_turn(session, tts_narration=f"{lbl.capitalize()} deleted.")
                else:
                    _emit_turn(session, tts_narration="I could not find that item to delete.")
            except Exception:
                _emit_turn(session, tts_narration="Could not delete the item.")
        else:
            _emit_turn(session, tts_narration="Done.")
    else:
        _emit_turn(session, tts_narration="Canceled.")


def _handle_item_query(session: VoiceSession, command_text: str) -> None:
    from visual_memory.api.routes.item_ask import process_item_ask_request
    ctx, reason = _focused_context(session)
    if ctx is None:
        _emit_turn(
            session,
            next_state="idle",
            clear_keys=("scan_matches", "scan_id", "item_index", "current_label"),
            tts_narration="Focused item context expired. Please scan again.",
        )
        _log.warning({"event": "ws_focus_fallback", "tag": LogTag.API, "reason": reason})
        return
    scan_id = ctx["scan_id"]
    label = ctx["label"]
    state_ctx = build_state_contract(mode=session.state, context=session.context)
    result, status = process_item_ask_request(
        scan_id=scan_id, label=label, query=command_text, state_context=state_ctx
    )
    narration = result.get("narration", "") if isinstance(result, dict) else ""
    _emit_turn(
        session,
        next_state="focused_on_item",
        action_result={"type": "item_ask", "data": result},
        tts_narration=narration or "Done.",
    )


def _handle_feedback(session: VoiceSession) -> None:
    ctx, reason = _focused_context(session, require_match=False)
    if ctx is None:
        _emit_turn(session, tts_narration="No active scan to give feedback on.")
        _log.warning({"event": "ws_focus_fallback", "tag": LogTag.API, "reason": reason})
        return
    scan_id = ctx["scan_id"]
    label = ctx["label"]
    try:
        embeddings = get_scan_pipeline().get_cached_embeddings(scan_id, label)
        if isinstance(embeddings, (tuple, list)) and len(embeddings) == 2:
            anchor_emb, query_emb = embeddings
            get_feedback_store().record_negative(anchor_emb, query_emb, label)
        else:
            _emit_error("cache_expired", "Feedback expired. Please rescan.")
            return
    except Exception as exc:
        _log.warning({"event": "ws_feedback_failed", "tag": LogTag.API, "error": str(exc)})
    _emit_turn(session, tts_narration="Got it, noted as incorrect.")


def _handle_positive_feedback(session: VoiceSession) -> None:
    ctx, reason = _focused_context(session, require_match=False)
    if ctx is None:
        _emit_turn(session, tts_narration="No active scan to give feedback on.")
        _log.warning({"event": "ws_focus_fallback", "tag": LogTag.API, "reason": reason})
        return
    scan_id = ctx["scan_id"]
    label = ctx["label"]
    try:
        embeddings = get_scan_pipeline().get_cached_embeddings(scan_id, label)
        if isinstance(embeddings, (tuple, list)) and len(embeddings) == 2:
            anchor_emb, query_emb = embeddings
            get_feedback_store().record_positive(anchor_emb, query_emb, label)
        else:
            _emit_error("cache_expired", "Feedback expired. Please rescan.")
            return
    except Exception as exc:
        _log.warning({"event": "ws_positive_feedback_failed", "tag": LogTag.API, "error": str(exc)})
    _emit_turn(session, tts_narration="Got it, noted as correct.")


def _handle_navigate_back(session: VoiceSession) -> None:
    _emit_turn(
        session,
        next_state="idle",
        clear_keys=("scan_matches", "scan_id", "item_index", "current_label"),
        action_result={"type": "navigate_back", "data": {"action": "navigate_back"}},
        tts_narration="Going back.",
    )


def _handle_open_settings(session: VoiceSession) -> None:
    _emit_turn(
        session,
        next_state="idle",
        clear_keys=("scan_matches", "scan_id", "item_index", "current_label"),
        action_result={"type": "open_settings", "data": {"action": "open_settings"}},
        tts_narration="Opening settings.",
    )


def _check_turn_policy(session: VoiceSession, intent: str, onboarding_phase: str, state_contract: dict) -> bool:
    if onboarding_phase in {"teach_1", "teach_2"} and intent != "remember":
        _emit_turn(session, tts_narration=_onboarding_phase_prompt(onboarding_phase))
        return False
    if onboarding_phase == "await_scan" and intent != "scan":
        _emit_turn(session, tts_narration=_onboarding_phase_prompt(onboarding_phase))
        return False
    if onboarding_phase == "ask":
        _emit_turn(session, tts_narration=_onboarding_phase_prompt(onboarding_phase))
        return False
    if onboarding_phase == "ask_prompted" and intent not in {"find", "ask"}:
        _emit_turn(session, tts_narration=_onboarding_phase_prompt(onboarding_phase))
        return False
    if session.state == "onboarding_await_scan" and intent != "scan":
        _emit_turn(session, next_state=session.state, tts_narration=ONBOARDING_AWAIT_SCAN_PROMPT)
        return False
    if session.state == "focused_on_item" and intent == "remember":
        guidance = (state_contract.get("guidance") or {}).get("remember")
        _emit_turn(
            session,
            next_state=session.state,
            tts_narration=guidance or "Return home to teach a new item. Say go back, then teach.",
        )
        return False
    return True


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
            _emit_turn(
                session,
                next_state="awaiting_image",
                control={"action": "request_image"},
                tts_narration="Still waiting for an image.",
            )
            return
        pending = session.context.get("pending_action")
        if pending == "scan":
            _handle_scan(session, image_bytes, focal_length)
        elif pending == "remember":
            label = session.context.get("label", "")
            if not label:
                _emit_turn(session, next_state="idle", tts_narration="I lost the item name. Please start again.")
                return
            _handle_remember(session, image_bytes, label)
        elif pending in {"describe_scene", "describe_target"}:
            target = session.context.get("describe_target")
            if not isinstance(target, str):
                target = None
            _handle_idle_describe(session, image_bytes, target)
        else:
            apply_state_transition(session, next_state="idle")
            _emit_error("bad_state", "Unknown pending action.")
        return

    # awaiting_location: treat all input as a room name
    if state == "awaiting_location":
        _handle_location(session, command_text)
        return

    # awaiting_confirmation: yes/no on a pending action
    if state == "awaiting_confirmation":
        _handle_confirmation(session, command_text)
        return

    intent = _normalize_command_type(command_text, has_image=image_bytes is not None)
    state_contract = build_state_contract(mode=session.state, context=session.context)
    onboarding_phase = str(session.context.get("onboarding_phase", "") or "")

    if not _check_turn_policy(session, intent, onboarding_phase, state_contract):
        return

    # focused_on_item: intercept item-specific queries and feedback; rest falls through
    if state == "focused_on_item":
        q = command_text.strip().lower()
        if "wrong" in q:
            _handle_feedback(session)
            return
        if any(t in q for t in ("right", "correct", "yes this one", "this is right")):
            _handle_positive_feedback(session)
            return
        item_query_tokens = ("read", "what does", "text", "says", "export", "copy", "share",
                             "describe", "what is this", "looks like")
        if any(t in q for t in item_query_tokens):
            _handle_item_query(session, command_text)
            return
        # Not item-specific - fall through to normal command dispatch

    if state == "idle":
        is_describe, describe_target = _extract_idle_describe_target(command_text)
        if is_describe:
            if image_bytes is None:
                updates = {"pending_action": "describe_scene" if describe_target is None else "describe_target"}
                if describe_target is not None:
                    updates["describe_target"] = describe_target
                _emit_turn(
                    session,
                    next_state="awaiting_image",
                    context_updates=updates,
                    control={"action": "request_image", "context": "describe"},
                    tts_narration="Hold your phone up.",
                )
                return
            _handle_idle_describe(session, image_bytes, describe_target)
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
            _emit_turn(
                session,
                next_state="awaiting_image",
                context_updates={"pending_action": "scan"},
                control={"action": "request_image", "context": "scan"},
                tts_narration="Hold your phone up.",
            )
        else:
            _handle_scan(session, image_bytes, focal_length)

    elif intent == "remember":
        state_ctx = build_state_contract(mode=session.state, context=session.context)
        label = _extract_label(command_text, state_context=state_ctx)
        if image_bytes is None:
            _emit_turn(
                session,
                next_state="awaiting_image",
                context_updates={"pending_action": "remember", "label": label},
                control={"action": "request_image"},
                tts_narration=f"Point your phone at your {label}.",
            )
        else:
            _handle_remember(session, image_bytes, label)

    elif intent == "find":
        state_ctx = build_state_contract(mode=session.state, context=session.context)
        result, status = _run_find(command_text, state_context=state_ctx)
        narration = result.get("narration", "") if isinstance(result, dict) else ""
        # Onboarding ask-demo completion
        if session.context.get("onboarding_phase") == "ask_prompted":
            hint = _get_hint(session)
            completion = "You are all set. Say scan, teach, or ask anything."
            if hint:
                completion = f"{completion} {hint}"
            _emit_turn(
                session,
                next_state="idle",
                context_updates={"onboarding_phase": ""},
                action_result={"type": "find", "data": result},
                tts_narration=f"{narration} {completion}".strip(),
            )
        else:
            _emit_turn(
                session,
                next_state="idle",
                action_result={"type": "find", "data": result},
                tts_narration=narration or "Nothing found.",
            )

    else:
        from visual_memory.api.routes.ask import process_ask_query
        state_ctx = build_state_contract(mode=session.state, context=session.context)
        result, status = process_ask_query(command_text, state_context=state_ctx)
        narration = result.get("narration", "") if isinstance(result, dict) else ""
        if session.context.get("onboarding_phase") == "ask_prompted":
            hint = _get_hint(session)
            completion = "You are all set. Say scan, teach, or ask anything."
            if hint:
                completion = f"{completion} {hint}"
            _emit_turn(
                session,
                next_state="idle",
                context_updates={"onboarding_phase": ""},
                action_result={"type": "ask", "data": result},
                tts_narration=f"{narration} {completion}".strip(),
            )
        else:
            _emit_turn(
                session,
                next_state="idle",
                action_result={"type": "ask", "data": result},
                tts_narration=narration or "Not sure about that.",
            )


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
            _emit_turn(
                session,
                next_state="onboarding_teach",
                context_updates={"onboarding_phase": "teach_1"},
                tts_narration=(
                    "Welcome to Spaitra. I remember where your things are. "
                    "Grab two items near you and let's start. "
                    "Say teach, then the name of the first item."
                ),
            )
        else:
            _emit_turn(session, next_state="idle", tts_narration="Ready.")

    @sio.on("disconnect")
    def on_disconnect():
        _log.info({"event": "ws_disconnect", "tag": LogTag.API, "sid": request.sid})
        _log.info({"event": "ws_reconnect_rate_probe", "tag": LogTag.PERF, "sid": request.sid})
        clear_session(request.sid)

    @sio.on("audio")
    def on_audio(data):
        if not isinstance(data, dict):
            _emit_error("bad_payload", "Expected JSON object.")
            return

        session = get_session(request.sid)
        t0 = time.monotonic()

        audio_bytes, audio_decode_error = _decode_audio(data.get("audio", b""))
        if audio_decode_error is not None:
            _emit_error("bad_payload", "Invalid audio payload.")
            return
        raw_image = data.get("image")
        image_bytes, image_decode_error = _decode_image(raw_image)
        if image_decode_error is not None:
            _emit_error("bad_payload", "Invalid image payload.")
            return
        if len(audio_bytes) > _MAX_WS_MEDIA_BYTES:
            _emit_error("bad_payload", "Audio payload exceeds 50MB limit.")
            return
        if image_bytes is not None and len(image_bytes) > _MAX_WS_MEDIA_BYTES:
            _emit_error("bad_payload", "Image payload exceeds 50MB limit.")
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
            _emit_error(
                str(transcription.get("code") or "transcription_failed"),
                transcription.get("message") or transcription.get("error", "Transcription failed."),
            )
            return

        command_text = (transcription.get("text") or "").strip()
        emit("transcription", {
            "text": command_text,
            "context_used": transcription.get("context_used", False),
            "context_state_id": transcription.get("context_state_id"),
            "context_policy": transcription.get("context_policy"),
        })

        if transcription.get("decode_ms") is not None:
            _log.info({
                "event": "ws_audio_decode_latency",
                "tag": LogTag.PERF,
                "sid": session.sid,
                "decode_ms": transcription.get("decode_ms"),
            })
        if transcription.get("transcription_ms") is not None:
            _log.info({
                "event": "ws_transcription_latency",
                "tag": LogTag.PERF,
                "sid": session.sid,
                "transcription_ms": transcription.get("transcription_ms"),
            })

        _dispatch(session, command_text, image_bytes, focal_length)

        _log.info({
            "event": "ws_audio_handled",
            "tag": LogTag.API,
            "sid": session.sid,
            "state_after": session.state,
            "error_code": None,
            "duration_ms": round((time.monotonic() - t0) * 1000),
            "turn_total_latency_ms": round((time.monotonic() - t0) * 1000),
        })

    @sio.on("navigate")
    def on_navigate(data):
        if not isinstance(data, dict):
            _emit_error("bad_payload", "Expected JSON object.")
            return

        session = get_session(request.sid)
        matches = session.context.get("scan_matches", [])

        if not matches:
            _emit_turn(session, tts_narration="No scan results to navigate.")
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

        match = set_focused_match(session, matches, new_index)
        narration = match.get("narration") or match.get("label") or "Item."

        # Onboarding: after first swipe cycle, prompt the ask demo
        if session.context.get("onboarding_phase") == "ask" and new_index >= len(matches) - 1:
            ask_label = session.context.get("onboarding_ask_label", "")
            _emit_turn(
                session,
                next_state="focused_on_item",
                context_updates={"onboarding_phase": "ask_prompted"},
                action_result={"type": "item_focus", "data": match},
                tts_narration=(
                    f"{narration}. Swipe left to go back. That is how scan works. "
                    f"Now ask me where you left your {ask_label or 'first item'}."
                ),
            )
            return

        _emit_turn(
            session,
            next_state="focused_on_item",
            action_result={"type": "item_focus", "data": match},
            tts_narration=narration,
        )

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

    @sio.on("shortcut_start")
    def on_shortcut_start(data):
        if not isinstance(data, dict):
            _emit_shortcut_error(_SHORTCUT_SCAN, "bad_payload", "Expected JSON object.")
            return
        shortcut = str(data.get("shortcut") or "").strip().lower()
        if shortcut != _SHORTCUT_SCAN:
            _emit_shortcut_error(shortcut or _SHORTCUT_SCAN, "bad_payload", "Unsupported shortcut.")
            return
        session = get_session(request.sid)
        allowed, message = _shortcut_allowed(session, shortcut)
        if not allowed:
            _emit_shortcut_error(shortcut, "not_allowed_in_mode", message)
            return
        _emit_shortcut_ack(shortcut, "started")
        _emit_shortcut_listening(shortcut=shortcut, prompt="Hold your phone up.", state="awaiting_image")

    @sio.on("shortcut_submit")
    def on_shortcut_submit(data):
        if not isinstance(data, dict):
            _emit_shortcut_error(_SHORTCUT_SCAN, "bad_payload", "Expected JSON object.")
            return
        shortcut = str(data.get("shortcut") or "").strip().lower()
        if shortcut != _SHORTCUT_SCAN:
            _emit_shortcut_error(shortcut or _SHORTCUT_SCAN, "bad_payload", "Unsupported shortcut.")
            return
        session = get_session(request.sid)
        allowed, message = _shortcut_allowed(session, shortcut)
        if not allowed:
            _emit_shortcut_error(shortcut, "not_allowed_in_mode", message)
            return

        audio_bytes, audio_decode_error = _decode_audio(data.get("audio", b""))
        if audio_decode_error is not None:
            _emit_shortcut_error(shortcut, "bad_payload", "Invalid audio payload.")
            return
        if len(audio_bytes) > _MAX_WS_MEDIA_BYTES:
            _emit_shortcut_error(shortcut, "bad_payload", "Audio payload exceeds 50MB limit.")
            return

        image_bytes, image_decode_error = _decode_image(data.get("image"))
        if image_decode_error is not None:
            _emit_shortcut_error(shortcut, "bad_payload", "Invalid image payload.")
            return
        if image_bytes is not None and len(image_bytes) > _MAX_WS_MEDIA_BYTES:
            _emit_shortcut_error(shortcut, "bad_payload", "Image payload exceeds 50MB limit.")
            return

        focal_length = data.get("focal_length_px")
        if isinstance(focal_length, str):
            try:
                focal_length = float(focal_length)
            except ValueError:
                focal_length = None

        _emit_shortcut_ack(shortcut, "submitted")
        _dispatch(session, _SHORTCUT_SCAN, image_bytes, focal_length)

    @sio.on("shortcut_cancel")
    def on_shortcut_cancel(data):
        if not isinstance(data, dict):
            _emit_shortcut_error(_SHORTCUT_SCAN, "bad_payload", "Expected JSON object.")
            return
        shortcut = str(data.get("shortcut") or "").strip().lower()
        if shortcut != _SHORTCUT_SCAN:
            _emit_shortcut_error(shortcut or _SHORTCUT_SCAN, "bad_payload", "Unsupported shortcut.")
            return
        session = get_session(request.sid)
        _emit_shortcut_ack(shortcut, "canceled")
        if session.state == "awaiting_image" and session.context.get("pending_action") == "scan":
            _emit_turn(
                session,
                next_state="idle",
                clear_keys=("pending_action", "describe_target", "label", "_pending_ts"),
                tts_narration="Canceled.",
            )
