from __future__ import annotations

import base64
import json
import re
import time

from flask import Blueprint, jsonify, request

from visual_memory.api.pipelines import get_database, get_settings
from visual_memory.api.routes.ask import process_ask_query
from visual_memory.api.routes.find import _format_sighting, _ocr_content_match, build_narration
from visual_memory.api.routes.item_ask import process_item_ask_request
from visual_memory.api.routes.remember import process_remember_request
from visual_memory.api.routes.scan import process_scan_request
from visual_memory.api.routes.transcribe import transcribe_audio_bytes
from visual_memory.utils import get_logger
from visual_memory.utils.ollama_utils import extract_search_term

voice_bp = Blueprint("voice", __name__)
_log = get_logger(__name__)
_ROOM_PREFIX = re.compile(r"^(in (the|my) |the |my )", re.IGNORECASE)
_TEACH_PATTERNS = [
    re.compile(r"\b(?:remember|teach|this is)\b.*?\b(?:as|called|named)\b\s+(.+)", re.IGNORECASE),
    re.compile(r"\b(?:this is my|that's my|it's my)\b\s+(.+)", re.IGNORECASE),
    re.compile(r"\b(?:save this as|store this as|call this)\b\s+(.+)", re.IGNORECASE),
]


def _extract_audio_bytes(payload: dict) -> bytes:
    if request.files.get("audio") is not None:
        return request.files["audio"].read()
    if request.files.get("audio_file") is not None:
        return request.files["audio_file"].read()
    raw_audio = payload.get("audio")
    if isinstance(raw_audio, str) and raw_audio.strip():
        encoded = raw_audio.strip()
        if "," in encoded and encoded.startswith("data:"):
            encoded = encoded.split(",", 1)[1]
        try:
            return base64.b64decode(encoded, validate=True)
        except Exception:
            _log.warning({"event": "voice_audio_decode_failed", "reason": "invalid_base64"})
            return b""
    if isinstance(raw_audio, dict):
        b64 = raw_audio.get("base64")
        if isinstance(b64, str) and b64.strip():
            try:
                return base64.b64decode(b64.strip(), validate=True)
            except Exception:
                _log.warning({"event": "voice_audio_decode_failed", "reason": "invalid_base64_dict"})
                return b""
    return request.data or b""


def _resolve_query_for_find(query: str, label: str) -> str:
    if label:
        return label
    extracted = extract_search_term(query)
    if extracted:
        return extracted
    return query.strip()


def _normalize_room_name(name: str) -> str:
    value = _ROOM_PREFIX.sub("", (name or "").strip()).strip().lower()
    return value


def _extract_state(payload: dict) -> tuple[str, dict]:
    state = payload.get("state")
    if not isinstance(state, dict):
        state_raw = request.form.get("state", "")
        if state_raw:
            try:
                state = json.loads(state_raw)
            except Exception:
                state = {}
    if not isinstance(state, dict):
        header_raw = request.headers.get("X-State", "")
        if header_raw:
            try:
                state = json.loads(header_raw)
            except Exception:
                state = {}
    if not isinstance(state, dict):
        state = {}

    mode = str(state.get("current_mode") or state.get("mode") or "idle").strip().lower() or "idle"
    context = state.get("context")
    if not isinstance(context, dict):
        context = {}
    return mode, context


def extract_teach_label(transcription: str) -> str | None:
    text = (transcription or "").strip()
    if not text:
        return None
    for pattern in _TEACH_PATTERNS:
        match = pattern.search(text)
        if match:
            label = match.group(1).strip().strip("\"'").strip(".,!?;:")
            if label:
                return label
    return None


def classify_item_intent(query: str) -> str:
    lower = (query or "").lower()
    if any(kw in lower for kw in ["read", "text", "say", "says"]):
        return "read_ocr"
    if any(kw in lower for kw in ["export", "copy", "share"]):
        return "export_ocr"
    if any(kw in lower for kw in ["describe", "look", "what is"]):
        return "describe"
    if any(kw in lower for kw in ["rename", "call"]):
        return "rename"
    if any(kw in lower for kw in ["where", "location"]):
        return "find"
    return "read_ocr"


def classify_command(transcription: str, state: dict) -> dict:
    lower = (transcription or "").lower().strip()
    mode = state.get("current_mode", "idle")

    if mode == "awaiting_location":
        return {"command": "set_location", "location": transcription}

    if mode == "focused_on_item":
        intent = classify_item_intent(transcription)
        return {"command": "item_ask", "intent": intent}

    if lower in ["scan", "stop"]:
        return {"command": lower}

    teach_label = extract_teach_label(transcription)
    if teach_label:
        return {"command": "remember", "label": teach_label}

    if any(kw in lower for kw in ["where", "find"]):
        return {"command": "find", "query": transcription}

    if any(kw in lower for kw in ["receipt", "paper", "says", "written"]):
        return {"command": "ask", "query_type": "document"}

    if any(kw in lower for kw in ["what", "when", "how", "tell me"]):
        return {"command": "ask", "query_type": "general"}

    return {"command": "ask", "query_type": "unknown"}


def _process_set_location_request(label: str, location_text: str) -> tuple[dict, int]:
    label = (label or "").strip()
    room_name = _normalize_room_name(location_text)
    if not label:
        return {"error": "missing field: state.context.label"}, 400
    if not room_name:
        return {"error": "missing room name"}, 400

    get_database().add_sighting(label=label, room_name=room_name)
    return {
        "action": "set_location",
        "label": label,
        "location": room_name,
        "saved": True,
        "narration": f"Got it. {label.capitalize()} in the {room_name}.",
    }, 200


def _process_confirm_action_request(command_text: str, context: dict) -> tuple[dict, int]:
    q = (command_text or "").strip().lower()
    affirmative = {"yes", "yeah", "yep", "confirm", "do it", "correct"}
    negative = {"no", "nope", "cancel", "stop"}
    confirmed = any(token in q for token in affirmative)
    denied = any(token in q for token in negative)
    if not q:
        return {"error": "missing confirmation input"}, 400
    return {
        "action": "confirm_action",
        "confirmed": bool(confirmed and not denied),
        "target_action": context.get("action"),
        "target": context.get("target"),
        "narration": "Confirmed." if confirmed and not denied else "Canceled.",
    }, 200


def _process_find_request(query: str, label: str) -> tuple[dict, int]:
    search_label = _resolve_query_for_find(query, label)
    if not search_label:
        return {"error": "missing field: query"}, 400

    db = get_database()
    row = db.get_last_sighting(search_label)
    matched_by = "exact"
    resolved_label = search_label

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
            "narration": "I couldn't find anything matching that in your memory.",
            "sightings": [],
        }, 200

    sighting = _format_sighting(row)
    out = {
        "label": search_label,
        "found": True,
        "matched_label": resolved_label,
        "matched_by": matched_by,
        "last_sighting": sighting,
        "sightings": [sighting],
        "narration": build_narration(resolved_label, sighting),
    }
    return out, 200


@voice_bp.post("/voice")
def voice():
    t0 = time.monotonic()
    use_context = request.args.get("context", "1") != "0"
    payload = request.get_json(silent=True) if request.is_json else {}
    payload = payload or {}
    mode, state_context = _extract_state(payload)

    forced_type = (payload.get("request_type") or request.form.get("request_type", "")).strip().lower()
    provided_text = (payload.get("text") or request.form.get("text", "")).strip()
    scan_id = (payload.get("scan_id") or request.form.get("scan_id", "") or state_context.get("scan_id", "")).strip()
    label = (payload.get("label") or request.form.get("label", "") or state_context.get("label", "")).strip()

    transcription = {"text": provided_text, "source": "client", "context_used": False}
    status = 200

    if not provided_text:
        audio_bytes = _extract_audio_bytes(payload)
        transcription, status = transcribe_audio_bytes(audio_bytes, use_context=use_context)
        if status != 200:
            return jsonify(transcription), status
        transcription["source"] = "whisper"

    command_text = (transcription.get("text") or "").strip()
    if forced_type:
        classification = {"command": forced_type}
    elif mode == "awaiting_confirmation":
        classification = {"command": "confirm_action"}
    else:
        classification = classify_command(command_text, {"current_mode": mode})

    request_type = classification.get("command", "ask")
    next_state = None

    if request_type in {"rename", "read_ocr", "export_ocr", "describe", "item_ask"}:
        intent = classification.get("intent")
        result, status = process_item_ask_request(scan_id=scan_id, label=label, query=command_text, intent=intent)
        request_type = "item_ask"
    elif request_type == "remember":
        image_file = request.files.get("image")
        image_files = request.files.getlist("images[]")
        prompt = (
            request.form.get("prompt")
            or classification.get("label")
            or _resolve_query_for_find(command_text, "")
        ).strip()
        result, status = process_remember_request(prompt=prompt, image_file=image_file, image_files=image_files)
        if status == 200 and isinstance(result, dict) and result.get("success"):
            next_state = "awaiting_location"
    elif request_type == "scan":
        result, status = process_scan_request(
            image_file=request.files.get("image"),
            focal_length_raw=request.form.get("focal_length_px", ""),
        )
    elif request_type == "find":
        result, status = _process_find_request(
            query=classification.get("query", command_text),
            label=label,
        )
    elif request_type == "set_location":
        result, status = _process_set_location_request(
            label=label,
            location_text=classification.get("location", command_text),
        )
        if status == 200:
            next_state = "idle"
    elif request_type == "confirm_action":
        result, status = _process_confirm_action_request(command_text=command_text, context=state_context)
        if status == 200:
            next_state = "idle"
    elif request_type == "stop":
        result, status = {
            "action": "stop",
            "stopped": True,
            "narration": "Okay, stopping.",
        }, 200
    else:
        result, status = process_ask_query(command_text)
        request_type = "ask"

    narration = result.get("narration", "") if isinstance(result, dict) else ""
    response = {
        "request_type": request_type,
        "transcription": command_text,
        "transcription_meta": transcription,
        "classification": classification,
        "result": result,
        "narration": narration,
        "next_state": next_state,
        "latency_ms": round((time.monotonic() - t0) * 1000),
    }
    if classification.get("intent"):
        response["intent"] = classification["intent"]
    _log.info({
        "event": "voice_dispatch",
        "request_type": request_type,
        "status": status,
        "mode": mode,
        "next_state": next_state,
    })
    return jsonify(response), status
