"""Voice transcription helpers for the unified /voice endpoint."""

import time

from flask import Blueprint, jsonify, request

from visual_memory.api.pipelines import get_database, get_settings
from visual_memory.engine.model_registry import registry
from visual_memory.utils import get_logger
from visual_memory.utils.audio_utils import load_audio_bytes, validate_audio_format
from visual_memory.utils.logger import LogTag

transcribe_bp = Blueprint("transcribe", __name__)
_log = get_logger(__name__)


def _build_context_policy(
    mode_hint: str,
    state_contract: dict | None,
    known_labels: list[str],
) -> tuple[str, list[str]]:
    policy = "default"
    mode_terms: list[str] = []
    contract_mode = ""
    if isinstance(state_contract, dict):
        contract_mode = str(state_contract.get("current_mode") or state_contract.get("mode") or "").strip().lower()
    mode = mode_hint or contract_mode
    if mode in {"focused_on_item"}:
        policy = "focused_item"
        mode_terms = [
            "describe", "read", "text", "says", "export", "copy",
            "rename", "wrong", "correct", "right", "where is this",
        ]
    elif mode in {"awaiting_location"}:
        policy = "awaiting_location"
        mode_terms = [
            "kitchen", "bedroom", "bathroom", "living room",
            "office", "garage", "hallway",
        ]
    elif mode in {"awaiting_confirmation"}:
        policy = "awaiting_confirmation"
        mode_terms = ["yes", "no", "confirm", "cancel", "correct", "wrong"]
    elif mode in {"onboarding_teach"}:
        policy = "onboarding_teach"
        mode_terms = ["teach", "remember", "save", "this is", "my"]
    elif mode in {"onboarding_await_scan", "awaiting_image"}:
        policy = "camera_capture"
        mode_terms = ["scan", "hold", "phone", "camera", "describe", "what i am looking at"]
    else:
        policy = "idle_home"
        mode_terms = [
            "scan", "teach", "remember", "find", "where",
            "describe", "settings", "export", "read",
        ]
    if known_labels and policy in {"focused_item", "idle_home"}:
        mode_terms.extend(known_labels[:4])
    return policy, mode_terms


def transcribe_audio_bytes(
    audio_bytes: bytes,
    use_context: bool = True,
    voice_mode: str | None = None,
    voice_context: dict | None = None,
) -> tuple[dict, int]:
    t0 = time.monotonic()
    settings = get_settings()

    if not audio_bytes:
        return {"error": "missing audio data"}, 400

    _log.info({
        "event": "transcribe_request",
        "tag": LogTag.API,
        "audio_size_bytes": len(audio_bytes),
        "context_requested": use_context,
        "voice_mode": (voice_mode or "").strip().lower() or None,
    })

    try:
        audio_array, sample_rate = load_audio_bytes(audio_bytes, target_sr=settings.whisper_sample_rate)
    except ValueError as exc:
        _, fmt = validate_audio_format(audio_bytes)
        return {
            "error": "invalid audio format",
            "detail": str(exc),
            "format_detected": fmt,
        }, 400
    except RuntimeError as exc:
        return {
            "error": "audio decoder unavailable",
            "detail": str(exc),
        }, 500

    context_prompt = None
    mode_hint = (voice_mode or "").strip().lower()
    recognizer = registry.get_whisper_recognizer()
    state_contract = None
    known_labels: list[str] = []
    if isinstance(voice_context, dict):
        contract = voice_context.get("state_contract")
        if isinstance(contract, dict):
            state_contract = contract
            raw_labels = contract.get("known_labels")
            if isinstance(raw_labels, list):
                known_labels = [str(x).strip() for x in raw_labels if str(x).strip()][:8]
    context_policy = "none"
    context_state_id = mode_hint or "idle"
    if use_context and settings.whisper_context_enabled:
        context_prompt = recognizer.build_context_prompt(get_database())
        context_policy, policy_terms = _build_context_policy(mode_hint, state_contract, known_labels)
        context_state_id = (
            str((state_contract or {}).get("current_mode") or mode_hint or "idle").strip().lower() or "idle"
        )
        extra_terms: list[str] = []
        if mode_hint:
            extra_terms.append(mode_hint)
        extra_terms.extend(policy_terms)
        if known_labels:
            extra_terms.extend(known_labels[:8])
        if extra_terms:
            context_prompt = f"{context_prompt} {' '.join(extra_terms)}".strip()

    try:
        registry.prepare_for_voice()
        result = recognizer.transcribe(audio_array, sample_rate=sample_rate, context=context_prompt)
    except Exception as exc:
        _log.error({
            "event": "transcribe_error",
            "tag": LogTag.API,
            "error": str(exc),
        })
        return {"error": "transcription failed", "detail": str(exc)}, 500
    finally:
        registry.prepare_after_voice()

    duration_ms = round((time.monotonic() - t0) * 1000)
    audio_duration_s = len(audio_array) / sample_rate if sample_rate else 0.0
    response = {
        "text": result["text"],
        "language": result["language"],
        "confidence": result["confidence"],
        "duration_ms": duration_ms,
        "audio_duration_s": round(audio_duration_s, 2),
        "context_used": bool(context_prompt),
        "context_policy": context_policy,
        "context_state_id": context_state_id,
        "model": settings.whisper_model,
    }
    _log.info({
        "event": "transcribe_success",
        "tag": LogTag.API,
        "text_length": len(result["text"]),
        "duration_ms": duration_ms,
        "audio_duration_s": round(audio_duration_s, 2),
        "context_used": bool(context_prompt),
    })
    return response, 200


@transcribe_bp.post("/transcribe")
def transcribe():
    audio_bytes = request.data
    use_context = request.args.get("context", "1") != "0"
    response, status = transcribe_audio_bytes(audio_bytes, use_context=use_context)
    return jsonify(response), status
