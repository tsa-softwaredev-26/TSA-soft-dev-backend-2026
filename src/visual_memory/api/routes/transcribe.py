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


def transcribe_audio_bytes(audio_bytes: bytes, use_context: bool = True) -> tuple[dict, int]:
    t0 = time.monotonic()
    settings = get_settings()

    if not audio_bytes:
        return {"error": "missing audio data"}, 400

    _log.info({
        "event": "transcribe_request",
        "tag": LogTag.API,
        "audio_size_bytes": len(audio_bytes),
        "context_requested": use_context,
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
    recognizer = registry.get_whisper_recognizer()
    if use_context and settings.whisper_context_enabled:
        context_prompt = recognizer.build_context_prompt(get_database())

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
