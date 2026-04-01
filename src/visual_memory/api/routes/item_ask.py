"""POST /item/ask; item-context voice dispatcher.

The user is focused on a specific item (scrolling through scan results) and
asks something about it. Frontend provides scan_id + label so the backend
knows exactly which item is in focus; no search needed.

Supported actions:
    read_ocr   - read the OCR text aloud
    export_ocr - return OCR text for copy/export
    rename     - rename the item (auto-replace, no confirmation required)
    find       - last known location of this item
    describe   - visual description with fallback cascade
"""
import re
import time
from flask import Blueprint, request, jsonify

from visual_memory.api.pipelines import get_database, get_scan_pipeline, get_settings, get_user_settings
from visual_memory.api.routes.find import _format_sighting, build_narration
from visual_memory.engine.visual_attributes import describe_from_attributes
from visual_memory.engine.vlm import get_vlm_pipeline
from visual_memory.utils.ollama_utils import (
    extract_item_intent,
    extract_rename_target,
)
from visual_memory.utils import get_logger
from ._json_utils import read_json_dict

_log = get_logger(__name__)

item_ask_bp = Blueprint("item_ask", __name__)

# Keyword patterns used as fallback when Ollama is unavailable
_INTENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("ocr_question", re.compile(r"\b(?:who|when|how much|total|amount|date)\b.*\b(?:written|text|receipt|document|say|says)\b", re.IGNORECASE)),
    ("question", re.compile(r"\b(what color|how many|is there|can you see|does it have)\b", re.IGNORECASE)),
    ("export_ocr", re.compile(r"\b(export|copy|share|send)\b", re.IGNORECASE)),
    ("read_ocr", re.compile(r"\b(read|text|say|says|written|write|content)\b", re.IGNORECASE)),
    ("rename", re.compile(r"\b(rename|call|name)\b", re.IGNORECASE)),
    ("find", re.compile(r"\b(where|find|location|last seen|last time)\b", re.IGNORECASE)),
    ("describe", re.compile(r"\b(describe|what is|what's|look|looks like)\b", re.IGNORECASE)),
]


def _keyword_intent(query: str) -> str | None:
    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(query):
            return intent
    return None


def _extract_rename_target_keyword(query: str) -> str | None:
    """Keyword fallback: extract text after 'to', 'as', or 'it' in a rename request."""
    match = re.search(
        r"\b(?:rename|call|name)\b.*?\b(?:to|as|it)\b\s+(.+)",
        query,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip().strip("\"'")
    return None


def _minimal_description(label: str, ocr_text: str) -> str:
    text = (ocr_text or "").strip()
    if text:
        clipped = text[:120]
        if len(text) > 120:
            clipped = f"{clipped}..."
        return f"This is your {label}. I can read text on it: {clipped}"
    return f"This is your {label}."


@item_ask_bp.post("/item/ask")
def item_ask():
    data, err = read_json_dict(request)
    if err is not None:
        body, status = err
        return jsonify(body), status
    result, status = process_item_ask_request(
        scan_id=data.get("scan_id"),
        label=data.get("label"),
        query=data.get("query"),
    )
    return jsonify(result), status


def process_item_ask_request(
    scan_id: str,
    label: str,
    query: str,
    intent: str | None = None,
    state_context: dict | None = None,
) -> tuple[dict, int]:
    scan_id = (scan_id or "").strip()
    label = (label or "").strip()
    query = (query or "").strip()

    if not scan_id:
        return {"error": "missing field: scan_id"}, 400
    if not label:
        return {"error": "missing field: label"}, 400
    if not query:
        return {"error": "missing field: query"}, 400

    resolved_intent = intent
    ollama_used = False
    if resolved_intent is None:
        resolved_intent = _keyword_intent(query)
        if resolved_intent is None and get_settings().llm_query_fallback_enabled:
            resolved_intent = extract_item_intent(query, state_context=state_context)
            if resolved_intent is not None:
                ollama_used = True
        if resolved_intent is None:
            resolved_intent = "read_ocr"

    _log.info({
        "event": "item_ask",
        "label": label,
        "intent": resolved_intent,
        "ollama_used": ollama_used,
        "query": query,
    })

    db = get_database()

    # read_ocr / export_ocr
    if resolved_intent in ("read_ocr", "export_ocr"):
        rows = db.get_items_metadata(label=label)
        item = rows[0] if rows else {}
        ocr_text = item.get("ocr_text", "") if isinstance(item, dict) else ""

        if not ocr_text:
            return {
                "action": resolved_intent,
                "label": label,
                "ocr_text": "",
                "narration": f"There's no text stored for {label}.",
                "export": resolved_intent == "export_ocr",
            }, 200

        return {
            "action": resolved_intent,
            "label": label,
            "ocr_text": ocr_text,
            "narration": f"The text says: {ocr_text}",
            "export": resolved_intent == "export_ocr",
        }, 200


    if resolved_intent == "question":
        t0 = time.monotonic()
        rows = db.get_items_metadata(label=label)
        if not rows:
            return {
                "action": "question",
                "label": label,
                "question": query,
                "answer": "",
                "narration": f"I do not have {label} in memory yet.",
                "method": "vlm",
                "latency_ms": round((time.monotonic() - t0) * 1000),
            }, 404

        item = rows[0] if rows else {}
        image_path = (item.get("image_path") or "").strip() if isinstance(item, dict) else ""
        if not image_path:
            return {
                "action": "question",
                "label": label,
                "question": query,
                "answer": "",
                "narration": f"I do not have an image stored for {label}.",
                "method": "vlm",
                "latency_ms": round((time.monotonic() - t0) * 1000),
            }, 200

        vlm_timeout = float(get_user_settings().get_performance_config().vlm_timeout_seconds)
        try:
            answer = get_vlm_pipeline().answer(image_path, question=query, timeout=vlm_timeout)
            return {
                "action": "question",
                "label": label,
                "question": query,
                "answer": answer,
                "narration": answer,
                "method": "vlm",
                "latency_ms": round((time.monotonic() - t0) * 1000),
            }, 200
        except TimeoutError:
            return {
                "action": "question",
                "label": label,
                "question": query,
                "answer": "",
                "narration": "That took too long. Please try a shorter question.",
                "method": "vlm_timeout",
                "latency_ms": round((time.monotonic() - t0) * 1000),
            }, 504
        except FileNotFoundError:
            return {
                "action": "question",
                "label": label,
                "question": query,
                "answer": "",
                "narration": f"I could not find the image file for {label}.",
                "method": "vlm",
                "latency_ms": round((time.monotonic() - t0) * 1000),
            }, 404
        except Exception as exc:
            _log.warning({
                "event": "item_ask_question_vlm_error",
                "label": label,
                "error": str(exc),
            })
            return {
                "action": "question",
                "label": label,
                "question": query,
                "answer": "",
                "narration": "I could not answer that question about this item right now.",
                "method": "vlm_error",
                "latency_ms": round((time.monotonic() - t0) * 1000),
            }, 200

    if resolved_intent == "ocr_question":
        rows = db.get_items_metadata(label=label)
        item = rows[0] if rows else {}
        ocr_text = item.get("ocr_text", "") if isinstance(item, dict) else ""

        if not ocr_text:
            return {
                "action": "ocr_question",
                "label": label,
                "question": query,
                "ocr_text": "",
                "answer": "",
                "narration": f"There is no stored text for {label}.",
            }, 200

        normalized = ocr_text.strip()
        if re.search(r"\b(total|amount|how much)\b", query, re.IGNORECASE):
            amount_match = re.search(r"(?:(?:USD|EUR|GBP)\s*)?(?:\$\s?)?\d+(?:[\.,]\d{2})?", normalized, re.IGNORECASE)
            if amount_match:
                answer = amount_match.group(0)
                return {
                    "action": "ocr_question",
                    "label": label,
                    "question": query,
                    "ocr_text": normalized,
                    "answer": answer,
                    "narration": f"The amount is {answer}.",
                }, 200

        if re.search(r"\bdate\b", query, re.IGNORECASE):
            date_match = re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", normalized)
            if date_match:
                answer = date_match.group(0)
                return {
                    "action": "ocr_question",
                    "label": label,
                    "question": query,
                    "ocr_text": normalized,
                    "answer": answer,
                    "narration": f"The date is {answer}.",
                }, 200

        return {
            "action": "ocr_question",
            "label": label,
            "question": query,
            "ocr_text": normalized,
            "answer": normalized,
            "narration": f"The text says: {normalized}",
        }, 200

    if resolved_intent == "rename":
        new_label = _extract_rename_target_keyword(query)
        if new_label is None and get_settings().llm_query_fallback_enabled:
            new_label = extract_rename_target(query, state_context=state_context)
        if not new_label:
            return {
                "action": "rename",
                "narration": "I couldn't understand the new name. Try saying: rename this to [name].",
                "error": "new_label_not_found",
            }, 400

        new_label = new_label.strip()

        if new_label.lower() == label.lower():
            return {
                "action": "rename",
                "narration": f"That's already called {label}.",
                "unchanged": True,
            }, 200

        result = db.rename_label(label, new_label)
        try:
            get_scan_pipeline().reload_database()
        except Exception as exc:
            return {
                "action": "rename",
                "error": "database sync failed",
                "detail": str(exc),
                "narration": "Renamed, but I could not sync scan state yet.",
            }, 500

        return {
            "action": "rename",
            "old_label": label,
            "new_label": new_label,
            "narration": f"Renamed to {new_label}.",
            "replaced_existing": result["replaced"] > 0,
        }, 200

    if resolved_intent == "find":
        row = db.get_last_sighting(label)
        if row is None:
            return {
                "action": "find",
                "label": label,
                "found": False,
                "narration": f"I haven't seen {label} in any known location yet.",
            }, 200
        sighting = _format_sighting(row)
        return {
            "action": "find",
            "label": label,
            "found": True,
            "narration": build_narration(label, sighting),
            "last_sighting": sighting,
        }, 200

    if resolved_intent == "describe":
        t0 = time.monotonic()
        rows = db.get_items_metadata(label=label)
        if not rows:
            return {
                "action": "describe",
                "label": label,
                "narration": f"I do not have {label} in memory yet.",
                "method": "minimal",
                "latency_ms": round((time.monotonic() - t0) * 1000),
            }, 404

        item = rows[0]
        image_path = (item.get("image_path") or "").strip()
        attributes = item.get("visual_attributes") or {}
        ocr_text = item.get("ocr_text") or ""

        perf_cfg = get_user_settings().get_performance_config()
        vlm_enabled = bool(perf_cfg.vlm_enabled)
        vlm_timeout = float(perf_cfg.vlm_timeout_seconds)

        if vlm_enabled and image_path:
            try:
                narration = get_vlm_pipeline().describe(image_path, timeout=vlm_timeout)
                return {
                    "action": "describe",
                    "label": label,
                    "narration": narration,
                    "method": "vlm",
                    "latency_ms": round((time.monotonic() - t0) * 1000),
                }, 200
            except TimeoutError:
                _log.warning({
                    "event": "item_ask_describe_vlm_timeout",
                    "label": label,
                    "timeout": vlm_timeout,
                })
            except Exception as exc:
                _log.warning({
                    "event": "item_ask_describe_vlm_error",
                    "label": label,
                    "error": str(exc),
                })

        if attributes:
            return {
                "action": "describe",
                "label": label,
                "narration": describe_from_attributes(label, attributes),
                "method": "attributes",
                "latency_ms": round((time.monotonic() - t0) * 1000),
            }, 200

        return {
            "action": "describe",
            "label": label,
            "narration": _minimal_description(label, ocr_text),
            "method": "minimal",
            "latency_ms": round((time.monotonic() - t0) * 1000),
        }, 200

    return {"error": f"unknown intent: {resolved_intent}"}, 400
