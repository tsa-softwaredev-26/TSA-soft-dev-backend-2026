import re

from flask import Blueprint

from visual_memory.api.pipelines import get_database, get_scan_pipeline, get_settings
from visual_memory.api.routes.find import _format_sighting, build_narration
from visual_memory.utils import get_logger
from visual_memory.utils.ollama_utils import extract_item_intent, extract_rename_target

item_ask_bp = Blueprint("item_ask", __name__)
_log = get_logger(__name__)

_INTENT_PATTERNS: list[tuple[str, re.Pattern]] = [
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
    match = re.search(r"\b(?:rename|call|name)\b.*?\b(?:to|as|it)\b\s+(.+)", query, re.IGNORECASE)
    if match:
        return match.group(1).strip().strip('"\'')
    return None


def process_item_ask_request(scan_id: str, label: str, query: str) -> tuple[dict, int]:
    scan_id = (scan_id or "").strip()
    label = (label or "").strip()
    query = (query or "").strip()

    if not scan_id:
        return {"error": "missing field: scan_id"}, 400
    if not label:
        return {"error": "missing field: label"}, 400
    if not query:
        return {"error": "missing field: query"}, 400

    intent = _keyword_intent(query)
    ollama_used = False
    if intent is None and get_settings().llm_query_fallback_enabled:
        intent = extract_item_intent(query)
        if intent is not None:
            ollama_used = True
    if intent is None:
        intent = "read_ocr"

    _log.info({
        "event": "item_ask",
        "label": label,
        "intent": intent,
        "ollama_used": ollama_used,
        "query": query,
    })

    db = get_database()

    if intent in ("read_ocr", "export_ocr"):
        rows = db.get_items_metadata(label=label)
        ocr_text = rows[0]["ocr_text"] if rows else ""
        if not ocr_text:
            return {
                "action": intent,
                "label": label,
                "ocr_text": "",
                "narration": f"There's no text stored for {label}.",
                "export": intent == "export_ocr",
            }, 200
        return {
            "action": intent,
            "label": label,
            "ocr_text": ocr_text,
            "narration": f"The text says: {ocr_text}",
            "export": intent == "export_ocr",
        }, 200

    if intent == "rename":
        new_label = _extract_rename_target_keyword(query)
        if new_label is None and get_settings().llm_query_fallback_enabled:
            new_label = extract_rename_target(query)
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
        get_scan_pipeline().reload_database()
        return {
            "action": "rename",
            "old_label": label,
            "new_label": new_label,
            "narration": f"Renamed to {new_label}.",
            "replaced_existing": result["replaced"] > 0,
        }, 200

    if intent == "find":
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

    if intent == "describe":
        return {
            "action": "describe",
            "narration": "Visual description is not available yet.",
            "deferred": True,
        }, 200

    return {"error": f"unknown intent: {intent}"}, 400
