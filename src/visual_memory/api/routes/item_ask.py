"""POST /item/ask; item-context voice dispatcher.

The user is focused on a specific item (scrolling through scan results) and
asks something about it. Frontend provides scan_id + label so the backend
knows exactly which item is in focus; no search needed.

Supported actions:
    read_ocr   - read the OCR text aloud
    export_ocr - return OCR text for copy/export
    rename     - rename the item (auto-replace, no confirmation required)
    find       - last known location of this item
    describe   - visual description (deferred: requires VLM)
"""
import re
from flask import Blueprint, request, jsonify

from visual_memory.api.pipelines import get_database, get_scan_pipeline, get_settings
from visual_memory.api.routes.find import _format_sighting, build_narration
from visual_memory.utils.ollama_utils import (
    extract_item_intent,
    extract_rename_target,
)
from visual_memory.utils import get_logger

_log = get_logger(__name__)

item_ask_bp = Blueprint("item_ask", __name__)

# Keyword patterns used as fallback when Ollama is unavailable
_INTENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("export_ocr", re.compile(r"\b(export|copy|share|send)\b", re.IGNORECASE)),
    ("read_ocr",   re.compile(r"\b(read|text|say|says|written|write|content)\b", re.IGNORECASE)),
    ("rename",     re.compile(r"\b(rename|call|name)\b", re.IGNORECASE)),
    ("find",       re.compile(r"\b(where|find|location|last seen|last time)\b", re.IGNORECASE)),
    ("describe",   re.compile(r"\b(describe|what is|what's|look|looks like)\b", re.IGNORECASE)),
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


@item_ask_bp.post("/item/ask")
def item_ask():
    """Ask a question about a specific focused item.

    Body (JSON):
        scan_id  (str, required) - scan_id from the most recent POST /scan
        label    (str, required) - the label of the item in focus
        query    (str, required) - the user's spoken question

    Response varies by action:

    read_ocr / export_ocr:
        {
          "action": "read_ocr",
          "label": "wallet",
          "ocr_text": "RFID Blocking",
          "narration": "The text says: RFID Blocking.",
          "export": false
        }

    rename:
        {
          "action": "rename",
          "old_label": "wallet",
          "new_label": "my wallet",
          "narration": "Renamed to my wallet.",
          "replaced_existing": false
        }
        or if same label:
        {
          "action": "rename",
          "narration": "That's already called wallet.",
          "unchanged": true
        }

    find:
        {
          "action": "find",
          "label": "wallet",
          "found": bool,
          "narration": "...",
          "last_sighting": { ... }
        }

    describe:
        {
          "action": "describe",
          "narration": "Visual description is not available yet.",
          "deferred": true
        }
    """
    data = request.get_json(silent=True) or {}
    result, status = process_item_ask_request(
        scan_id=data.get("scan_id"),
        label=data.get("label"),
        query=data.get("query"),
    )
    return jsonify(result), status


def process_item_ask_request(scan_id: str, label: str, query: str, intent: str | None = None) -> tuple[dict, int]:
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
            resolved_intent = extract_item_intent(query)
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

    if resolved_intent in ("read_ocr", "export_ocr"):
        rows = db.get_items_metadata(label=label)
        ocr_text = rows[0]["ocr_text"] if rows else ""

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

    if resolved_intent == "rename":
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
        return {
            "action": "describe",
            "narration": "Visual description is not available yet.",
            "deferred": True,
        }, 200

    return {"error": f"unknown intent: {resolved_intent}"}, 400
