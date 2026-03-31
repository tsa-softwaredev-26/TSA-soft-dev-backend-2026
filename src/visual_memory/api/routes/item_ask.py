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


<<<<<<< HEAD
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
=======
def process_item_ask_request(scan_id: str, label: str, query: str) -> tuple[dict, int]:
>>>>>>> api/whisper-update
    scan_id = (scan_id or "").strip()
    label = (label or "").strip()
    query = (query or "").strip()

    if not scan_id:
        return {"error": "missing field: scan_id"}, 400
    if not label:
        return {"error": "missing field: label"}, 400
    if not query:
        return {"error": "missing field: query"}, 400

<<<<<<< HEAD
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
=======
    intent = _keyword_intent(query)
    ollama_used = False
    if intent is None and get_settings().llm_query_fallback_enabled:
        intent = extract_item_intent(query)
        if intent is not None:
            ollama_used = True
    if intent is None:
        intent = "read_ocr"
>>>>>>> api/whisper-update

    _log.info({
        "event": "item_ask",
        "label": label,
        "intent": resolved_intent,
        "ollama_used": ollama_used,
        "query": query,
    })

    db = get_database()

<<<<<<< HEAD
    if resolved_intent in ("read_ocr", "export_ocr"):
=======
    if intent in ("read_ocr", "export_ocr"):
>>>>>>> api/whisper-update
        rows = db.get_items_metadata(label=label)
        ocr_text = rows[0]["ocr_text"] if rows else ""
        if not ocr_text:
            return {
<<<<<<< HEAD
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
=======
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
>>>>>>> api/whisper-update
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
<<<<<<< HEAD

=======
>>>>>>> api/whisper-update
        return {
            "action": "rename",
            "old_label": label,
            "new_label": new_label,
            "narration": f"Renamed to {new_label}.",
            "replaced_existing": result["replaced"] > 0,
        }, 200

<<<<<<< HEAD
    if resolved_intent == "find":
=======
    if intent == "find":
>>>>>>> api/whisper-update
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

<<<<<<< HEAD
    if resolved_intent == "describe":
=======
    if intent == "describe":
>>>>>>> api/whisper-update
        return {
            "action": "describe",
            "narration": "Visual description is not available yet.",
            "deferred": True,
        }, 200

<<<<<<< HEAD
    return {"error": f"unknown intent: {resolved_intent}"}, 400
=======
    return {"error": f"unknown intent: {intent}"}, 400
>>>>>>> api/whisper-update
