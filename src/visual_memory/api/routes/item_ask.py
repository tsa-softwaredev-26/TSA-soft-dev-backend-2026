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
          "narration": "...",
          "method": "vlm" | "attributes" | "minimal",
          "latency_ms": int
        }
    """
    data = request.get_json(silent=True) or {}
    scan_id = (data.get("scan_id") or "").strip()
    label = (data.get("label") or "").strip()
    query = (data.get("query") or "").strip()

    if not scan_id:
        return jsonify({"error": "missing field: scan_id"}), 400
    if not label:
        return jsonify({"error": "missing field: label"}), 400
    if not query:
        return jsonify({"error": "missing field: query"}), 400

    # Resolve intent: keywords first, Ollama for ambiguous cases only
    # Keywords are unambiguous signals; use them as the primary classifier.
    # Ollama supplements only when no keyword pattern fires.
    intent = _keyword_intent(query)
    ollama_used = False
    if intent is None and get_settings().llm_query_fallback_enabled:
        intent = extract_item_intent(query)
        if intent is not None:
            ollama_used = True
    if intent is None:
        intent = "read_ocr"  # safest default for an unknown voice command

    _log.info({
        "event": "item_ask",
        "label": label,
        "intent": intent,
        "ollama_used": ollama_used,
        "query": query,
    })

    db = get_database()

    # read_ocr / export_ocr
    if intent in ("read_ocr", "export_ocr"):
        rows = db.get_items_metadata(label=label)
        ocr_text = rows[0]["ocr_text"] if rows else ""

        if not ocr_text:
            return jsonify({
                "action": intent,
                "label": label,
                "ocr_text": "",
                "narration": f"There's no text stored for {label}.",
                "export": intent == "export_ocr",
            })

        return jsonify({
            "action": intent,
            "label": label,
            "ocr_text": ocr_text,
            "narration": f"The text says: {ocr_text}",
            "export": intent == "export_ocr",
        })

    # rename
    if intent == "rename":
        # Keyword extraction first (deterministic), Ollama as fallback
        new_label = _extract_rename_target_keyword(query)
        if new_label is None and get_settings().llm_query_fallback_enabled:
            new_label = extract_rename_target(query)
        if not new_label:
            return jsonify({
                "action": "rename",
                "narration": "I couldn't understand the new name. Try saying: rename this to [name].",
                "error": "new_label_not_found",
            }), 400

        new_label = new_label.strip()

        if new_label.lower() == label.lower():
            return jsonify({
                "action": "rename",
                "narration": f"That's already called {label}.",
                "unchanged": True,
            })

        result = db.rename_label(label, new_label)
        # Reload scan pipeline DB so the rename is immediately visible in scans
        get_scan_pipeline().reload_database()

        return jsonify({
            "action": "rename",
            "old_label": label,
            "new_label": new_label,
            "narration": f"Renamed to {new_label}.",
            "replaced_existing": result["replaced"] > 0,
        })

    # find
    if intent == "find":
        row = db.get_last_sighting(label)
        if row is None:
            return jsonify({
                "action": "find",
                "label": label,
                "found": False,
                "narration": f"I haven't seen {label} in any known location yet.",
            })
        sighting = _format_sighting(row)
        return jsonify({
            "action": "find",
            "label": label,
            "found": True,
            "narration": build_narration(label, sighting),
            "last_sighting": sighting,
        })

    # describe (3-tier fallback: VLM -> attributes -> minimal)
    if intent == "describe":
        t0 = time.monotonic()
        rows = db.get_items_metadata(label=label)
        if not rows:
            return jsonify({
                "action": "describe",
                "label": label,
                "narration": f"I do not have {label} in memory yet.",
                "method": "minimal",
                "latency_ms": round((time.monotonic() - t0) * 1000),
            }), 404

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
                return jsonify({
                    "action": "describe",
                    "label": label,
                    "narration": narration,
                    "method": "vlm",
                    "latency_ms": round((time.monotonic() - t0) * 1000),
                })
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
            return jsonify({
                "action": "describe",
                "label": label,
                "narration": describe_from_attributes(label, attributes),
                "method": "attributes",
                "latency_ms": round((time.monotonic() - t0) * 1000),
            })

        return jsonify({
            "action": "describe",
            "label": label,
            "narration": _minimal_description(label, ocr_text),
            "method": "minimal",
            "latency_ms": round((time.monotonic() - t0) * 1000),
        })

    # Unreachable but defensive
    return jsonify({"error": f"unknown intent: {intent}"}), 400
