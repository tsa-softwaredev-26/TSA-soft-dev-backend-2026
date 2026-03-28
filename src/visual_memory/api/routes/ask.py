"""POST /ask — open-ended natural language memory search.

The user asks anything about their stored world in plain speech.
Backend parses intent (via Ollama when available), runs semantic search
over labels and OCR text, and returns a narration string ready to speak.
"""
from flask import Blueprint, request, jsonify

from visual_memory.api.pipelines import get_database, get_scan_pipeline, get_settings
from visual_memory.api.routes.find import (
    _fuzzy_label_match,
    _ocr_content_match,
    _format_sighting,
    build_narration,
)
from visual_memory.utils.ollama_utils import extract_search_term
from visual_memory.utils import get_logger

_log = get_logger(__name__)

ask_bp = Blueprint("ask", __name__)


@ask_bp.post("/ask")
def ask():
    """Open-ended natural language memory search.

    Body (JSON):
        query  (str, required) - the user's raw spoken query

    Response (found):
        {
          "query": "where did I put my money",
          "search_term": "money",          -- extracted by Ollama, or raw query if unavailable
          "ollama_used": true,
          "found": true,
          "matched_label": "wallet",
          "matched_by": "fuzzy_label",     -- "exact" | "fuzzy_label" | "ocr"
          "narration": "Your wallet is in the kitchen, to your left. Last seen 5 minutes ago.",
          "last_sighting": { ... },
          "sightings": [ ... ]
        }

    Response (not found):
        {
          "query": "...",
          "search_term": "...",
          "ollama_used": bool,
          "found": false,
          "narration": "I couldn't find anything matching that in your memory."
        }
    """
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "missing field: query"}), 400

    settings = get_settings()
    db = get_database()

    # ---- Step 1: extract search term via Ollama ----
    search_term = extract_search_term(query)
    ollama_used = search_term is not None
    if not ollama_used:
        search_term = query

    _log.info({
        "event": "ask_search",
        "query": query,
        "search_term": search_term,
        "ollama_used": ollama_used,
    })

    # ---- Step 2: exact label match ----
    rows = db.get_sightings(label=search_term, limit=1)
    matched_label: str | None = None
    matched_by: str | None = None

    if rows:
        matched_label = search_term
        matched_by = "exact"

    # ---- Step 3: fuzzy label match ----
    if not rows:
        candidates = _fuzzy_label_match(search_term, settings.text_similarity_threshold)
        if candidates:
            matched_label = candidates[0]
            matched_by = "fuzzy_label"
            rows = db.get_sightings(label=matched_label, limit=1)

    # ---- Step 4: OCR content match ----
    if not rows:
        ocr_label = _ocr_content_match(search_term, settings.text_similarity_threshold)
        if ocr_label:
            matched_label = ocr_label
            matched_by = "ocr"
            rows = db.get_sightings(label=matched_label, limit=1)

    if not rows or matched_label is None:
        return jsonify({
            "query": query,
            "search_term": search_term,
            "ollama_used": ollama_used,
            "found": False,
            "narration": "I couldn't find anything matching that in your memory.",
        })

    sightings = [_format_sighting(r) for r in rows]
    narration = build_narration(matched_label, sightings[0])

    return jsonify({
        "query": query,
        "search_term": search_term,
        "ollama_used": ollama_used,
        "found": True,
        "matched_label": matched_label,
        "matched_by": matched_by,
        "narration": narration,
        "last_sighting": sightings[0],
        "sightings": sightings,
    })
