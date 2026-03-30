"""Natural language memory search helpers."""

from flask import Blueprint

from visual_memory.api.pipelines import get_database, get_settings
from visual_memory.api.routes.find import _fuzzy_label_match, _format_sighting, _ocr_content_match, build_narration
from visual_memory.utils import get_logger
from visual_memory.utils.ollama_utils import extract_search_term, is_unsafe_query

ask_bp = Blueprint("ask", __name__)
_log = get_logger(__name__)


def process_ask_query(raw_query) -> tuple[dict, int]:
    if raw_query is None:
        return {"error": "missing field: query"}, 400
    if not isinstance(raw_query, str):
        return {"error": "query must be a string"}, 400

    query = raw_query.strip()
    if not query:
        return {"error": "missing field: query"}, 400

    if is_unsafe_query(query):
        _log.warning({"event": "ask_blocked_unsafe_query", "query": query})
        return {
            "query": query,
            "search_term": None,
            "ollama_used": False,
            "found": False,
            "blocked": True,
            "reason": "unsafe_query",
            "narration": "I can only help with memory-related object lookup requests.",
        }, 400

    settings = get_settings()
    db = get_database()

    rows = db.get_sightings(label=query, limit=1)
    matched_label: str | None = None
    matched_by: str | None = None
    ollama_used = False
    search_term = query

    if rows:
        matched_label = query
        matched_by = "exact"

    if not rows and settings.llm_query_fallback_enabled:
        extracted = extract_search_term(query)
        if extracted and extracted.lower() != query.lower():
            ollama_used = True
            search_term = extracted
        if ollama_used:
            rows = db.get_sightings(label=search_term, limit=1)
            if rows:
                matched_label = search_term
                matched_by = "exact"

    _log.info({
        "event": "ask_search",
        "query": query,
        "search_term": search_term,
        "ollama_used": ollama_used,
    })

    if not rows:
        candidates = _fuzzy_label_match(search_term, settings.text_similarity_threshold)
        if candidates:
            matched_label = candidates[0]
            matched_by = "fuzzy_label"
            rows = db.get_sightings(label=matched_label, limit=1)

    if not rows:
        ocr_label = _ocr_content_match(search_term, settings.text_similarity_threshold)
        if ocr_label:
            matched_label = ocr_label
            matched_by = "ocr"
            rows = db.get_sightings(label=matched_label, limit=1)

    if not rows or matched_label is None:
        return {
            "query": query,
            "search_term": search_term,
            "ollama_used": ollama_used,
            "found": False,
            "narration": "I couldn't find anything matching that in your memory.",
        }, 200

    sightings = [_format_sighting(r) for r in rows]
    return {
        "query": query,
        "search_term": search_term,
        "ollama_used": ollama_used,
        "found": True,
        "matched_label": matched_label,
        "matched_by": matched_by,
        "narration": build_narration(matched_label, sightings[0]),
        "last_sighting": sightings[0],
        "sightings": sightings,
    }, 200
