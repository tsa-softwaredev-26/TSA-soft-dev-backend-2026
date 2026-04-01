"""POST /ask; open-ended natural language memory search.

The user asks anything about their stored world in plain speech.
Backend parses intent (via Ollama when available), runs semantic search
over labels and OCR text, and returns a narration string ready to speak.
"""
from flask import Blueprint, request, jsonify

from visual_memory.api.pipelines import get_database, get_settings
from visual_memory.api.routes.find import (
    _fuzzy_label_match,
    _ocr_content_match,
    _format_sighting,
    build_narration,
    is_document_query,
)
from visual_memory.utils.ollama_utils import extract_search_term, is_unsafe_query
from visual_memory.utils import get_logger
from visual_memory.utils.voice_state_context import build_state_contract
from ._json_utils import read_json_dict

_log = get_logger(__name__)

ask_bp = Blueprint("ask", __name__)


def process_ask_query(raw_query, state_context: dict | None = None) -> tuple[dict, int]:
    if raw_query is None:
        return {"error": "missing field: query"}, 400
    if not isinstance(raw_query, str):
        return {"error": "query must be a string"}, 400
    query = raw_query.strip()
    if not query:
        return {"error": "missing field: query"}, 400

    if is_unsafe_query(query):
        _log.warning({
            "event": "ask_blocked_unsafe_query",
            "query": query,
        })
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

    rows = None
    matched_label: str | None = None
    matched_by: str | None = None
    ollama_used = False
    search_term = query
    strategies_tried: list[str] = []

    # Step 1: LLM query extraction as preprocessing (when enabled)
    if settings.llm_query_fallback_enabled:
        extracted = extract_search_term(query, state_context=state_context)
        if extracted:
            search_term = extracted
            strategies_tried.append("llm_extraction")
            if extracted.lower() != query.lower():
                ollama_used = True

    # Step 2: exact label match on preprocessed term
    rows = db.get_sightings(label=search_term, limit=1)
    strategies_tried.append("exact")
    if rows:
        matched_label = search_term
        matched_by = "exact"

    _log.info({
        "event": "ask_search",
        "query": query,
        "search_term": search_term,
        "ollama_used": ollama_used,
        "strategies_tried": strategies_tried,
    })

    # Step 3: complexity classification and primary strategy
    document_query = is_document_query(query) or is_document_query(search_term)
    primary_strategy = "ocr_semantic" if document_query else "fuzzy_label"
    fallback_strategy = "fuzzy_label" if document_query else "ocr_semantic"

    if not rows:
        if primary_strategy == "ocr_semantic":
            ocr_label = _ocr_content_match(search_term, settings.text_similarity_threshold)
            strategies_tried.append("ocr_semantic")
            if ocr_label:
                matched_label = ocr_label
                matched_by = "ocr_semantic"
                rows = db.get_sightings(label=matched_label, limit=1)
        else:
            candidates = _fuzzy_label_match(search_term, settings.text_similarity_threshold)
            strategies_tried.append("fuzzy_label")
            if candidates:
                matched_label = candidates[0]
                matched_by = "fuzzy_label"
                rows = db.get_sightings(label=matched_label, limit=1)

    # Step 4: cross-strategy fallback
    if not rows:
        if fallback_strategy == "ocr_semantic":
            ocr_label = _ocr_content_match(search_term, settings.text_similarity_threshold)
            strategies_tried.append("ocr_semantic")
            if ocr_label:
                matched_label = ocr_label
                matched_by = "ocr_semantic_fallback"
                rows = db.get_sightings(label=matched_label, limit=1)
        else:
            candidates = _fuzzy_label_match(search_term, settings.text_similarity_threshold)
            strategies_tried.append("fuzzy_label")
            if candidates:
                matched_label = candidates[0]
                matched_by = "fuzzy_label_fallback"
                rows = db.get_sightings(label=matched_label, limit=1)

    if not rows or matched_label is None:
        _log.info({
            "event": "ask_search_miss",
            "query": query,
            "search_term": search_term,
            "document_query": document_query,
            "primary_strategy": primary_strategy,
            "strategies_tried": strategies_tried,
        })
        return {
            "query": query,
            "search_term": search_term,
            "ollama_used": ollama_used,
            "found": False,
            "strategies_tried": strategies_tried,
            "narration": "I couldn't find anything matching that in your memory.",
        }, 200

    sightings = [_format_sighting(r) for r in rows]
    narration = build_narration(matched_label, sightings[0])
    _log.info({
        "event": "ask_search_hit",
        "query": query,
        "search_term": search_term,
        "document_query": document_query,
        "matched_label": matched_label,
        "matched_by": matched_by,
        "strategies_tried": strategies_tried,
    })

    return {
        "query": query,
        "search_term": search_term,
        "ollama_used": ollama_used,
        "found": True,
        "document_query": document_query,
        "matched_label": matched_label,
        "matched_by": matched_by,
        "strategies_tried": strategies_tried,
        "narration": narration,
        "last_sighting": sightings[0],
        "sightings": sightings,
    }, 200


@ask_bp.post("/ask")
def ask():
    data, err = read_json_dict(request)
    if err is not None:
        body, status = err
        return jsonify(body), status
    state = data.get("state") if isinstance(data.get("state"), dict) else {}
    mode = data.get("current_mode") or data.get("mode")
    if not mode and isinstance(state, dict):
        mode = state.get("current_mode") or state.get("mode")
    context = data.get("context")
    if not isinstance(context, dict) and isinstance(state, dict):
        context = state.get("context")
    state_contract = build_state_contract(mode=mode, context=context)
    result, status = process_ask_query(data.get("query"), state_context=state_contract)
    return jsonify(result), status
