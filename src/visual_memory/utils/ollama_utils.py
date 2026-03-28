"""Lightweight Ollama helpers used by /ask and /item/ask routes.

All functions degrade gracefully when the `ollama` package is not installed
or the Ollama daemon is not running. Callers receive None and fall back to
pure embedding-based search.

Retry counts are read from Settings at call time so they can be patched
via PATCH /settings or PATCH /debug/config without a restart.
"""
from __future__ import annotations

_OLLAMA_MODEL = "llama3.2:1b"


def _get_max_retries() -> int:
    try:
        from visual_memory.config.settings import Settings
        return Settings().ollama_max_retries
    except Exception:
        return 2


def _chat(prompt: str, max_retries: int | None = None) -> str | None:
    """Send a single prompt to Ollama with retries. Returns response text or None on failure."""
    if max_retries is None:
        max_retries = _get_max_retries()
    for attempt in range(max(1, max_retries)):
        try:
            import ollama  # type: ignore
            response = ollama.chat(
                model=_OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return response["message"]["content"].strip()
        except Exception:
            if attempt == max(1, max_retries) - 1:
                return None
    return None


def extract_search_term(query: str) -> str | None:
    """Extract the core object/item the user is searching for from a natural language query.

    Examples:
        "where did I put my money last week?" -> "money"
        "the receipt with my office chair on it" -> "receipt office chair"
        "have you seen my blue keys?" -> "keys"

    Returns None if Ollama is unavailable; caller should embed the raw query instead.
    """
    prompt = (
        "Extract only the object or item the user is looking for from the query below. "
        "Reply with just the item name or a short phrase (2-4 words max). "
        "No punctuation, no explanation.\n\n"
        f"Query: {query}"
    )
    return _chat(prompt)


def extract_item_intent(query: str) -> str | None:
    """Classify the user's intent when asking about a specific focused item.

    Returns one of: read_ocr | export_ocr | rename | find | describe
    Returns None if Ollama is unavailable; caller should fall back to keyword matching.
    """
    prompt = (
        "Classify the user's request about an item they are currently looking at. "
        "Reply with exactly one of these words: read_ocr, export_ocr, rename, find, describe\n\n"
        "read_ocr   = user wants the text read aloud (e.g. 'read this', 'what does it say')\n"
        "export_ocr = user wants the text exported/copied (e.g. 'export', 'copy the text')\n"
        "rename     = user wants to rename the item (e.g. 'call this X', 'rename to X')\n"
        "find       = user wants the last known location (e.g. 'where is this normally', 'last seen')\n"
        "describe   = user wants a visual description (e.g. 'describe this', 'what is this')\n\n"
        f"Request: {query}"
    )
    result = _chat(prompt)
    valid = {"read_ocr", "export_ocr", "rename", "find", "describe"}
    if result and result.lower() in valid:
        return result.lower()
    return None


def extract_rename_target(query: str) -> str | None:
    """Extract the new name from a rename request.

    Examples:
        "rename this to my wallet" -> "my wallet"
        "call it house keys" -> "house keys"

    Returns None if Ollama is unavailable.
    """
    prompt = (
        "Extract only the new name the user wants to give to an item. "
        "Reply with just the new name, nothing else.\n\n"
        f"Request: {query}"
    )
    return _chat(prompt)
