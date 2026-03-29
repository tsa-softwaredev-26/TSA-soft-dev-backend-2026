"""Lightweight Ollama helpers used by /ask and /item/ask routes.

All functions degrade gracefully when the `ollama` package is not installed
or the Ollama daemon is not running. Callers receive None and fall back to
pure embedding-based search.

Retry counts and timeouts are read from Settings at call time so they can be
patched via PATCH /settings or PATCH /debug/config without a restart.

Circuit breaker: after _CB_FAILURE_THRESHOLD consecutive failures the breaker
opens and all calls return None immediately for _CB_COOLDOWN_SECONDS. This
prevents a stalled Ollama daemon from adding latency to every request.
"""
from __future__ import annotations

import json
import os
import threading
import time

_OLLAMA_MODEL = "llama3.2:1b"

_CB_FAILURE_THRESHOLD = 3
_CB_COOLDOWN_SECONDS = 60.0

_cb_lock = threading.Lock()
_cb_state: dict = {"failures": 0, "opened_at": None}


def _cb_is_open() -> bool:
    with _cb_lock:
        opened_at = _cb_state["opened_at"]
        if opened_at is None:
            return False
        if time.time() - opened_at >= _CB_COOLDOWN_SECONDS:
            _cb_state["failures"] = 0
            _cb_state["opened_at"] = None
            return False
        return True


def _cb_record_failure() -> None:
    with _cb_lock:
        _cb_state["failures"] += 1
        if _cb_state["failures"] >= _CB_FAILURE_THRESHOLD:
            _cb_state["opened_at"] = time.time()


def _cb_record_success() -> None:
    with _cb_lock:
        _cb_state["failures"] = 0
        _cb_state["opened_at"] = None


def _get_max_retries() -> int:
    try:
        from visual_memory.config.settings import Settings
        return Settings().ollama_max_retries
    except Exception:
        return 2


def _get_timeout() -> float:
    try:
        from visual_memory.config.settings import Settings
        return Settings().ollama_timeout_seconds
    except Exception:
        return float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "5.0"))


def _get_host() -> str | None:
    """Return the Ollama host URL from env, or None to use the library default."""
    return os.environ.get("OLLAMA_HOST")


def _chat_raw(
    prompt: str,
    max_retries: int | None = None,
    json_mode: bool = False,
) -> str | None:
    """Send a prompt to Ollama. Returns response text or None on failure.

    When json_mode=True, passes format="json" to Ollama so the response is
    guaranteed to be valid JSON. Use this for structured extraction calls.
    """
    if _cb_is_open():
        return None

    if max_retries is None:
        max_retries = _get_max_retries()

    timeout = _get_timeout()
    host = _get_host()

    for attempt in range(max(1, max_retries)):
        try:
            import ollama  # type: ignore

            client_kwargs: dict = {"timeout": timeout}
            if host:
                client_kwargs["host"] = host
            client = ollama.Client(**client_kwargs)

            chat_kwargs: dict = {
                "model": _OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
            }
            if json_mode:
                chat_kwargs["format"] = "json"

            response = client.chat(**chat_kwargs)
            _cb_record_success()
            return response["message"]["content"].strip()

        except Exception:
            _cb_record_failure()
            if attempt == max(1, max_retries) - 1:
                return None

    return None


def _chat(prompt: str, max_retries: int | None = None) -> str | None:
    """Plain-text chat call. Used by remember pipeline for third-pass detection."""
    return _chat_raw(prompt, max_retries, json_mode=False)


def extract_search_term(query: str) -> str | None:
    """Extract the core item the user is searching for from a natural language query.

    Examples:
        "where did I put my money last week?" -> "money"
        "the receipt with my office chair on it" -> "receipt office chair"
        "have you seen my blue keys?" -> "keys"

    Returns None if Ollama is unavailable; caller should search with the raw query.
    """
    prompt = (
        "Extract the object or item the user is looking for. "
        "Respond with JSON only.\n"
        'Output format: {"term": "<item name, 1-4 words>"}\n'
        "No punctuation, no explanation, just the JSON.\n\n"
        f"Query: {query}"
    )
    raw = _chat_raw(prompt, json_mode=True)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        term = str(data.get("term", "")).strip()
        return term if term else None
    except (json.JSONDecodeError, AttributeError, TypeError):
        return None


def extract_item_intent(query: str) -> str | None:
    """Classify the user's intent when asking about a specific focused item.

    Returns one of: read_ocr | export_ocr | rename | find | describe
    Returns None if Ollama is unavailable; caller falls back to keyword matching.
    """
    prompt = (
        "Classify the user's request about an item they are looking at. "
        "Respond with JSON only.\n"
        'Output format: {"intent": "<value>"}\n'
        "Valid values:\n"
        "  read_ocr   - user wants text read aloud (e.g. 'read this', 'what does it say')\n"
        "  export_ocr - user wants text exported or copied (e.g. 'export', 'copy the text')\n"
        "  rename     - user wants to rename the item (e.g. 'call this X', 'rename to X')\n"
        "  find       - user wants last known location (e.g. 'where is this normally')\n"
        "  describe   - user wants a visual description (e.g. 'describe this', 'what is this')\n\n"
        f"Request: {query}"
    )
    raw = _chat_raw(prompt, json_mode=True)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        intent = str(data.get("intent", "")).strip().lower()
        if intent in {"read_ocr", "export_ocr", "rename", "find", "describe"}:
            return intent
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    return None


def extract_rename_target(query: str) -> str | None:
    """Extract the new name from a rename request.

    Examples:
        "rename this to my wallet" -> "my wallet"
        "call it house keys" -> "house keys"

    Returns None if Ollama is unavailable or no name is found.
    """
    prompt = (
        "Extract the new name the user wants to give to an item. "
        "Respond with JSON only.\n"
        'Output format: {"name": "<new name>"} or {"name": null} if unclear.\n'
        "No punctuation around the name, no explanation.\n\n"
        f"Request: {query}"
    )
    raw = _chat_raw(prompt, json_mode=True)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        name = data.get("name")
        if name is None:
            return None
        name = str(name).strip()
        return name if name else None
    except (json.JSONDecodeError, AttributeError, TypeError):
        return None
