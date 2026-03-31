"""
Unit tests for ollama_utils: extract_search_term, extract_item_intent,
extract_rename_target, circuit breaker, retry behavior.

All tests mock _chat_raw - no Ollama service required.
"""
from __future__ import annotations

import json
import os
import sys
import time
import threading

os.environ["ENABLE_DEPTH"] = "0"

from unittest.mock import patch

from visual_memory.tests.scripts.test_harness import TestRunner
import visual_memory.utils.ollama_utils as _ou

_runner = TestRunner("ollama_utils")


def _reset_cb():
    """Reset circuit breaker state between tests."""
    with _ou._cb_lock:
        _ou._cb_state["failures"] = 0
        _ou._cb_state["opened_at"] = None


def test_extract_search_term_valid():
    with patch.object(_ou, "_chat_raw", return_value='{"term": "wallet"}'):
        result = _ou.extract_search_term("where did I put my wallet?")
    assert result == "wallet"


def test_extract_search_term_none_on_failure():
    with patch.object(_ou, "_chat_raw", return_value=None):
        result = _ou.extract_search_term("query")
    assert result is None


def test_extract_search_term_malformed_json():
    with patch.object(_ou, "_chat_raw", return_value="not json"):
        result = _ou.extract_search_term("query")
    assert result is None


def test_extract_search_term_missing_key():
    with patch.object(_ou, "_chat_raw", return_value='{"other": "val"}'):
        result = _ou.extract_search_term("query")
    assert result is None


def test_extract_search_term_empty_string():
    with patch.object(_ou, "_chat_raw", return_value='{"term": ""}'):
        result = _ou.extract_search_term("query")
    assert result is None


def test_extract_search_term_prompt_includes_known_items_and_examples():
    captured = {}
    def _capture(prompt, max_retries=None, json_mode=False):
        captured["prompt"] = prompt
        captured["json_mode"] = json_mode
        return '{"term": "wallet"}'
    labels = ["wallet", "keys", "receipt"]
    with patch.object(_ou, "_chat_raw", side_effect=_capture):
        result = _ou.extract_search_term("where did I leave my wallet", known_labels=labels)
    assert result == "wallet"
    assert captured.get("json_mode") is True
    prompt = captured.get("prompt", "")
    assert "Known items: wallet, keys, receipt" in prompt
    assert 'Q: "where did I leave my wallet last night?"' in prompt
    assert "Now extract from the user query." in prompt


def test_extract_item_intent_valid_intents():
    valid = ["read_ocr", "export_ocr", "rename", "find", "describe"]
    for intent in valid:
        with patch.object(_ou, "_chat_raw", return_value=json.dumps({"intent": intent})):
            result = _ou.extract_item_intent("some query")
        assert result == intent, f"expected '{intent}', got {result!r}"


def test_extract_item_intent_invalid_returns_none():
    with patch.object(_ou, "_chat_raw", return_value='{"intent": "unknown_action"}'):
        result = _ou.extract_item_intent("query")
    assert result is None


def test_extract_item_intent_none_on_failure():
    with patch.object(_ou, "_chat_raw", return_value=None):
        result = _ou.extract_item_intent("query")
    assert result is None


def test_extract_rename_target_valid():
    with patch.object(_ou, "_chat_raw", return_value='{"name": "my wallet"}'):
        result = _ou.extract_rename_target("call this my wallet")
    assert result == "my wallet"


def test_extract_rename_target_null():
    with patch.object(_ou, "_chat_raw", return_value='{"name": null}'):
        result = _ou.extract_rename_target("rename this")
    assert result is None


def test_extract_rename_target_none_on_failure():
    with patch.object(_ou, "_chat_raw", return_value=None):
        result = _ou.extract_rename_target("rename this")
    assert result is None


def test_build_known_items_context_limits_to_20():
    labels = [f"item{i}" for i in range(30)]
    context = _ou._build_known_items_context(labels)
    assert "Known items:" in context
    assert "item19" in context
    assert "item20" not in context


def test_settings_get_ollama_model_per_mode():
    from visual_memory.config.settings import Settings

    s = Settings(
        ollama_model_balanced="llama3.2:1b",
        ollama_model_accurate="phi3:mini",
    )
    assert s.get_ollama_model("fast") is None
    assert s.get_ollama_model("balanced") == "llama3.2:1b"
    assert s.get_ollama_model("accurate") == "phi3:mini"


def test_circuit_breaker_opens_after_threshold():
    _reset_cb()
    calls = []
    def _failing(*args, **kwargs):
        calls.append(1)
        raise Exception("ollama down")

    # Trigger enough failures to open the breaker.
    # _cb_record_failure is called inside _chat_raw on each exception.
    # We'll simulate by calling _cb_record_failure directly.
    for _ in range(_ou._CB_FAILURE_THRESHOLD):
        _ou._cb_record_failure()

    assert _ou._cb_is_open() is True
    _reset_cb()


def test_circuit_breaker_cooldown_resets():
    _reset_cb()
    for _ in range(_ou._CB_FAILURE_THRESHOLD):
        _ou._cb_record_failure()
    assert _ou._cb_is_open() is True

    # Simulate cooldown elapsed
    with _ou._cb_lock:
        _ou._cb_state["opened_at"] = time.time() - _ou._CB_COOLDOWN_SECONDS - 1

    assert _ou._cb_is_open() is False
    _reset_cb()


def test_circuit_breaker_success_clears():
    _reset_cb()
    for _ in range(_ou._CB_FAILURE_THRESHOLD - 1):
        _ou._cb_record_failure()
    _ou._cb_record_success()
    with _ou._cb_lock:
        assert _ou._cb_state["failures"] == 0
        assert _ou._cb_state["opened_at"] is None
    _reset_cb()


def test_circuit_breaker_open_returns_none():
    _reset_cb()
    for _ in range(_ou._CB_FAILURE_THRESHOLD):
        _ou._cb_record_failure()
    # When breaker is open, _chat_raw should return None
    result = _ou._chat_raw("any prompt", max_retries=1)
    assert result is None
    _reset_cb()


def test_json_mode_flag_forwarded():
    # Verify that json_mode=True reaches _chat_raw (inspected via extract functions)
    calls = []
    original = _ou._chat_raw
    def _capture(prompt, max_retries=None, json_mode=False):
        calls.append(json_mode)
        return '{"term": "wallet"}'
    with patch.object(_ou, "_chat_raw", side_effect=_capture):
        _ou.extract_search_term("query")
    # extract_search_term uses _chat_raw with json_mode=True
    assert calls and calls[0] is True


for name, fn in [
    ("ollama:extract_search_term_valid", test_extract_search_term_valid),
    ("ollama:extract_search_term_none_on_failure", test_extract_search_term_none_on_failure),
    ("ollama:extract_search_term_malformed_json", test_extract_search_term_malformed_json),
    ("ollama:extract_search_term_missing_key", test_extract_search_term_missing_key),
    ("ollama:extract_search_term_empty_string", test_extract_search_term_empty_string),
    ("ollama:extract_search_term_prompt_has_context", test_extract_search_term_prompt_includes_known_items_and_examples),
    ("ollama:intent_valid_intents", test_extract_item_intent_valid_intents),
    ("ollama:intent_invalid_returns_none", test_extract_item_intent_invalid_returns_none),
    ("ollama:intent_none_on_failure", test_extract_item_intent_none_on_failure),
    ("ollama:rename_target_valid", test_extract_rename_target_valid),
    ("ollama:rename_target_null", test_extract_rename_target_null),
    ("ollama:rename_target_none_on_failure", test_extract_rename_target_none_on_failure),
    ("ollama:known_items_context_limit", test_build_known_items_context_limits_to_20),
    ("ollama:settings_model_per_mode", test_settings_get_ollama_model_per_mode),
    ("ollama:cb_opens_after_threshold", test_circuit_breaker_opens_after_threshold),
    ("ollama:cb_cooldown_resets", test_circuit_breaker_cooldown_resets),
    ("ollama:cb_success_clears", test_circuit_breaker_success_clears),
    ("ollama:cb_open_returns_none", test_circuit_breaker_open_returns_none),
    ("ollama:json_mode_forwarded", test_json_mode_flag_forwarded),
]:
    _runner.run(name, fn)

sys.exit(_runner.summary())
