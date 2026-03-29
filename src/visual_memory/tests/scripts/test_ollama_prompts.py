"""
Semi-automated Ollama prompt quality testing.

Requires a live Ollama service. Skips gracefully if unavailable.
NOT part of the default run_all suite - run explicitly.

Outputs a summary report with pass rates per function.
Results saved to logs/ollama_prompt_report.json for tracking.

Usage:
    python -m visual_memory.tests.scripts.test_ollama_prompts
    python -m visual_memory.tests.scripts.test_ollama_prompts --verbose
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_VERBOSE = "--verbose" in sys.argv

# Test cases: (query, expected_term_or_intent_or_name)
_SEARCH_TERM_CASES = [
    ("where did I put my wallet?",              "wallet"),
    ("have you seen my blue keys?",             "keys"),
    ("I need to find my phone",                 "phone"),
    ("where is the receipt from last week?",    "receipt"),
    ("can you locate my airpods?",              "airpods"),
]

_INTENT_CASES = [
    ("read this to me",                     "read_ocr"),
    ("what does this say",                  "read_ocr"),
    ("export the text from this",           "export_ocr"),
    ("copy the text here",                  "export_ocr"),
    ("rename this to my wallet",            "rename"),
    ("call it house keys",                  "rename"),
    ("where is this normally kept?",        "find"),
    ("where do I usually put this",         "find"),
    ("describe what this looks like",       "describe"),
    ("what is this object?",                "describe"),
]

_RENAME_CASES = [
    ("rename this to my wallet",    "my wallet"),
    ("call it house keys",          "house keys"),
    ("name this blue notebook",     "blue notebook"),
]


def _check_ollama() -> bool:
    try:
        import ollama  # type: ignore
        client = ollama.Client(timeout=3.0)
        client.list()
        return True
    except Exception:
        return False


def _run_cases(fn, cases: list, fn_name: str) -> dict:
    results = []
    for query, expected in cases:
        try:
            got = fn(query)
        except Exception as exc:
            got = None
        passed = got == expected or (got is not None and expected in str(got).lower())
        results.append({
            "query": query,
            "expected": expected,
            "got": got,
            "passed": passed,
        })
        if _VERBOSE:
            status = "[PASS]" if passed else "[FAIL]"
            print(f"  {status}  {fn_name}({query!r}) -> {got!r}  (expected: {expected!r})")
    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    return {
        "function": fn_name,
        "passed": passed_count,
        "total": total,
        "pass_rate": round(passed_count / total, 3) if total > 0 else 0.0,
        "cases": results,
    }


def main():
    if not _check_ollama():
        print("SKIP: Ollama service unavailable")
        sys.exit(0)

    from visual_memory.utils.ollama_utils import (
        extract_search_term,
        extract_item_intent,
        extract_rename_target,
    )

    print("\nOllama Prompt Quality Report\n")

    sections = [
        _run_cases(extract_search_term, _SEARCH_TERM_CASES, "extract_search_term"),
        _run_cases(extract_item_intent, _INTENT_CASES,      "extract_item_intent"),
        _run_cases(extract_rename_target, _RENAME_CASES,    "extract_rename_target"),
    ]

    for sec in sections:
        print(f"  {sec['function']}: {sec['passed']}/{sec['total']}  ({sec['pass_rate']*100:.0f}%)")

    overall_passed = sum(s["passed"] for s in sections)
    overall_total  = sum(s["total"]  for s in sections)
    print(f"\nOverall: {overall_passed}/{overall_total}")

    report = {
        "date": time.strftime("%Y-%m-%d"),
        "sections": sections,
        "overall_passed": overall_passed,
        "overall_total": overall_total,
    }

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    report_path = logs_dir / "ollama_prompt_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
