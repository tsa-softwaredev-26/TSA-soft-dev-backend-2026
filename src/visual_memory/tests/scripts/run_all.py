"""
Unified test runner for the visual_memory test suite.

Usage:
    python -m visual_memory.tests.scripts.run_all
    python -m visual_memory.tests.scripts.run_all --suite unit
    python -m visual_memory.tests.scripts.run_all --suite api
    python -m visual_memory.tests.scripts.run_all --suite all
    python -m visual_memory.tests.scripts.run_all --tag remember,scan
    python -m visual_memory.tests.scripts.run_all --fail-fast
    TEST_VERBOSITY=3 python -m visual_memory.tests.scripts.run_all
    TEST_BASE_URL=http://server:5000 python -m visual_memory.tests.scripts.run_all --suite api

Suites:
    unit    - no Flask, no models (fast, < 5s)
    api     - Flask test client, stubs (< 30s)
    system  - real models (slow, requires GPU or CPU with patience)
    all     - unit + api (default when no --suite given)
"""
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent


# Test registry: (module_name, suite_tag, feature_tags)
_UNIT_TESTS = [
    ("test_quality_utils",    "unit", ["quality"]),
    ("test_similarity_utils", "unit", ["similarity"]),
    ("test_database_store",   "unit", ["database"]),
    ("test_ollama_utils",     "unit", ["ollama"]),
    ("test_projection_head",  "unit", ["learning"]),
    ("test_learning_pipeline","unit", ["learning", "api"]),
    ("test_scan_batching",    "unit", ["scan", "embedding"]),
]

_API_TESTS = [
    ("test_remember_route",   "api",  ["remember"]),
    ("test_scan_route",       "api",  ["scan"]),
    ("test_transcribe_route", "api",  ["voice", "transcribe"]),
    ("test_feedback_retrain", "api",  ["feedback", "retrain", "learning"]),
    ("test_items_crud",       "api",  ["items"]),
    ("test_sightings_route",  "api",  ["sightings"]),
    ("test_settings_routes",  "api",  ["settings"]),
    ("test_health_debug",     "api",  ["health", "debug"]),
    ("test_ask_mode",         "api",  ["ask", "find"]),
]

_SYSTEM_TESTS = [
    ("run_tests", "system", ["system"]),
]

_ALL_TESTS = _UNIT_TESTS + _API_TESTS


def _match_tags(test_tags: list[str], filter_tags: list[str]) -> bool:
    if not filter_tags:
        return True
    return any(t in test_tags for t in filter_tags)


def _run_module(module_name: str) -> int:
    """Run a test module as a subprocess. Returns exit code."""
    result = subprocess.run(
        [sys.executable, "-m", f"visual_memory.tests.scripts.{module_name}"],
        cwd=_SCRIPTS_DIR.parents[4],
    )
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="visual_memory test runner")
    parser.add_argument("--suite", choices=["unit", "api", "system", "all"], default="all")
    parser.add_argument("--tag", default="", help="Comma-separated feature tags to filter")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    args = parser.parse_args()

    filter_tags = [t.strip() for t in args.tag.split(",") if t.strip()] if args.tag else []

    if args.suite == "unit":
        candidates = _UNIT_TESTS
    elif args.suite == "api":
        candidates = _API_TESTS
    elif args.suite == "system":
        candidates = _SYSTEM_TESTS
    else:
        candidates = _ALL_TESTS

    tests_to_run = [
        (mod, suite, tags) for mod, suite, tags in candidates
        if _match_tags(tags, filter_tags)
    ]

    if not tests_to_run:
        print(f"No tests matched suite={args.suite!r} tag={args.tag!r}")
        sys.exit(0)

    passed = 0
    failed = 0
    t0 = time.monotonic()

    print(f"\nRunning {len(tests_to_run)} test module(s) [{args.suite}]\n")

    for module_name, suite, tags in tests_to_run:
        rc = _run_module(module_name)
        if rc == 0:
            passed += 1
        else:
            failed += 1
            if args.fail_fast:
                print(f"\n[FAIL-FAST] Stopped after failure in {module_name}")
                break

    elapsed = (time.monotonic() - t0) * 1000
    total = passed + failed
    print(f"\nSuite [{args.suite}]: {passed}/{total} modules passed  ({elapsed:.0f}ms)")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
