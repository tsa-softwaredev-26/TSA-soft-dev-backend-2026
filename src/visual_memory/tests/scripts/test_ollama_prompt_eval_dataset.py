"""
Ollama extraction reliability evaluator using unified voice_eval_dataset.json.

This evaluates text extraction behavior against expected fields from the same
cases used by Whisper eval.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from visual_memory.tests.scripts.voice_eval_common import load_voice_cases
from visual_memory.utils.ollama_utils import (
    extract_item_intent,
    extract_rename_target,
    extract_search_term,
)


def _norm(v):
    if v is None:
        return None
    return str(v).strip().lower()


def main() -> int:
    parser = argparse.ArgumentParser(description="Ollama prompt extraction dataset evaluator")
    parser.add_argument("--dataset", default="src/visual_memory/tests/input_data/voice_eval_dataset.json")
    parser.add_argument("--core-only", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[4]
    cases = load_voice_cases(args.dataset, target="ollama", core_only=args.core_only)
    if not cases:
        print("ERROR: no ollama cases in dataset")
        return 2

    rows = []
    term_ok = 0
    intent_ok = 0
    rename_ok = 0

    for case in cases:
        q = case["utterance"]
        got_term = extract_search_term(q)
        got_intent = extract_item_intent(q)
        got_rename = extract_rename_target(q)

        exp_term = case.get("expected_search_term")
        exp_intent = case.get("expected_intent")
        exp_rename = case.get("expected_rename_target")

        ok_term = _norm(got_term) == _norm(exp_term)
        ok_intent = _norm(got_intent) == _norm(exp_intent)
        ok_rename = _norm(got_rename) == _norm(exp_rename)

        term_ok += int(ok_term)
        intent_ok += int(ok_intent)
        rename_ok += int(ok_rename)

        rows.append(
            {
                "id": case.get("id"),
                "feature": case.get("feature"),
                "priority": case.get("priority"),
                "utterance": q,
                "expected": {
                    "search_term": exp_term,
                    "intent": exp_intent,
                    "rename_target": exp_rename,
                },
                "got": {
                    "search_term": got_term,
                    "intent": got_intent,
                    "rename_target": got_rename,
                },
                "pass": {
                    "search_term": ok_term,
                    "intent": ok_intent,
                    "rename_target": ok_rename,
                },
            }
        )

    n = len(rows)
    summary = {
        "dataset": args.dataset,
        "core_only": bool(args.core_only),
        "total_cases": n,
        "search_term_exact": term_ok,
        "intent_exact": intent_ok,
        "rename_exact": rename_ok,
        "search_term_rate": round(term_ok / n, 3),
        "intent_rate": round(intent_ok / n, 3),
        "rename_rate": round(rename_ok / n, 3),
    }

    report = {
        "date": time.strftime("%Y-%m-%d"),
        "summary": summary,
        "cases": rows,
    }

    logs_dir = repo_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    out_path = logs_dir / "ollama_prompt_eval_report.json"
    out_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(summary))
    print(f"report={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
