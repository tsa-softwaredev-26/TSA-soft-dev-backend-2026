"""
Ollama extraction reliability evaluator using unified voice_eval_dataset.json.

This evaluates text extraction behavior against expected fields from the same
cases used by Whisper eval.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from visual_memory.tests.scripts.voice_eval_common import load_voice_cases
from visual_memory.utils.ollama_utils import (
    classify_item_intent_deterministic,
    extract_item_intent,
    extract_rename_target,
    extract_search_term,
)


def _norm(v):
    if v is None:
        return None
    return str(v).strip().lower()


def _token_f1(expected: str | None, got: str | None) -> float:
    exp = [t for t in (_norm(expected) or "").split() if t]
    pred = [t for t in (_norm(got) or "").split() if t]
    if not exp and not pred:
        return 1.0
    if not exp or not pred:
        return 0.0
    exp_counts = Counter(exp)
    pred_counts = Counter(pred)
    overlap = sum(min(exp_counts[t], pred_counts[t]) for t in exp_counts.keys() & pred_counts.keys())
    if overlap <= 0:
        return 0.0
    precision = overlap / max(len(pred), 1)
    recall = overlap / max(len(exp), 1)
    return (2.0 * precision * recall) / (precision + recall)


def _classify_expected_path(feature: str, query: str, rename_target: str | None) -> str | None:
    q = (query or "").strip()
    if feature == "ask":
        # Matches voice router behavior for non-item queries: "where/find/locate" => find.
        lowered = q.lower()
        if any(token in lowered for token in ("where", "find", "locate", "last seen", "location of", "where did i")):
            return "find"
        return "ask"
    if feature == "item_ask":
        # Item context route resolves to item action classes.
        deterministic = classify_item_intent_deterministic(q)
        if deterministic is not None:
            return deterministic
        return extract_item_intent(q)
    if feature == "teach":
        # Teach/remember cases are rename-target extraction in this dataset.
        return "rename" if _norm(rename_target) else None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Ollama prompt extraction dataset evaluator")
    parser.add_argument("--dataset", default="src/visual_memory/tests/input_data/voice_eval_dataset.json")
    parser.add_argument("--core-only", action="store_true")
    parser.add_argument("--output", default="", help="Optional path to write full JSON report")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[4]
    cases = load_voice_cases(args.dataset, target="ollama", core_only=args.core_only)
    if not cases:
        print("ERROR: no ollama cases in dataset")
        return 2

    rows = []
    term_ok = 0
    term_fuzzy_ok = 0
    intent_ok = 0
    rename_ok = 0
    confusion: Counter[tuple[str, str]] = Counter()
    class_totals: Counter[str] = Counter()
    class_correct: Counter[str] = Counter()

    for case in cases:
        q = case["utterance"]
        got_term = extract_search_term(q)
        got_rename = extract_rename_target(q)
        got_intent = _classify_expected_path(case.get("feature", ""), q, got_rename)

        exp_term = case.get("expected_search_term")
        exp_intent = case.get("expected_intent")
        exp_rename = case.get("expected_rename_target")

        ok_term = _norm(got_term) == _norm(exp_term)
        term_f1 = _token_f1(exp_term, got_term)
        ok_term_fuzzy = term_f1 >= 0.5
        ok_intent = _norm(got_intent) == _norm(exp_intent)
        ok_rename = _norm(got_rename) == _norm(exp_rename)

        term_ok += int(ok_term)
        term_fuzzy_ok += int(ok_term_fuzzy)
        intent_ok += int(ok_intent)
        rename_ok += int(ok_rename)
        exp_key = _norm(exp_intent) or "none"
        got_key = _norm(got_intent) or "none"
        confusion[(exp_key, got_key)] += 1
        class_totals[exp_key] += 1
        class_correct[exp_key] += int(ok_intent)

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
                    "search_term_fuzzy": ok_term_fuzzy,
                    "intent": ok_intent,
                    "rename_target": ok_rename,
                },
                "metrics": {
                    "search_term_token_f1": round(term_f1, 3),
                },
            }
        )

    n = len(rows)
    matrix = {
        exp: {got: count for (exp_i, got), count in confusion.items() if exp_i == exp}
        for exp in sorted(class_totals.keys())
    }
    confusions_only = sorted(
        [
            {"expected": exp, "predicted": got, "count": count}
            for (exp, got), count in confusion.items()
            if exp != got
        ],
        key=lambda x: x["count"],
        reverse=True,
    )[:5]
    weak_classes = []
    for cls in sorted(class_totals.keys()):
        total = class_totals[cls]
        acc = (class_correct[cls] / total) if total else 0.0
        if acc < 0.7:
            weak_classes.append({
                "class": cls,
                "accuracy": round(acc, 3),
                "correct": class_correct[cls],
                "total": total,
            })

    summary = {
        "dataset": args.dataset,
        "core_only": bool(args.core_only),
        "total_cases": n,
        "search_term_exact": term_ok,
        "search_term_fuzzy": term_fuzzy_ok,
        "intent_exact": intent_ok,
        "rename_exact": rename_ok,
        "search_term_rate": round(term_ok / n, 3),
        "search_term_fuzzy_rate": round(term_fuzzy_ok / n, 3),
        "intent_rate": round(intent_ok / n, 3),
        "rename_rate": round(rename_ok / n, 3),
    }

    report = {
        "date": time.strftime("%Y-%m-%d"),
        "summary": summary,
        "intent_confusion_matrix": matrix,
        "top_confusion_pairs": confusions_only,
        "weak_intent_classes": weak_classes,
        "cases": rows,
    }

    print(json.dumps(summary))
    print(json.dumps({
        "intent_confusion_matrix": matrix,
        "top_confusion_pairs": confusions_only,
        "weak_intent_classes": weak_classes,
    }))
    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = repo_root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"report={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
