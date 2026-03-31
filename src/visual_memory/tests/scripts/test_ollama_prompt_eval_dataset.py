"""
Ollama extraction reliability evaluator using unified voice_eval_dataset.json.

This evaluates text extraction behavior against expected fields from the same
cases used by Whisper eval.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path

from visual_memory.tests.scripts.voice_eval_common import load_voice_cases, resolve_audio_path
from visual_memory.utils.ollama_utils import (
    classify_item_intent_deterministic,
    extract_item_intent,
    extract_rename_target,
    extract_search_term,
)


EXPECTED_DATASET_CASES = 120
SEARCH_F1_THRESHOLD = 0.5
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _norm(v):
    if v is None:
        return None
    return str(v).strip().lower()


def _norm_text(v: str | None) -> str:
    if v is None:
        return ""
    lowered = str(v).strip().lower()
    stripped = NON_ALNUM_RE.sub(" ", lowered)
    return " ".join(stripped.split())


def _tokenize(v: str | None) -> list[str]:
    return [token for token in _norm_text(v).split() if token]


def _token_f1(expected: str | None, got: str | None) -> float:
    exp = _tokenize(expected)
    pred = _tokenize(got)
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


def _subset_match(expected: str | None, got: str | None) -> bool:
    exp_tokens = set(_tokenize(expected))
    got_tokens = set(_tokenize(got))
    if not exp_tokens and not got_tokens:
        return True
    if not exp_tokens or not got_tokens:
        return False
    return exp_tokens.issubset(got_tokens) or got_tokens.issubset(exp_tokens)


def _to_float(v: str | None) -> float | None:
    raw = (v or "").strip()
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _metric_delta(current: float, baseline: float | None) -> float | None:
    if baseline is None:
        return None
    return round(current - baseline, 3)


def _delta_label(delta: float | None) -> str:
    if delta is None:
        return "baseline_missing"
    if delta > 0:
        return "improved"
    if delta < 0:
        return "regressed"
    return "no_change"


def _load_baseline_metrics(repo_root: Path) -> dict:
    baseline_csv = repo_root / "benchmarks" / "baselines" / "audio_intent" / "baseline_metrics.csv"
    out = {
        "source": str(baseline_csv),
        "found": False,
        "rates": {
            "intent_rate": None,
            "intent_first_rate": None,
            "rename_rate": None,
            "search_term_rate": None,
            "search_term_fuzzy_rate": None,
        },
    }
    if not baseline_csv.exists():
        return out

    with baseline_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (row.get("benchmark") or "").strip() != "ollama_prompt_eval":
                continue
            out["found"] = True
            rates = out["rates"]
            rates["intent_rate"] = _to_float(row.get("intent_rate"))
            rates["intent_first_rate"] = _to_float(row.get("intent_first_rate")) or rates["intent_rate"]
            rates["rename_rate"] = _to_float(row.get("rename_rate"))
            rates["search_term_rate"] = _to_float(row.get("search_term_rate"))
            rates["search_term_fuzzy_rate"] = _to_float(row.get("search_term_fuzzy_rate"))
            total = _to_float(row.get("total"))
            fuzzy_count = _to_float(row.get("search_term_fuzzy"))
            if rates["search_term_fuzzy_rate"] is None and fuzzy_count is not None and total and total > 0:
                rates["search_term_fuzzy_rate"] = round(fuzzy_count / total, 3)
            break
    return out


def _run_audio_preflight(dataset_path: Path, repo_root: Path) -> tuple[bool, list[str], int]:
    payload = json.loads(dataset_path.read_text())
    all_cases = payload.get("cases", [])
    if len(all_cases) != EXPECTED_DATASET_CASES:
        print(
            "ERROR: dataset preflight expected "
            f"{EXPECTED_DATASET_CASES} cases, found {len(all_cases)}"
        )
        return False, [], len(all_cases)

    missing_ids: list[str] = []
    for index, case in enumerate(all_cases):
        case_id = str(case.get("id") or f"index_{index}").strip()
        if resolve_audio_path(case, repo_root) is None:
            missing_ids.append(case_id)

    if missing_ids:
        print(f"ERROR: dataset preflight missing audio for {len(missing_ids)} cases")
        print(json.dumps({"missing_audio_ids": missing_ids[:20], "missing_audio_count": len(missing_ids)}))
        return False, missing_ids, len(all_cases)

    return True, [], len(all_cases)


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
    parser.add_argument(
        "--cases-csv",
        default="logs/ollama_prompt_eval_cases.csv",
        help="Path to write per-case CSV metrics",
    )
    parser.add_argument(
        "--output",
        default="logs/ollama_prompt_eval_report.json",
        help="Path to write full JSON report (default: logs/ollama_prompt_eval_report.json)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[4]
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = repo_root / dataset_path

    preflight_ok, missing_audio_ids, total_dataset_cases = _run_audio_preflight(dataset_path, repo_root)
    if not preflight_ok:
        return 2

    cases = load_voice_cases(str(dataset_path), target="ollama", core_only=args.core_only)
    if not cases:
        print("ERROR: no ollama cases in dataset")
        return 2

    baseline = _load_baseline_metrics(repo_root)
    baseline_rates = baseline["rates"]

    rows = []
    term_ok = 0
    term_subset_ok = 0
    term_fuzzy_ok = 0
    intent_ok = 0
    intent_first_ok = 0
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

        ok_term = _norm_text(got_term) == _norm_text(exp_term)
        ok_term_subset = _subset_match(exp_term, got_term)
        term_f1 = _token_f1(exp_term, got_term)
        ok_term_fuzzy = ok_term or ok_term_subset or term_f1 >= SEARCH_F1_THRESHOLD
        ok_intent = _norm(got_intent) == _norm(exp_intent)
        ok_intent_first = ok_intent
        ok_rename = _norm(got_rename) == _norm(exp_rename)

        term_ok += int(ok_term)
        term_subset_ok += int(ok_term_subset)
        term_fuzzy_ok += int(ok_term_fuzzy)
        intent_ok += int(ok_intent)
        intent_first_ok += int(ok_intent_first)
        rename_ok += int(ok_rename)
        exp_key = _norm(exp_intent) or "none"
        got_key = _norm(got_intent) or "none"
        confusion[(exp_key, got_key)] += 1
        class_totals[exp_key] += 1
        class_correct[exp_key] += int(ok_intent)

        case_delta = {
            "intent_rate_delta": _metric_delta(float(ok_intent), baseline_rates["intent_rate"]),
            "intent_first_rate_delta": _metric_delta(float(ok_intent_first), baseline_rates["intent_first_rate"]),
            "rename_rate_delta": _metric_delta(float(ok_rename), baseline_rates["rename_rate"]),
            "search_term_rate_delta": _metric_delta(float(ok_term), baseline_rates["search_term_rate"]),
            "search_term_fuzzy_rate_delta": _metric_delta(float(ok_term_fuzzy), baseline_rates["search_term_fuzzy_rate"]),
        }

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
                    "search_term_subset": ok_term_subset,
                    "search_term_fuzzy": ok_term_fuzzy,
                    "intent": ok_intent,
                    "intent_first": ok_intent_first,
                    "rename_target": ok_rename,
                },
                "metrics": {
                    "search_term_token_f1": round(term_f1, 3),
                },
                "delta_vs_baseline": case_delta,
            }
        )

    n = len(rows)
    matrix: dict[str, dict[str, int]] = {
        exp: {got: count for (exp_i, got), count in confusion.items() if exp_i == exp}
        for exp in sorted(class_totals.keys())
    }
    per_intent_accuracy = {
        cls: {
            "correct": class_correct[cls],
            "total": class_totals[cls],
            "accuracy": round((class_correct[cls] / class_totals[cls]) if class_totals[cls] else 0.0, 3),
        }
        for cls in sorted(class_totals.keys())
    }
    confusions_only = sorted(
        [
            {"expected": exp, "predicted": got, "count": count}
            for (exp, got), count in confusion.items()
            if exp != got
        ],
        key=lambda x: x["count"],
        reverse=True,
    )[:3]
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
        "dataset_case_count_preflight": total_dataset_cases,
        "preflight_missing_audio_count": len(missing_audio_ids),
        "total_cases": n,
        "search_term_exact": term_ok,
        "search_term_subset": term_subset_ok,
        "search_term_fuzzy": term_fuzzy_ok,
        "intent_exact": intent_ok,
        "intent_first_exact": intent_first_ok,
        "rename_exact": rename_ok,
        "search_term_rate": round(term_ok / n, 3),
        "search_term_subset_rate": round(term_subset_ok / n, 3),
        "search_term_fuzzy_rate": round(term_fuzzy_ok / n, 3),
        "intent_rate": round(intent_ok / n, 3),
        "intent_first_rate": round(intent_first_ok / n, 3),
        "rename_rate": round(rename_ok / n, 3),
    }
    delta_vs_baseline = {
        "baseline_metrics_path": baseline["source"],
        "baseline_found": baseline["found"],
        "summary": {
            "intent_rate": {
                "baseline": baseline_rates["intent_rate"],
                "current": summary["intent_rate"],
                "delta": _metric_delta(summary["intent_rate"], baseline_rates["intent_rate"]),
            },
            "intent_first_rate": {
                "baseline": baseline_rates["intent_first_rate"],
                "current": summary["intent_first_rate"],
                "delta": _metric_delta(summary["intent_first_rate"], baseline_rates["intent_first_rate"]),
            },
            "rename_rate": {
                "baseline": baseline_rates["rename_rate"],
                "current": summary["rename_rate"],
                "delta": _metric_delta(summary["rename_rate"], baseline_rates["rename_rate"]),
            },
            "search_term_rate": {
                "baseline": baseline_rates["search_term_rate"],
                "current": summary["search_term_rate"],
                "delta": _metric_delta(summary["search_term_rate"], baseline_rates["search_term_rate"]),
            },
            "search_term_fuzzy_rate": {
                "baseline": baseline_rates["search_term_fuzzy_rate"],
                "current": summary["search_term_fuzzy_rate"],
                "delta": _metric_delta(summary["search_term_fuzzy_rate"], baseline_rates["search_term_fuzzy_rate"]),
            },
        },
        "per_case": [
            {
                "id": row["id"],
                "intent_rate_delta": row["delta_vs_baseline"]["intent_rate_delta"],
                "intent_first_rate_delta": row["delta_vs_baseline"]["intent_first_rate_delta"],
                "rename_rate_delta": row["delta_vs_baseline"]["rename_rate_delta"],
                "search_term_rate_delta": row["delta_vs_baseline"]["search_term_rate_delta"],
                "search_term_fuzzy_rate_delta": row["delta_vs_baseline"]["search_term_fuzzy_rate_delta"],
            }
            for row in rows
        ],
    }

    report = {
        "date": time.strftime("%Y-%m-%d"),
        "summary": summary,
        "confusion_matrix": matrix,
        "per_intent_accuracy": per_intent_accuracy,
        "delta_vs_baseline": delta_vs_baseline,
        "top_confusion_pairs": confusions_only,
        "weak_intent_classes": weak_classes,
        "cases": rows,
    }

    print(json.dumps(summary))
    print(json.dumps({
        "confusion_matrix": matrix,
        "per_intent_accuracy": per_intent_accuracy,
        "top_confusion_pairs": confusions_only,
        "weak_intent_classes": weak_classes,
        "delta_vs_baseline_summary": delta_vs_baseline["summary"],
    }))
    cases_csv_path = Path(args.cases_csv)
    if not cases_csv_path.is_absolute():
        cases_csv_path = repo_root / cases_csv_path
    cases_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with cases_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "feature",
                "priority",
                "intent_expected",
                "intent_got",
                "intent_first_pass",
                "search_term_expected",
                "search_term_got",
                "search_term_exact_pass",
                "search_term_subset_pass",
                "search_term_fuzzy_pass",
                "search_term_token_f1",
                "rename_expected",
                "rename_got",
                "rename_pass",
                "intent_rate_delta",
                "intent_rate_delta_label",
                "intent_first_rate_delta",
                "intent_first_rate_delta_label",
                "search_term_rate_delta",
                "search_term_rate_delta_label",
                "search_term_fuzzy_rate_delta",
                "search_term_fuzzy_rate_delta_label",
                "rename_rate_delta",
                "rename_rate_delta_label",
            ],
        )
        writer.writeheader()
        for row in rows:
            row_delta = row["delta_vs_baseline"]
            writer.writerow(
                {
                    "id": row["id"],
                    "feature": row["feature"],
                    "priority": row["priority"],
                    "intent_expected": row["expected"]["intent"],
                    "intent_got": row["got"]["intent"],
                    "intent_first_pass": int(row["pass"]["intent_first"]),
                    "search_term_expected": row["expected"]["search_term"],
                    "search_term_got": row["got"]["search_term"],
                    "search_term_exact_pass": int(row["pass"]["search_term"]),
                    "search_term_subset_pass": int(row["pass"]["search_term_subset"]),
                    "search_term_fuzzy_pass": int(row["pass"]["search_term_fuzzy"]),
                    "search_term_token_f1": row["metrics"]["search_term_token_f1"],
                    "rename_expected": row["expected"]["rename_target"],
                    "rename_got": row["got"]["rename_target"],
                    "rename_pass": int(row["pass"]["rename_target"]),
                    "intent_rate_delta": row_delta["intent_rate_delta"],
                    "intent_rate_delta_label": _delta_label(row_delta["intent_rate_delta"]),
                    "intent_first_rate_delta": row_delta["intent_first_rate_delta"],
                    "intent_first_rate_delta_label": _delta_label(row_delta["intent_first_rate_delta"]),
                    "search_term_rate_delta": row_delta["search_term_rate_delta"],
                    "search_term_rate_delta_label": _delta_label(row_delta["search_term_rate_delta"]),
                    "search_term_fuzzy_rate_delta": row_delta["search_term_fuzzy_rate_delta"],
                    "search_term_fuzzy_rate_delta_label": _delta_label(row_delta["search_term_fuzzy_rate_delta"]),
                    "rename_rate_delta": row_delta["rename_rate_delta"],
                    "rename_rate_delta_label": _delta_label(row_delta["rename_rate_delta"]),
                }
            )
    print(f"cases_csv={cases_csv_path}")
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"report={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
