"""Evaluate Ollama extraction quality and latency on a prompt dataset."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path


def _check_ollama() -> bool:
    try:
        import ollama  # type: ignore
        client = ollama.Client(timeout=3.0)
        client.list()
        return True
    except Exception:
        return False


def _latency_stats(latencies: list[float]) -> dict:
    if not latencies:
        return {"mean_ms": 0.0, "p95_ms": 0.0}
    values = sorted(latencies)
    idx = int(round(0.95 * (len(values) - 1)))
    return {
        "mean_ms": round(statistics.mean(values), 2),
        "p95_ms": round(values[idx], 2),
    }


def _json_probe(_ou, query: str) -> bool:
    prompt = (
        "Return strict JSON with one field only. "
        "Output format: {\"query\": \"<echo>\"}. "
        f"User query: {json.dumps(query)}"
    )
    raw = _ou._chat_raw(prompt, max_retries=1, json_mode=True)
    if not raw:
        return False
    try:
        data = json.loads(raw)
        return isinstance(data, dict) and "query" in data
    except Exception:
        return False


def _score_case(_ou, case: dict) -> tuple[str | None, bool]:
    kind = str(case.get("kind", "")).strip()
    query = str(case.get("query", "")).strip()
    known_labels = case.get("known_labels") or None

    if kind == "search_term":
        value = _ou.extract_search_term(query, known_labels=known_labels)
    elif kind == "intent":
        value = _ou.extract_item_intent(query, known_labels=known_labels)
    elif kind == "rename":
        value = _ou.extract_rename_target(query, known_labels=known_labels)
    else:
        return None, False

    expected = str(case.get("expected", "")).strip().lower()
    got = (value or "").strip().lower()
    return value, got == expected


def _evaluate_model(model: str, cases: list[dict]) -> dict:
    import visual_memory.utils.ollama_utils as _ou

    os.environ["OLLAMA_MODEL"] = model

    latencies: list[float] = []
    json_ok = 0
    correct = 0
    rows = []

    for idx, case in enumerate(cases, start=1):
        t0 = time.perf_counter()
        value, is_correct = _score_case(_ou, case)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)

        json_pass = _json_probe(_ou, str(case.get("query", "")))
        if json_pass:
            json_ok += 1
        if is_correct:
            correct += 1

        rows.append({
            "index": idx,
            "kind": case.get("kind"),
            "query": case.get("query"),
            "expected": case.get("expected"),
            "got": value,
            "correct": is_correct,
            "latency_ms": round(latency_ms, 2),
            "json_probe_ok": json_pass,
        })

    stats = _latency_stats(latencies)
    total = max(1, len(cases))
    return {
        "model": model,
        "total": len(cases),
        "accuracy": round(correct / total, 4),
        "json_reliability": round(json_ok / total, 4),
        "latency": stats,
        "rows": rows,
    }


def _recommend(results: list[dict], balanced_target_ms: float) -> dict:
    if not results:
        return {"balanced": None, "accurate": None}

    balanced_pool = [r for r in results if r["latency"]["mean_ms"] <= balanced_target_ms]
    if not balanced_pool:
        balanced_pool = results

    balanced = sorted(
        balanced_pool,
        key=lambda r: (r["accuracy"], r["json_reliability"], -r["latency"]["mean_ms"]),
        reverse=True,
    )[0]

    accurate = sorted(
        results,
        key=lambda r: (r["accuracy"], r["json_reliability"], -r["latency"]["p95_ms"]),
        reverse=True,
    )[0]

    return {
        "balanced": balanced["model"],
        "accurate": accurate["model"],
        "balanced_target_ms": balanced_target_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Ollama model prompt eval")
    parser.add_argument(
        "--dataset",
        default="src/visual_memory/tests/input_data/ollama_eval_20.json",
        help="Path to JSON dataset",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llama3.2:1b", "phi3:mini"],
        help="Models to compare",
    )
    parser.add_argument(
        "--balanced-target-ms",
        type=float,
        default=200.0,
        help="Latency target for balanced recommendation",
    )
    args = parser.parse_args()

    if not _check_ollama():
        print("SKIP: Ollama service unavailable")
        return 0

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: dataset not found: {dataset_path}")
        return 1

    cases = json.loads(dataset_path.read_text())
    if not isinstance(cases, list) or not cases:
        print("ERROR: dataset must be a non-empty JSON list")
        return 1

    print(f"Running Ollama eval on {len(cases)} cases")
    results = []
    for model in args.models:
        print(f"- evaluating {model}")
        results.append(_evaluate_model(model, cases))

    recommendation = _recommend(results, args.balanced_target_ms)
    report = {
        "dataset": str(dataset_path),
        "results": results,
        "recommendation": recommendation,
    }

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    out_path = logs_dir / "ollama_prompt_eval_dataset_report.json"
    out_path.write_text(json.dumps(report, indent=2))

    for result in results:
        print(
            f"{result['model']}: accuracy={result['accuracy']:.1%}, "
            f"json={result['json_reliability']:.1%}, "
            f"mean={result['latency']['mean_ms']}ms, p95={result['latency']['p95_ms']}ms"
        )

    print(
        "Recommendation: "
        f"BALANCED={recommendation['balanced']}, "
        f"ACCURATE={recommendation['accurate']}"
    )
    print(f"Report saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
