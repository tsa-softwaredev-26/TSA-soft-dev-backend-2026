"""Generate Phase 6 degradation curve artifacts.

Usage:
    PYTHONPATH=src python3 -m visual_memory.benchmarks.degradation_curves

Outputs:
    benchmarks/results/blur_curve.csv
    benchmarks/results/brightness_curve.csv
    benchmarks/results/noise_curve.csv
    benchmarks/results/degradation_zones.csv
    benchmarks/figures/*_degradation_curve.svg
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_BENCHMARKS_DIR = _PROJECT_ROOT / "benchmarks"
_DEFAULT_DEGRADED = _BENCHMARKS_DIR / "dataset_degraded.csv"
_DEFAULT_RESULTS = _BENCHMARKS_DIR / "results.csv"
_DEFAULT_OUT_DIR = _BENCHMARKS_DIR / "results"
_DEFAULT_FIG_DIR = _BENCHMARKS_DIR / "figures"
_KINDS = ("blur", "brightness", "noise")


def _normalize_kind_and_level(row: dict) -> Optional[Tuple[str, float]]:
    raw_kind = (row.get("degradation_type") or "").strip().lower()
    if not raw_kind:
        return None

    raw_level = row.get("degradation_level")
    if raw_level in (None, ""):
        raw_level = row.get("degradation_param")
    if raw_level in (None, ""):
        return None

    try:
        level = float(raw_level)
    except (TypeError, ValueError):
        return None

    # Backward compatibility:
    # - create_degraded historically emitted "compression" + "degradation_param"
    # - degradation_curves historically consumed "brightness" + "degradation_level"
    if raw_kind == "compression":
        return "brightness", level / 100.0
    if raw_kind in _KINDS:
        return raw_kind, level
    return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate degradation curves")
    p.add_argument("--dataset-degraded", type=Path, default=_DEFAULT_DEGRADED)
    p.add_argument("--results-csv", type=Path, default=_DEFAULT_RESULTS)
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUT_DIR)
    p.add_argument("--figures-dir", type=Path, default=_DEFAULT_FIG_DIR)
    p.add_argument("--no-figures", action="store_true")
    return p.parse_args()


def _load_rows(path: Path) -> List[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(line for line in f if not line.lstrip().startswith("#")))


def _avg(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    if not vals:
        return None
    return sum(vals) / len(vals)


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return ""
    return f"{v:.6f}"


def _proxy_quality(kind: str, level: float) -> float:
    if kind == "blur":
        return max(0.0, 1.0 - (level / 10.0))
    if kind == "brightness":
        return max(0.0, min(1.0, level))
    return max(0.0, 1.0 - (level / 40.0))


def _severity_sort_key(kind: str, level: float) -> float:
    if kind == "brightness":
        return -level
    return level


def _zones(curve_rows: List[dict], kind: str) -> Tuple[str, str, str, str]:
    measured = [r for r in curve_rows if r["personalized_accuracy"] is not None]
    if measured:
        safe_levels = [r["degradation_level"] for r in measured if r["personalized_accuracy"] >= 0.75]
        danger_levels = [r["degradation_level"] for r in measured if r["personalized_accuracy"] < 0.50]
        if safe_levels:
            safe = f">= {min(safe_levels):g}" if kind == "brightness" else f"<= {max(safe_levels):g}"
        else:
            safe = "none"
        if danger_levels:
            danger = f"<= {max(danger_levels):g}" if kind == "brightness" else f">= {min(danger_levels):g}"
        else:
            danger = "none"
        return safe, "between safe and danger thresholds", danger, "benchmark_results_csv"

    fallback = {
        "blur": ("<= 4", "6", ">= 8"),
        "brightness": (">= 0.8", "0.6", "<= 0.4"),
        "noise": ("<= 20", "30", ">= 40"),
    }[kind]
    return fallback[0], fallback[1], fallback[2], "proxy_only_no_results_csv"


def _build_curves(degraded_rows: List[dict], result_rows: List[dict]) -> Tuple[Dict[str, List[dict]], List[dict]]:
    by_image = {r.get("image", "").strip(): r for r in result_rows if r.get("image")}
    grouped: Dict[Tuple[str, float], List[dict]] = {}

    for row in degraded_rows:
        norm = _normalize_kind_and_level(row)
        if norm is None:
            continue
        kind, level = norm
        grouped.setdefault((kind, level), []).append(row)

    curves: Dict[str, List[dict]] = {k: [] for k in _KINDS}
    zones_rows: List[dict] = []

    for kind in _KINDS:
        keys = sorted((k for k in grouped if k[0] == kind), key=lambda k: _severity_sort_key(k[0], k[1]))
        for _, level in keys:
            rows = grouped[(kind, level)]
            n_images = len(rows)
            matched = [by_image.get(r["image"]) for r in rows if by_image.get(r["image"]) is not None]
            n_with_results = len(matched)

            baseline_accuracy = None
            personalized_accuracy = None
            detection_rate = None
            mean_similarity_gap = None
            if matched:
                baseline_accuracy = _avg(float(r["baseline_correct"]) for r in matched)
                personalized_accuracy = _avg(float(r["personalized_correct"]) for r in matched)
                detection_rate = _avg(float(r["dino_detected"]) for r in matched if r.get("dino_detected", "") != "")
                mean_similarity_gap = _avg(float(r["similarity_gap"]) for r in matched if r.get("similarity_gap", "") != "")

            curves[kind].append({
                "degradation_type": kind,
                "degradation_level": level,
                "n_images": n_images,
                "n_with_results": n_with_results,
                "baseline_accuracy": baseline_accuracy,
                "personalized_accuracy": personalized_accuracy,
                "detection_rate": detection_rate,
                "mean_similarity_gap": mean_similarity_gap,
                "proxy_quality_score": _proxy_quality(kind, level),
                "data_status": "measured" if n_with_results == n_images and n_images > 0 else (
                    "partial_results" if n_with_results > 0 else "proxy_only"
                ),
            })

        safe, caution, danger, basis = _zones(curves[kind], kind)
        zones_rows.append({
            "degradation_type": kind,
            "safe_zone": safe,
            "caution_zone": caution,
            "danger_zone": danger,
            "basis": basis,
        })
    return curves, zones_rows


def _write_curve(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "degradation_type", "degradation_level", "n_images", "n_with_results",
        "baseline_accuracy", "personalized_accuracy", "detection_rate",
        "mean_similarity_gap", "proxy_quality_score", "data_status",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = dict(r)
            out["degradation_level"] = _fmt(out["degradation_level"])
            out["baseline_accuracy"] = _fmt(out["baseline_accuracy"])
            out["personalized_accuracy"] = _fmt(out["personalized_accuracy"])
            out["detection_rate"] = _fmt(out["detection_rate"])
            out["mean_similarity_gap"] = _fmt(out["mean_similarity_gap"])
            out["proxy_quality_score"] = _fmt(out["proxy_quality_score"])
            w.writerow(out)


def _write_zones(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["degradation_type", "safe_zone", "caution_zone", "danger_zone", "basis"])
        w.writeheader()
        w.writerows(rows)


def _plot(curves: Dict[str, List[dict]], figures_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figures")
        return

    figures_dir.mkdir(parents=True, exist_ok=True)
    for kind, rows in curves.items():
        if not rows:
            continue
        x = [r["degradation_level"] for r in rows]
        has_measured = any(r["personalized_accuracy"] is not None for r in rows)
        if has_measured:
            y = [r["personalized_accuracy"] if r["personalized_accuracy"] is not None else float("nan") for r in rows]
            ylabel = "personalized_accuracy"
            title = f"{kind} degradation curve"
        else:
            y = [r["proxy_quality_score"] for r in rows]
            ylabel = "proxy_quality_score"
            title = f"{kind} degradation proxy curve"
        plt.figure(figsize=(6, 4))
        plt.plot(x, y, marker="o")
        plt.title(title)
        plt.xlabel("degradation_level")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / f"{kind}_degradation_curve.svg")
        plt.close()


def main() -> None:
    args = _parse_args()
    if not args.dataset_degraded.exists():
        raise SystemExit(f"Missing input: {args.dataset_degraded}")
    degraded_rows = _load_rows(args.dataset_degraded)
    result_rows = _load_rows(args.results_csv) if args.results_csv.exists() else []
    curves, zones_rows = _build_curves(degraded_rows, result_rows)

    _write_curve(args.output_dir / "blur_curve.csv", curves["blur"])
    _write_curve(args.output_dir / "brightness_curve.csv", curves["brightness"])
    _write_curve(args.output_dir / "noise_curve.csv", curves["noise"])
    _write_zones(args.output_dir / "degradation_zones.csv", zones_rows)

    if not args.no_figures:
        _plot(curves, args.figures_dir)

    print(f"Wrote: {args.output_dir / 'blur_curve.csv'}")
    print(f"Wrote: {args.output_dir / 'brightness_curve.csv'}")
    print(f"Wrote: {args.output_dir / 'noise_curve.csv'}")
    print(f"Wrote: {args.output_dir / 'degradation_zones.csv'}")
    if not args.results_csv.exists():
        print("Note: benchmarks/results.csv not found; emitted proxy-only curves")


if __name__ == "__main__":
    main()
