"""Generate BENCHMARKS.md at project root from benchmarks/results.json.

Usage:
    python -m visual_memory.benchmarks.format_results
    python -m visual_memory.benchmarks.format_results --results benchmarks/results.json
    python -m visual_memory.benchmarks.format_results --output BENCHMARKS.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_BENCHMARKS_DIR = _PROJECT_ROOT / "benchmarks"


# ---- arg parsing ----

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Format benchmark results into BENCHMARKS.md")
    p.add_argument("--results", type=Path, default=_BENCHMARKS_DIR / "results.json")
    p.add_argument("--output", type=Path, default=_PROJECT_ROOT / "BENCHMARKS.md")
    return p.parse_args()


# ---- stats helpers ----

def _pct(n: int, d: int) -> str:
    if d == 0:
        return "n/a"
    return f"{100 * n / d:.1f}%"


def _fmt_sim(v: float) -> str:
    return f"{v:.4f}"


def _condition_of(image: str) -> str:
    # wallet_a_1ft_bright_clean.jpg -> 1ft_bright_clean
    stem = image.rsplit(".", 1)[0]
    parts = stem.split("_")
    return "_".join(parts[-3:])


# ---- per-label aggregation ----

def _per_label(results: List[dict]) -> Dict[str, dict]:
    by_label: Dict[str, dict] = {}
    for r in results:
        lbl = r["label"]
        if lbl not in by_label:
            by_label[lbl] = dict(
                n=0,
                bl_correct=0, pe_correct=0,
                bl_sims=[], pe_sims=[], gaps=[],
                detected=0, det_confs=[],
                depth_abs=[], depth_pct=[],
            )
        s = by_label[lbl]
        s["n"] += 1
        s["bl_correct"] += r["baseline_correct"]
        s["pe_correct"] += r["personalized_correct"]
        s["bl_sims"].append(r["baseline_similarity"])
        s["pe_sims"].append(r["personalized_similarity"])
        s["gaps"].append(r["similarity_gap"])
        s["detected"] += r["dino_detected"]
        if r["dino_detected"]:
            s["det_confs"].append(r["dino_confidence"])
        if r["depth_absolute_error"] != "":
            s["depth_abs"].append(float(r["depth_absolute_error"]))
            s["depth_pct"].append(float(r["depth_percentage_error"]))
    return by_label


# ---- per-condition aggregation ----

def _per_condition(results: List[dict]) -> Dict[str, dict]:
    by_cond: Dict[str, dict] = {}
    for r in results:
        cond = _condition_of(r["image"])
        if cond not in by_cond:
            by_cond[cond] = dict(n=0, bl_correct=0, pe_correct=0, detected=0)
        s = by_cond[cond]
        s["n"] += 1
        s["bl_correct"] += r["baseline_correct"]
        s["pe_correct"] += r["personalized_correct"]
        s["detected"] += r["dino_detected"]
    return by_cond


def _mean(xs: list) -> Optional[float]:
    if not xs:
        return None
    return sum(xs) / len(xs)


# ---- markdown generation ----

def _row(*cells: str) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _sep(*widths: int) -> str:
    return "|" + "|".join("-" * (w + 2) for w in widths) + "|"


def generate(results_path: Path, output_path: Path) -> None:
    with open(results_path) as f:
        data = json.load(f)

    meta = data["metadata"]
    results = data["results"]

    by_label = _per_label(results)
    by_cond = _per_condition(results)

    lines: List[str] = []
    a = lines.append

    # ---- header ----
    a("# Benchmark Results - VisualMemory")
    a("")
    a(f"**Date**: {meta['timestamp'][:10]}")
    a(f"**Model**: DINOv3 ViT-L (1024-dim) + CLIP text encoder (512-dim), combined 1536-dim")
    a(f"**Similarity threshold**: {meta['similarity_threshold']:.2f}")
    a(f"**Dataset**: {meta['n_train']} train / {meta['n_test']} test images (seed {meta['seed']})")
    a(f"**Training**: {meta['epochs']} epochs, lr {meta['lr']}, final triplet loss {meta['final_triplet_loss']:.4f}")
    a("")
    a("---")
    a("")

    # ---- retrieval table ----
    a("## Retrieval")
    a("")
    total_n = sum(s["n"] for s in by_label.values())
    total_bl = sum(s["bl_correct"] for s in by_label.values())
    total_pe = sum(s["pe_correct"] for s in by_label.values())
    all_bl_sims = [v for s in by_label.values() for v in s["bl_sims"]]
    all_pe_sims = [v for s in by_label.values() for v in s["pe_sims"]]
    all_gaps = [v for s in by_label.values() for v in s["gaps"]]

    a(_row("Label", "Baseline", "Personalized", "Delta", "Mean Sim (BL)", "Mean Sim (PL)", "Mean Gap"))
    a(_sep(14, 12, 14, 8, 14, 14, 10))
    for lbl in sorted(by_label):
        s = by_label[lbl]
        n = s["n"]
        delta_pp = 100 * (s["pe_correct"] - s["bl_correct"]) / max(n, 1)
        sign = "+" if delta_pp >= 0 else ""
        gap_mean = _mean(s["gaps"])
        gap_sign = "+" if (gap_mean or 0) >= 0 else ""
        a(_row(
            lbl,
            f"{s['bl_correct']}/{n} ({_pct(s['bl_correct'], n)})",
            f"{s['pe_correct']}/{n} ({_pct(s['pe_correct'], n)})",
            f"{sign}{delta_pp:.1f} pp",
            _fmt_sim(_mean(s["bl_sims"]) or 0.0),
            _fmt_sim(_mean(s["pe_sims"]) or 0.0),
            f"{gap_sign}{(_mean(s['gaps']) or 0.0):.4f}",
        ))

    # totals row
    total_delta = 100 * (total_pe - total_bl) / max(total_n, 1)
    sign = "+" if total_delta >= 0 else ""
    mean_gap = _mean(all_gaps) or 0.0
    gap_sign = "+" if mean_gap >= 0 else ""
    a(_row(
        "**Total**",
        f"**{total_bl}/{total_n} ({_pct(total_bl, total_n)})**",
        f"**{total_pe}/{total_n} ({_pct(total_pe, total_n)})**",
        f"**{sign}{total_delta:.1f} pp**",
        f"**{_fmt_sim(_mean(all_bl_sims) or 0.0)}**",
        f"**{_fmt_sim(_mean(all_pe_sims) or 0.0)}**",
        f"**{gap_sign}{mean_gap:.4f}**",
    ))
    a("")
    a("---")
    a("")

    # ---- detection table ----
    a("## Detection (GroundingDINO)")
    a("")
    total_det = sum(s["detected"] for s in by_label.values())

    a(_row("Label", "Detected", "Rate", "Mean Confidence"))
    a(_sep(14, 12, 8, 18))
    for lbl in sorted(by_label):
        s = by_label[lbl]
        n = s["n"]
        conf_mean = _mean(s["det_confs"])
        conf_str = f"{conf_mean:.4f}" if conf_mean is not None else "n/a"
        a(_row(lbl, f"{s['detected']}/{n}", _pct(s["detected"], n), conf_str))
    a(_row(
        "**Total**",
        f"**{total_det}/{total_n}**",
        f"**{_pct(total_det, total_n)}**",
        "",
    ))
    a("")
    a("---")
    a("")

    # ---- depth table ----
    a("## Depth Estimation (detected images with ground truth distance)")
    a("")
    all_depth_abs = [v for s in by_label.values() for v in s["depth_abs"]]
    if not all_depth_abs:
        a("Depth evaluation was skipped (--no-depth or checkpoint missing).")
    else:
        a(_row("Label", "Evaluated", "Mean Abs Error", "Mean % Error"))
        a(_sep(14, 12, 16, 14))
        total_depth_n = 0
        for lbl in sorted(by_label):
            s = by_label[lbl]
            n_d = len(s["depth_abs"])
            total_depth_n += n_d
            if n_d == 0:
                a(_row(lbl, "0", "n/a", "n/a"))
            else:
                a(_row(
                    lbl,
                    str(n_d),
                    f"{(_mean(s['depth_abs']) or 0.0):.2f} ft",
                    f"{(_mean(s['depth_pct']) or 0.0):.1f}%",
                ))
        overall_abs = _mean(all_depth_abs)
        all_depth_pct = [v for s in by_label.values() for v in s["depth_pct"]]
        overall_pct = _mean(all_depth_pct)
        a(_row(
            "**Total**",
            f"**{total_depth_n}**",
            f"**{(overall_abs or 0.0):.2f} ft**",
            f"**{(overall_pct or 0.0):.1f}%**",
        ))
    a("")
    a("---")
    a("")

    # ---- image quality distribution ----
    a("## Image Quality Distribution")
    a("")
    
    dark_imgs = [r for r in results if r.get("is_dark", False)]
    blurry_imgs = [r for r in results if r.get("is_blurry", False)]
    text_imgs = [r for r in results if r.get("text_likelihood", 0.0) >= meta.get("ocr_text_likelihood_threshold", 0.10)]
    ocr_skip_imgs = [r for r in results if r.get("should_skip_ocr", False)]
    
    n_dark = len(dark_imgs)
    n_blurry = len(blurry_imgs)
    n_text = len(text_imgs)
    n_skip = len(ocr_skip_imgs)
    
    a(f"- **Dark images** (luminance < {meta.get('darkness_threshold', 30.0):.1f}): "
      f"**{n_dark}/{total_n} ({_pct(n_dark, total_n)})**")
    a(f"- **Blurry images** (blur score < {meta.get('blur_threshold', 100.0):.1f}): "
      f"**{n_blurry}/{total_n} ({_pct(n_blurry, total_n)})**")
    a(f"- **Text-bearing images** (likelihood >= {meta.get('ocr_text_likelihood_threshold', 0.10):.2f}): "
      f"**{n_text}/{total_n} ({_pct(n_text, total_n)})**")
    a(f"- **OCR skipped** (text likelihood too low): **{n_skip}/{total_n} ({_pct(n_skip, total_n)})**")
    a("")
    
    # Quality vs performance
    a("### Quality Impact on Performance")
    a("")
    
    def _quality_stats(imgs, name):
        if not imgs:
            return f"- **{name}**: No images in this category"
        n = len(imgs)
        det_success = sum(1 for r in imgs if r.get("dino_detected", 0))
        bl_correct = sum(r.get("baseline_correct", 0) for r in imgs)
        pe_correct = sum(r.get("personalized_correct", 0) for r in imgs)
        return (f"- **{name}** (n={n}): "
                f"Detection {_pct(det_success, n)}, "
                f"Retrieval baseline {_pct(bl_correct, n)}, "
                f"personalized {_pct(pe_correct, n)}")
    
    a(_quality_stats(dark_imgs, "Dark images"))
    a(_quality_stats(blurry_imgs, "Blurry images"))
    a(_quality_stats(text_imgs, "Text-bearing images"))
    
    non_text_imgs = [r for r in results if r.get("text_likelihood", 0.0) < meta.get("ocr_text_likelihood_threshold", 0.10)]
    if text_imgs and non_text_imgs:
        text_bl_acc = sum(r.get("baseline_correct", 0) for r in text_imgs) / len(text_imgs)
        non_text_bl_acc = sum(r.get("baseline_correct", 0) for r in non_text_imgs) / len(non_text_imgs)
        text_pe_acc = sum(r.get("personalized_correct", 0) for r in text_imgs) / len(text_imgs)
        non_text_pe_acc = sum(r.get("personalized_correct", 0) for r in non_text_imgs) / len(non_text_imgs)
        a("")
        a(f"**Text vs Non-text retrieval**: "
          f"Baseline {text_bl_acc*100:.1f}% vs {non_text_bl_acc*100:.1f}%, "
          f"Personalized {text_pe_acc*100:.1f}% vs {non_text_pe_acc*100:.1f}%")
    
    a("")
    a("---")
    a("")

    # ---- per-condition breakdown ----
    a("## By Condition")
    a("")
    a(_row("Condition", "N", "Baseline Acc", "Personalized Acc", "Detection Rate"))
    a(_sep(18, 4, 14, 16, 16))
    for cond in sorted(by_cond):
        s = by_cond[cond]
        n = s["n"]
        a(_row(
            cond,
            str(n),
            f"{s['bl_correct']}/{n} ({_pct(s['bl_correct'], n)})",
            f"{s['pe_correct']}/{n} ({_pct(s['pe_correct'], n)})",
            f"{s['detected']}/{n} ({_pct(s['detected'], n)})",
        ))
    a("")
    a("---")
    a("")
    # ---- latency table ----
    a("## Latency")
    a("")
    lat_phases = [
        ("embed_image",  "lat_embed_img_s"),
        ("ocr",          "lat_ocr_s"),
        ("embed_text",   "lat_embed_txt_s"),
        ("detect",       "lat_detect_s"),
        ("retrieve_bl",  "lat_retrieve_bl_s"),
        ("retrieve_pl",  "lat_retrieve_pe_s"),
        ("depth",        "lat_depth_s"),
    ]
    a(_row("Phase", "Mean", "Min", "Max", "Outliers (>2x mean)"))
    a(_sep(14, 8, 8, 8, 40))
    for phase_name, col in lat_phases:
        vals = [(r[col], r["image"]) for r in results if r.get(col, 0) > 0]
        if not vals:
            continue
        times = [v for v, _ in vals]
        mean_t = sum(times) / len(times)
        threshold_t = mean_t * 2.0
        outlier_strs = [f"{lbl} ({t:.2f}s)" for t, lbl in vals if t > threshold_t]
        outlier_cell = "; ".join(outlier_strs[:2]) if outlier_strs else "-"
        a(_row(
            phase_name,
            f"{mean_t:.3f}s",
            f"{min(times):.3f}s",
            f"{max(times):.3f}s",
            outlier_cell,
        ))
    a("")
    a("---")
    a("")

    # ---- false positive table ----
    negatives = data.get("negatives", [])
    a("## False Positive Rate (Negative Images)")
    a("")
    if not negatives:
        a("No negative dataset evaluated.")
    else:
        n_neg = len(negatives)
        bl_fp = sum(r["baseline_fp"] for r in negatives)
        pe_fp = sum(r["personalized_fp"] for r in negatives)
        delta_fp = pe_fp - bl_fp
        fp_sign = "+" if delta_fp > 0 else ""
        a(f"Baseline FP rate: **{bl_fp}/{n_neg} ({100*bl_fp/n_neg:.1f}%)**  "
          f"Personalized: **{pe_fp}/{n_neg} ({100*pe_fp/n_neg:.1f}%)**  "
          f"Delta: **{fp_sign}{100*delta_fp/n_neg:.1f} pp**")
        a("")
        a(_row("Image", "BL match", "BL sim", "BL FP", "PL match", "PL sim", "PL FP"))
        a(_sep(30, 20, 8, 6, 20, 8, 6))
        for r in negatives:
            a(_row(
                r["image"],
                r["baseline_match"] or "-",
                _fmt_sim(r["baseline_sim"]) if r["baseline_fp"] else "-",
                "FP" if r["baseline_fp"] else "ok",
                r["personalized_match"] or "-",
                _fmt_sim(r["personalized_sim"]) if r["personalized_fp"] else "-",
                "FP" if r["personalized_fp"] else "ok",
            ))
    a("")
    a("---")
    a("")
    a(f"*Generated by `python -m visual_memory.benchmarks.format_results`*")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Written: {output_path}")


def main() -> None:
    args = _parse_args()
    if not args.results.exists():
        print(f"results.json not found: {args.results}")
        print("Run full_benchmark.py first.")
        raise SystemExit(1)
    generate(args.results, args.output)


if __name__ == "__main__":
    main()
