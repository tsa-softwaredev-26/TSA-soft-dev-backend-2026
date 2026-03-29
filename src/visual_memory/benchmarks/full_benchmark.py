"""Full system benchmark: retrieval, detection, depth, latency, and false-positive rate.

Usage:
    python -m visual_memory.benchmarks.full_benchmark \
        --dataset benchmarks/dataset.csv \
        --images benchmarks/images \
        --seed 42

    # Fast smoke test (skip depth + OCR, ~5-10 min):
    python -m visual_memory.benchmarks.full_benchmark \
        --dataset benchmarks/dataset.csv \
        --images benchmarks/images \
        --seed 42 --no-depth --no-ocr --epochs 5
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

from visual_memory.config import Settings
from visual_memory.engine.embedding import make_combined_embedding
from visual_memory.engine.model_registry import registry
from visual_memory.engine.text_recognition import TextRecognizer
from visual_memory.learning import ProjectionHead, ProjectionTrainer
from visual_memory.utils.image_utils import load_image
from visual_memory.utils.quality_utils import mean_luminance, estimate_text_likelihood
from visual_memory.utils.similarity_utils import find_match

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_BENCHMARKS_DIR = _PROJECT_ROOT / "benchmarks"
_CHECKPOINT = _PROJECT_ROOT / "checkpoints" / "depth_pro.pt"
_DEFAULT_NEG_IMAGES = _PROJECT_ROOT / "src" / "visual_memory" / "tests" / "input_images"

BLUR_THRESHOLD = 100.0


def _blur_score(image: Image.Image) -> float:
    """
    Laplacian variance of the image. Higher = sharper.
    Uses 4-neighbor discrete Laplacian via numpy; no extra dependencies.
    """
    gray = np.array(image.convert("L"), dtype=np.float32)
    lap = (
        np.roll(gray, 1, 0) + np.roll(gray, -1, 0) +
        np.roll(gray, 1, 1) + np.roll(gray, -1, 1) - 4.0 * gray
    )
    return float(lap.var())


# arg parsing

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full system benchmark")
    p.add_argument("--dataset", type=Path, default=_BENCHMARKS_DIR / "dataset.csv")
    p.add_argument("--images", type=Path, default=_BENCHMARKS_DIR / "images")
    p.add_argument("--negative-dataset", type=Path,
                   default=_BENCHMARKS_DIR / "negative_dataset.csv")
    p.add_argument("--images-neg", type=Path, default=_DEFAULT_NEG_IMAGES,
                   help="Directory containing negative test images")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--focal-length", type=float, default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--no-depth", action="store_true")
    p.add_argument("--no-ocr", action="store_true")
    return p.parse_args()


# phase 1: load & embed

def _load_dataset(csv_path: Path) -> List[dict]:
    rows = []
    with open(csv_path, newline="") as f:
        filtered = (line for line in f if not line.lstrip().startswith("#"))
        for row in csv.DictReader(filtered):
            rows.append({
                "image": row["image"],
                "label": row["label"],
                "distance_ft": float(row["ground_truth_distance_ft"]),
                "dino_prompt": row["dino_prompt"],
            })
    return rows


def _load_negative_dataset(csv_path: Path) -> List[dict]:
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, newline="") as f:
        filtered = (line for line in f if not line.lstrip().startswith("#"))
        for row in csv.DictReader(filtered):
            rows.append({
                "image": row["image"],
                "label": "negative",
                "distance_ft": 0.0,
                "dino_prompt": "",
            })
    return rows


def _embed_rows(
    rows: List[dict],
    images_dir: Path,
    no_ocr: bool,
    settings: Settings,
) -> Dict[str, dict]:
    embedded: Dict[str, dict] = {}
    ocr_client = TextRecognizer() if not no_ocr else None
    n = len(rows)
    for i, row in enumerate(rows):
        fname = row["image"]
        img_path = images_dir / fname
        if not img_path.exists():
            print(f"  [warn] not found, skipping: {img_path}", file=sys.stderr)
            continue
        print(f"  [{i+1}/{n}] {fname}")
        img = load_image(str(img_path))

        lum = mean_luminance(img)
        blur = _blur_score(img)
        text_likelihood = estimate_text_likelihood(img)

        t0 = time.perf_counter()
        img_emb = registry.img_embedder.embed(img)
        lat_img = time.perf_counter() - t0

        text_emb = None
        lat_ocr = lat_txt = 0.0
        if ocr_client is not None:
            t0 = time.perf_counter()
            ocr = ocr_client.recognize(img)
            lat_ocr = time.perf_counter() - t0
            text = ocr.get("text", "")
            if text and text.strip():
                t0 = time.perf_counter()
                text_emb = registry.text_embedder.embed_text(text)
                lat_txt = time.perf_counter() - t0

        emb = make_combined_embedding(img_emb, text_emb)
        embedded[fname] = {
            "label": row["label"],
            "distance_ft": row["distance_ft"],
            "dino_prompt": row["dino_prompt"],
            "embedding": emb,
            "image": img,
            "lat_embed_img": lat_img,
            "lat_ocr": lat_ocr,
            "lat_embed_txt": lat_txt,
            "darkness_level": round(lum, 2),
            "is_dark": lum < settings.darkness_threshold,
            "blur_score": round(blur, 2),
            "is_blurry": blur < BLUR_THRESHOLD,
            "text_likelihood": round(text_likelihood, 3),
            "should_skip_ocr": text_likelihood < settings.ocr_text_likelihood_threshold,
        }
    return embedded


# phase 2: train/test split

def _split(
    embedded: Dict[str, dict],
    seed: int,
) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    by_label: Dict[str, List[str]] = {}
    for fname, data in embedded.items():
        by_label.setdefault(data["label"], []).append(fname)

    train_set: List[str] = []
    test_set: List[str] = []
    for label in sorted(by_label):
        fnames = list(by_label[label])
        rng.shuffle(fnames)
        if len(fnames) == 1:
            train_set.extend(fnames)
        elif len(fnames) == 2:
            train_set.append(fnames[0])
            test_set.append(fnames[1])
        else:
            n_train = max(1, round(len(fnames) * 0.6))
            train_set.extend(fnames[:n_train])
            test_set.extend(fnames[n_train:])
    return train_set, test_set


# phase 3: build reference database

def _build_database(
    train_set: List[str],
    embedded: Dict[str, dict],
) -> List[Tuple[str, torch.Tensor]]:
    return [(embedded[f]["label"], embedded[f]["embedding"]) for f in train_set]


# phase 4: generate triplets and train projection head

def _build_triplets(
    train_set: List[str],
    embedded: Dict[str, dict],
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    triplets = []
    for fname in train_set:
        anchor_emb = embedded[fname]["embedding"]
        label = embedded[fname]["label"]
        same = [f for f in train_set if f != fname and embedded[f]["label"] == label]
        diff = [f for f in train_set if embedded[f]["label"] != label]
        if not same or not diff:
            continue
        triplets.append((anchor_emb, embedded[same[0]]["embedding"], embedded[diff[0]]["embedding"]))
    return triplets


def _train_head(
    train_set: List[str],
    embedded: Dict[str, dict],
    epochs: int,
    lr: float,
    save_path: Path,
) -> Tuple[ProjectionHead, float]:
    head = ProjectionHead()
    triplets = _build_triplets(train_set, embedded)
    if not triplets:
        print("  [warn] no triplets; need >= 2 labels with >= 2 train images each")
        head.eval()
        return head, 0.0
    print(f"  {len(triplets)} triplets, {epochs} epochs")
    trainer = ProjectionTrainer(head, lr=lr)
    final_loss = trainer.train(triplets, epochs=epochs)
    head.eval()
    head.save(save_path)
    return head, final_loss


# phase 5: retrieval evaluation

def _eval_retrieval(
    test_set: List[str],
    embedded: Dict[str, dict],
    database: List[Tuple[str, torch.Tensor]],
    head: ProjectionHead,
    threshold: float,
) -> List[dict]:
    head.eval()
    projected_db = [(lbl, head.project(e)) for lbl, e in database]
    results = []
    for fname in test_set:
        data = embedded[fname]
        test_emb = data["embedding"]
        true_label = data["label"]

        t0 = time.perf_counter()
        bl_label, bl_sim = find_match(test_emb, database, threshold)
        lat_bl = time.perf_counter() - t0

        t0 = time.perf_counter()
        pe_label, pe_sim = find_match(head.project(test_emb), projected_db, threshold)
        lat_pe = time.perf_counter() - t0

        results.append({
            "image": fname,
            "label": true_label,
            "distance_ft": data["distance_ft"],
            "dino_prompt": data["dino_prompt"],
            "baseline_similarity": bl_sim,
            "personalized_similarity": pe_sim,
            "similarity_gap": pe_sim - bl_sim,
            "baseline_correct": int(bl_label == true_label),
            "personalized_correct": int(pe_label == true_label),
            "lat_embed_img": data["lat_embed_img"],
            "lat_ocr": data["lat_ocr"],
            "lat_embed_txt": data["lat_embed_txt"],
            "lat_retrieve_bl": lat_bl,
            "lat_retrieve_pe": lat_pe,
            "darkness_level": data["darkness_level"],
            "is_dark": data["is_dark"],
            "blur_score": data["blur_score"],
            "is_blurry": data["is_blurry"],
            "text_likelihood": data["text_likelihood"],
            "should_skip_ocr": data["should_skip_ocr"],
        })
    return results


# phase 6: grounding dino evaluation

def _eval_detection(
    test_set: List[str],
    embedded: Dict[str, dict],
) -> Dict[str, dict]:
    detector = registry.gdino_detector
    results: Dict[str, dict] = {}
    for fname in test_set:
        data = embedded[fname]
        t0 = time.perf_counter()
        det = detector.detect(data["image"], data["dino_prompt"])
        lat_detect = time.perf_counter() - t0
        if det:
            results[fname] = {"detected": 1, "confidence": det["score"],
                              "box": det["box"], "lat_detect": lat_detect}
        else:
            results[fname] = {"detected": 0, "confidence": 0.0,
                              "box": None, "lat_detect": lat_detect}
    return results


# phase 7: depth evaluation

def _load_depth_estimator():
    if not _CHECKPOINT.exists():
        return None
    return registry.depth_estimator


def _eval_depth(
    test_set: List[str],
    embedded: Dict[str, dict],
    dino_results: Dict[str, dict],
    focal_length: Optional[float],
    no_depth: bool,
) -> Dict[str, dict]:
    blank = {"predicted_ft": None, "abs_error": None, "pct_error": None, "lat_depth": 0.0}
    results = {fname: dict(blank) for fname in test_set}

    if no_depth:
        return results

    estimator = _load_depth_estimator()
    if estimator is None:
        print("  [info] depth checkpoint not found; skipping")
        return results

    for fname in test_set:
        data = embedded[fname]
        det = dino_results[fname]
        gt_dist = data["distance_ft"]
        if not det["detected"] or gt_dist <= 0:
            continue
        t0 = time.perf_counter()
        depth_map = estimator.estimate(data["image"], focal_length_px=focal_length)
        pred_ft = estimator.get_depth_at_bbox(depth_map, det["box"])
        lat_depth = time.perf_counter() - t0
        abs_err = abs(pred_ft - gt_dist)
        results[fname] = {
            "predicted_ft": pred_ft,
            "abs_error": abs_err,
            "pct_error": abs_err / gt_dist * 100.0,
            "lat_depth": lat_depth,
        }
    return results


# negative FP evaluation

def _eval_negatives(
    neg_embedded: Dict[str, dict],
    database: List[Tuple[str, torch.Tensor]],
    head: ProjectionHead,
    threshold: float,
) -> List[dict]:
    head.eval()
    projected_db = [(lbl, head.project(e)) for lbl, e in database]
    results = []
    for fname, data in neg_embedded.items():
        emb = data["embedding"]
        bl_lbl, bl_sim = find_match(emb, database, threshold)
        pe_lbl, pe_sim = find_match(head.project(emb), projected_db, threshold)
        results.append({
            "image": fname,
            "baseline_fp": int(bl_lbl is not None),
            "baseline_match": bl_lbl or "",
            "baseline_sim": round(bl_sim, 6),
            "personalized_fp": int(pe_lbl is not None),
            "personalized_match": pe_lbl or "",
            "personalized_sim": round(pe_sim, 6),
        })
    return results


# output

_CSV_FIELDS = [
    "image", "label", "ground_truth_distance_ft",
    "baseline_similarity", "personalized_similarity", "similarity_gap",
    "baseline_correct", "personalized_correct",
    "dino_detected", "dino_confidence",
    "predicted_distance_ft", "depth_absolute_error", "depth_percentage_error",
    "lat_embed_img_s", "lat_ocr_s", "lat_embed_txt_s",
    "lat_retrieve_bl_s", "lat_retrieve_pe_s", "lat_detect_s", "lat_depth_s",
    "darkness_level", "is_dark", "blur_score", "is_blurry",
    "text_likelihood", "should_skip_ocr",
]


def _merge_rows(
    retrieval: List[dict],
    dino: Dict[str, dict],
    depth: Dict[str, dict],
) -> List[dict]:
    rows = []
    for r in retrieval:
        fname = r["image"]
        d = dino.get(fname, {"detected": 0, "confidence": 0.0, "lat_detect": 0.0})
        dep = depth.get(fname, {"predicted_ft": None, "abs_error": None,
                                 "pct_error": None, "lat_depth": 0.0})
        rows.append({
            "image": fname,
            "label": r["label"],
            "ground_truth_distance_ft": r["distance_ft"],
            "baseline_similarity": round(r["baseline_similarity"], 6),
            "personalized_similarity": round(r["personalized_similarity"], 6),
            "similarity_gap": round(r["similarity_gap"], 6),
            "baseline_correct": r["baseline_correct"],
            "personalized_correct": r["personalized_correct"],
            "dino_detected": d["detected"],
            "dino_confidence": round(d["confidence"], 6),
            "predicted_distance_ft": (
                round(dep["predicted_ft"], 3) if dep["predicted_ft"] is not None else ""),
            "depth_absolute_error": (
                round(dep["abs_error"], 3) if dep["abs_error"] is not None else ""),
            "depth_percentage_error": (
                round(dep["pct_error"], 2) if dep["pct_error"] is not None else ""),
            "lat_embed_img_s": round(r["lat_embed_img"], 4),
            "lat_ocr_s": round(r["lat_ocr"], 4),
            "lat_embed_txt_s": round(r["lat_embed_txt"], 4),
            "lat_retrieve_bl_s": round(r["lat_retrieve_bl"], 6),
            "lat_retrieve_pe_s": round(r["lat_retrieve_pe"], 6),
            "lat_detect_s": round(d["lat_detect"], 4),
            "lat_depth_s": round(dep["lat_depth"], 4),
            "darkness_level": r["darkness_level"],
            "is_dark": r["is_dark"],
            "blur_score": r["blur_score"],
            "is_blurry": r["is_blurry"],
            "text_likelihood": r["text_likelihood"],
            "should_skip_ocr": r["should_skip_ocr"],
        })
    return rows


def _latency_stats(values: List[float], label_col: Optional[List[str]] = None):
    """Return (mean, min, max, outliers) where outliers are >2x mean."""
    if not values:
        return 0.0, 0.0, 0.0, []
    mean = sum(values) / len(values)
    threshold = mean * 2.0
    outliers = []
    if label_col:
        for v, lbl in zip(values, label_col):
            if v > threshold:
                outliers.append(f"{lbl} ({v:.2f}s)")
    return mean, min(values), max(values), outliers


def _write_output(
    rows: List[dict],
    neg_results: List[dict],
    args: argparse.Namespace,
    final_loss: float,
    threshold: float,
    train_set: List[str],
    settings: Settings,
) -> None:
    _BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _BENCHMARKS_DIR / "results.csv"
    json_path = _BENCHMARKS_DIR / "results.json"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "similarity_threshold": threshold,
        "darkness_threshold": settings.darkness_threshold,
        "ocr_text_likelihood_threshold": settings.ocr_text_likelihood_threshold,
        "blur_threshold": BLUR_THRESHOLD,
        "detection_quality_low_max": settings.detection_quality_low_max,
        "detection_quality_high_min": settings.detection_quality_high_min,
        "n_train": len(train_set),
        "n_test": len(rows),
        "n_negatives": len(neg_results),
        "final_triplet_loss": round(final_loss, 6),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(json_path, "w") as f:
        json.dump({"metadata": metadata, "results": rows, "negatives": neg_results}, f, indent=2)

    print(f"\nResults written to: {csv_path}, {json_path}")


def _print_summary(
    retrieval: List[dict],
    dino: Dict[str, dict],
    depth: Dict[str, dict],
    neg_results: List[dict],
    final_loss: float,
    threshold: float,
    no_depth: bool,
) -> None:
    n = len(retrieval)
    bl_correct = sum(r["baseline_correct"] for r in retrieval)
    pe_correct = sum(r["personalized_correct"] for r in retrieval)
    bl_sims = [r["baseline_similarity"] for r in retrieval]
    pe_sims = [r["personalized_similarity"] for r in retrieval]
    gaps = [r["similarity_gap"] for r in retrieval]
    fnames = [r["image"] for r in retrieval]

    mean_bl = sum(bl_sims) / n if n else 0.0
    mean_pe = sum(pe_sims) / n if n else 0.0
    mean_gap = sum(gaps) / n if n else 0.0

    det_vals = list(dino.values())
    det_count = sum(v["detected"] for v in det_vals)
    det_total = len(det_vals)

    depth_evaluated = [v for v in depth.values() if v["abs_error"] is not None]
    mean_abs = (sum(v["abs_error"] for v in depth_evaluated) / len(depth_evaluated)
                if depth_evaluated else None)
    mean_pct = (sum(v["pct_error"] for v in depth_evaluated) / len(depth_evaluated)
                if depth_evaluated else None)

    delta_pp = 100.0 * (pe_correct - bl_correct) / max(n, 1)
    sign = "+" if delta_pp >= 0 else ""
    gap_sign = "+" if mean_gap >= 0 else ""

    n_neg = len(neg_results)
    bl_fp = sum(r["baseline_fp"] for r in neg_results)
    pe_fp = sum(r["personalized_fp"] for r in neg_results)

    print("\n=== Benchmark Summary ===")
    print(f"Similarity threshold: {threshold:.2f}  (production value)")
    print(f"Test images:          {n}")
    print()
    print("--- Retrieval ---")
    print(f"Baseline accuracy:    {bl_correct} / {n}  ({100*bl_correct/max(n,1):.1f}%)")
    print(f"Personalized acc.:    {pe_correct} / {n}  ({100*pe_correct/max(n,1):.1f}%)")
    print(f"Accuracy delta:       {sign}{delta_pp:.1f} pp")
    print(f"Mean sim (baseline):  {mean_bl:.4f}")
    print(f"Mean sim (personal.): {mean_pe:.4f}")
    print(f"Mean sim gap:         {gap_sign}{mean_gap:.4f}")
    print(f"Triplet final loss:   {final_loss:.4f}")
    print()
    print("--- Detection (GroundingDINO) ---")
    print(f"Detection rate:       {det_count} / {det_total}  ({100*det_count/max(det_total,1):.1f}%)")
    print()
    print("--- Depth (conditional on detection) ---")
    if no_depth or not _CHECKPOINT.exists():
        print("                      skipped")
    elif not depth_evaluated:
        print(f"Evaluated on:         0 / {det_total} images")
    else:
        print(f"Evaluated on:         {len(depth_evaluated)} / {det_total} images")
        print(f"Mean abs. error:      {mean_abs:.1f} ft")
        print(f"Mean % error:         {mean_pct:.1f}%")
    print()
    if n_neg > 0:
        delta_fp = pe_fp - bl_fp
        fp_sign = "+" if delta_fp > 0 else ""
        print("--- False Positives (Negative Images) ---")
        print(f"Baseline FP rate:     {bl_fp} / {n_neg}  ({100*bl_fp/n_neg:.1f}%)")
        print(f"Personalized FP rate: {pe_fp} / {n_neg}  ({100*pe_fp/n_neg:.1f}%)")
        print(f"Delta:                {fp_sign}{100*delta_fp/n_neg:.1f} pp")
        if bl_fp or pe_fp:
            for r in neg_results:
                tags = []
                if r["baseline_fp"]:
                    tags.append(f"BL matched {r['baseline_match']} ({r['baseline_sim']:.3f})")
                if r["personalized_fp"]:
                    tags.append(f"PL matched {r['personalized_match']} ({r['personalized_sim']:.3f})")
                if tags:
                    print(f"  FP: {r['image']}; {', '.join(tags)}")
        print()
    # Latency summary
    print("--- Latency ---")
    phases = [
        ("embed_image", [r["lat_embed_img"] for r in retrieval]),
        ("ocr",         [r["lat_ocr"] for r in retrieval if r["lat_ocr"] > 0]),
        ("embed_text",  [r["lat_embed_txt"] for r in retrieval if r["lat_embed_txt"] > 0]),
        ("detect",      [v["lat_detect"] for v in dino.values() if v["lat_detect"] > 0]),
        ("retrieve_bl", [r["lat_retrieve_bl"] for r in retrieval]),
        ("retrieve_pl", [r["lat_retrieve_pe"] for r in retrieval]),
        ("depth",       [v["lat_depth"] for v in depth.values() if v["lat_depth"] > 0]),
    ]
    for phase_name, vals in phases:
        if not vals:
            continue
        mean, lo, hi, outliers = _latency_stats(
            vals, fnames if len(vals) == len(fnames) else None)
        outlier_str = f"  outliers: {', '.join(outliers[:3])}" if outliers else ""
        print(f"  {phase_name:<14} mean {mean:.3f}s  [{lo:.3f}s - {hi:.3f}s]{outlier_str}")


# main

def main() -> None:
    args = _parse_args()
    settings = Settings()
    threshold = settings.similarity_threshold

    print(f"Dataset: {args.dataset}")
    print(f"Images:  {args.images}")
    print(f"Seed:    {args.seed}")
    print(f"Thresholds: darkness={settings.darkness_threshold:.1f}, "
          f"ocr_text_likelihood={settings.ocr_text_likelihood_threshold:.2f}, "
          f"blur={BLUR_THRESHOLD:.1f}")

    # Phase 1: Embed
    print("\n[1/7] Loading and embedding images...")
    rows = _load_dataset(args.dataset)
    embedded = _embed_rows(rows, args.images, args.no_ocr, settings)
    if not embedded:
        print("No images found. Check --images path.", file=sys.stderr)
        sys.exit(1)

    # Phase 2: Split
    print("\n[2/7] Train/test split (60/40 per label)...")
    train_set, test_set = _split(embedded, args.seed)
    print(f"  train: {len(train_set)}, test: {len(test_set)}")
    if not test_set:
        print("Not enough images for a test split.", file=sys.stderr)
        sys.exit(1)

    # Phase 3: Database
    print("\n[3/7] Building reference database...")
    database = _build_database(train_set, embedded)
    print(f"  {len(database)} entries")

    # Phase 4: Train
    print(f"\n[4/7] Training projection head ({args.epochs} epochs)...")
    head_path = _BENCHMARKS_DIR / "projection_head_bench.pt"
    head, final_loss = _train_head(train_set, embedded, args.epochs, args.lr, head_path)
    print(f"  final loss: {final_loss:.4f}")

    # Phase 5: Retrieval
    print("\n[5/7] Evaluating retrieval...")
    retrieval_results = _eval_retrieval(test_set, embedded, database, head, threshold)

    # Phase 6: Detection
    print("\n[6/7] Evaluating GroundingDINO detection...")
    dino_results = _eval_detection(test_set, embedded)

    # Phase 7: Depth
    print("\n[7/7] Evaluating depth estimation...")
    depth_results = _eval_depth(
        test_set, embedded, dino_results, args.focal_length, args.no_depth)

    # Negative FP evaluation
    neg_results: List[dict] = []
    if args.negative_dataset.exists():
        print("\n[neg] Evaluating false positive rate on negative images...")
        neg_rows = _load_negative_dataset(args.negative_dataset)
        neg_embedded = _embed_rows(neg_rows, args.images_neg, args.no_ocr, settings)
        if neg_embedded:
            neg_results = _eval_negatives(neg_embedded, database, head, threshold)
            print(f"  {len(neg_results)} negative images evaluated")
    else:
        print(f"\n[neg] negative_dataset.csv not found; skipping FP evaluation")

    # Output
    _print_summary(retrieval_results, dino_results, depth_results,
                   neg_results, final_loss, threshold, args.no_depth)
    merged = _merge_rows(retrieval_results, dino_results, depth_results)
    _write_output(merged, neg_results, args, final_loss, threshold, train_set, settings)


if __name__ == "__main__":
    main()
