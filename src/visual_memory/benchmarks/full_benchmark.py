"""Full system benchmark: retrieval, detection, and depth evaluation.

Usage:
    python -m visual_memory.benchmarks.full_benchmark \\
        --dataset benchmarks/dataset.csv \\
        --images benchmarks/images \\
        --seed 42

    # Fast smoke test (skip depth + OCR):
    python -m visual_memory.benchmarks.full_benchmark \\
        --dataset benchmarks/dataset.csv \\
        --images benchmarks/images \\
        --seed 42 --no-depth --no-ocr --epochs 5
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

from visual_memory.config import Settings
from visual_memory.engine.embedding import make_combined_embedding
from visual_memory.engine.model_registry import registry
from visual_memory.learning import ProjectionHead, ProjectionTrainer
from visual_memory.utils.image_utils import load_image
from visual_memory.utils.similarity_utils import find_match

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_BENCHMARKS_DIR = _PROJECT_ROOT / "benchmarks"
_CHECKPOINT = _PROJECT_ROOT / "checkpoints" / "depth_pro.pt"


# ---- arg parsing ----

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full system benchmark")
    p.add_argument("--dataset", type=Path, default=_BENCHMARKS_DIR / "dataset.csv")
    p.add_argument("--images", type=Path, default=_BENCHMARKS_DIR / "images")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--focal-length", type=float, default=None,
                   help="Focal length in pixels (iPhone 15 Plus: 3094.0). "
                        "Omit to let Depth Pro infer.")
    p.add_argument("--epochs", type=int, default=20,
                   help="Projection head training epochs (default: 20)")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Projection head learning rate (default: 1e-4)")
    p.add_argument("--no-depth", action="store_true",
                   help="Skip Depth Pro evaluation")
    p.add_argument("--no-ocr", action="store_true",
                   help="Skip PaddleOCR (faster for quick runs)")
    return p.parse_args()


# ---- phase 1: load & embed ----

def _load_dataset(csv_path: Path) -> List[dict]:
    rows = []
    with open(csv_path, newline="") as f:
        # Skip comment lines starting with '#'
        filtered = (line for line in f if not line.lstrip().startswith("#"))
        for row in csv.DictReader(filtered):
            rows.append({
                "image": row["image"],
                "label": row["label"],
                "distance_ft": float(row["ground_truth_distance_ft"]),
                "dino_prompt": row["dino_prompt"],
            })
    return rows


def _embed_rows(
    rows: List[dict],
    images_dir: Path,
    no_ocr: bool,
) -> Dict[str, dict]:
    embedded: Dict[str, dict] = {}
    n = len(rows)
    for i, row in enumerate(rows):
        fname = row["image"]
        img_path = images_dir / fname
        if not img_path.exists():
            print(f"  [warn] image not found, skipping: {img_path}", file=sys.stderr)
            continue
        print(f"  [{i+1}/{n}] embedding: {fname} ({row['label']})")
        img = load_image(str(img_path))
        img_emb = registry.img_embedder.embed(img)
        text_emb = None
        if not no_ocr:
            text = registry.text_recognizer.recognize(img)
            if text and text.strip():
                text_emb = registry.text_embedder.embed_text(text)
        emb = make_combined_embedding(img_emb, text_emb)
        embedded[fname] = {
            "label": row["label"],
            "distance_ft": row["distance_ft"],
            "dino_prompt": row["dino_prompt"],
            "embedding": emb,
            "image": img,
        }
    return embedded


# ---- phase 2: train/test split ----

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


# ---- phase 3: build reference database ----

def _build_database(
    train_set: List[str],
    embedded: Dict[str, dict],
) -> List[Tuple[str, torch.Tensor]]:
    # One entry per training image — identical format to production ScanPipeline.
    return [(embedded[f]["label"], embedded[f]["embedding"]) for f in train_set]


# ---- phase 4: generate triplets and train projection head ----

def _build_triplets(
    train_set: List[str],
    embedded: Dict[str, dict],
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    # One triplet per training image where both same-label and diff-label partners exist.
    triplets = []
    for fname in train_set:
        anchor_emb = embedded[fname]["embedding"]
        label = embedded[fname]["label"]
        same = [f for f in train_set if f != fname and embedded[f]["label"] == label]
        diff = [f for f in train_set if embedded[f]["label"] != label]
        if not same or not diff:
            continue
        positive_emb = embedded[same[0]]["embedding"]
        negative_emb = embedded[diff[0]]["embedding"]
        triplets.append((anchor_emb, positive_emb, negative_emb))
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
        print("  [warn] no triplets available "
              "(need >= 2 labels, each with >= 2 train images)")
        head.eval()
        return head, 0.0
    print(f"  {len(triplets)} triplets, {epochs} epochs")
    trainer = ProjectionTrainer(head, lr=lr)
    final_loss = trainer.train(triplets, epochs=epochs)
    head.eval()
    head.save(save_path)
    print(f"  saved: {save_path}")
    return head, final_loss


# ---- phase 5: retrieval evaluation ----

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

        bl_label, bl_sim = find_match(test_emb, database, threshold)
        proj_query = head.project(test_emb)
        pe_label, pe_sim = find_match(proj_query, projected_db, threshold)

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
        })
    return results


# ---- phase 6: grounding dino evaluation ----

def _eval_detection(
    test_set: List[str],
    embedded: Dict[str, dict],
) -> Dict[str, dict]:
    detector = registry.gdino_detector
    results: Dict[str, dict] = {}
    for fname in test_set:
        data = embedded[fname]
        det = detector.detect(data["image"], data["dino_prompt"])
        if det:
            results[fname] = {
                "detected": 1,
                "confidence": det["score"],
                "box": det["box"],
            }
        else:
            results[fname] = {
                "detected": 0,
                "confidence": 0.0,
                "box": None,
            }
    return results


# ---- phase 7: depth evaluation ----

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
    blank = {"predicted_ft": None, "abs_error": None, "pct_error": None}
    results = {fname: dict(blank) for fname in test_set}

    if no_depth:
        return results

    estimator = _load_depth_estimator()
    if estimator is None:
        print("  [info] depth checkpoint not found — skipping depth evaluation")
        return results

    for fname in test_set:
        data = embedded[fname]
        det = dino_results[fname]
        gt_dist = data["distance_ft"]
        if not det["detected"] or gt_dist <= 0:
            continue
        depth_map = estimator.estimate(data["image"], focal_length_px=focal_length)
        pred_ft = estimator.get_depth_at_bbox(depth_map, det["box"])
        abs_err = abs(pred_ft - gt_dist)
        pct_err = abs_err / gt_dist * 100.0
        results[fname] = {
            "predicted_ft": pred_ft,
            "abs_error": abs_err,
            "pct_error": pct_err,
        }
    return results


# ---- phase 8: output ----

_CSV_FIELDS = [
    "image", "label", "ground_truth_distance_ft",
    "baseline_similarity", "personalized_similarity", "similarity_gap",
    "baseline_correct", "personalized_correct",
    "dino_detected", "dino_confidence",
    "predicted_distance_ft", "depth_absolute_error", "depth_percentage_error",
]


def _merge_rows(
    retrieval: List[dict],
    dino: Dict[str, dict],
    depth: Dict[str, dict],
) -> List[dict]:
    rows = []
    for r in retrieval:
        fname = r["image"]
        d = dino.get(fname, {"detected": 0, "confidence": 0.0})
        dep = depth.get(fname, {"predicted_ft": None, "abs_error": None, "pct_error": None})
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
                round(dep["predicted_ft"], 3) if dep["predicted_ft"] is not None else ""
            ),
            "depth_absolute_error": (
                round(dep["abs_error"], 3) if dep["abs_error"] is not None else ""
            ),
            "depth_percentage_error": (
                round(dep["pct_error"], 2) if dep["pct_error"] is not None else ""
            ),
        })
    return rows


def _write_output(
    rows: List[dict],
    args: argparse.Namespace,
    final_loss: float,
    threshold: float,
    train_set: List[str],
    n_test: int,
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
        "n_train": len(train_set),
        "n_test": n_test,
        "final_triplet_loss": round(final_loss, 6),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(json_path, "w") as f:
        json.dump({"metadata": metadata, "results": rows}, f, indent=2)

    print(f"\nResults written to: {csv_path}, {json_path}")


def _print_summary(
    retrieval: List[dict],
    dino: Dict[str, dict],
    depth: Dict[str, dict],
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

    mean_bl = sum(bl_sims) / n if n else 0.0
    mean_pe = sum(pe_sims) / n if n else 0.0
    mean_gap = sum(gaps) / n if n else 0.0

    det_vals = list(dino.values())
    det_count = sum(v["detected"] for v in det_vals)
    det_total = len(det_vals)

    depth_evaluated = [v for v in depth.values() if v["abs_error"] is not None]
    mean_abs = (
        sum(v["abs_error"] for v in depth_evaluated) / len(depth_evaluated)
        if depth_evaluated else None
    )
    mean_pct = (
        sum(v["pct_error"] for v in depth_evaluated) / len(depth_evaluated)
        if depth_evaluated else None
    )

    delta_pp = 100.0 * (pe_correct - bl_correct) / max(n, 1)
    sign = "+" if delta_pp >= 0 else ""
    gap_sign = "+" if mean_gap >= 0 else ""

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
        print(
            f"Evaluated on:         0 / {det_total} images  "
            "(no detected images with ground truth distance)"
        )
    else:
        print(
            f"Evaluated on:         {len(depth_evaluated)} / {det_total} images  "
            "(detected + have ground truth)"
        )
        print(f"Mean abs. error:      {mean_abs:.1f} ft")
        print(f"Mean % error:         {mean_pct:.1f}%")


# ---- main ----

def main() -> None:
    args = _parse_args()
    settings = Settings()
    threshold = settings.similarity_threshold

    print(f"Dataset: {args.dataset}")
    print(f"Images:  {args.images}")
    print(f"Seed:    {args.seed}")

    # Phase 1
    print("\n[1/7] Loading dataset and embedding images...")
    rows = _load_dataset(args.dataset)
    embedded = _embed_rows(rows, args.images, args.no_ocr)
    if not embedded:
        print("No images embedded. Check --images path and dataset.csv.", file=sys.stderr)
        sys.exit(1)

    # Phase 2
    print("\n[2/7] Splitting train/test (60/40 per label)...")
    train_set, test_set = _split(embedded, args.seed)
    print(f"  train: {len(train_set)}, test: {len(test_set)}")
    if not test_set:
        print(
            "Not enough images for a test split "
            "(need at least 2 images per label).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Phase 3
    print("\n[3/7] Building reference database...")
    database = _build_database(train_set, embedded)
    print(f"  {len(database)} entries")

    # Phase 4
    print(f"\n[4/7] Training projection head ({args.epochs} epochs)...")
    head_path = _BENCHMARKS_DIR / "projection_head_bench.pt"
    head, final_loss = _train_head(train_set, embedded, args.epochs, args.lr, head_path)
    print(f"  final loss: {final_loss:.4f}")

    # Phase 5
    print("\n[5/7] Evaluating retrieval...")
    retrieval_results = _eval_retrieval(test_set, embedded, database, head, threshold)

    # Phase 6
    print("\n[6/7] Evaluating GroundingDINO detection...")
    dino_results = _eval_detection(test_set, embedded)

    # Phase 7
    print("\n[7/7] Evaluating depth estimation...")
    depth_results = _eval_depth(
        test_set, embedded, dino_results, args.focal_length, args.no_depth
    )

    # Phase 8
    _print_summary(retrieval_results, dino_results, depth_results, final_loss, threshold, args.no_depth)
    merged = _merge_rows(retrieval_results, dino_results, depth_results)
    _write_output(merged, args, final_loss, threshold, train_set, len(test_set))


if __name__ == "__main__":
    main()
