"""Full system benchmark: retrieval, detection, depth, latency, and false-positive rate.

Usage:
    python -m visual_memory.benchmarks.full_benchmark \
        --dataset benchmarks/dataset.csv \
        --images benchmarks/images

    # Fast smoke test (skip depth + OCR, ~5-10 min):
    python -m visual_memory.benchmarks.full_benchmark \
        --dataset benchmarks/dataset.csv \
        --images benchmarks/images \
        --no-depth --no-ocr --epochs 5

Split strategy: 1ft_bright_clean images per label form the reference DB (simulating a
teach session), all 1ft images train the projection head, and 3ft/6ft images are the
test set (simulating real scan conditions after teaching).
"""
from __future__ import annotations

import argparse
import gc
import csv
import hashlib
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# Same second-pass templates as RememberPipeline - must stay in sync.
_SECOND_PASS_TEMPLATES = [
    "a {prompt}",
    "{prompt} object",
    "close up of a {prompt}",
]

from visual_memory.config import Settings
from visual_memory.engine.embedding import make_combined_embedding
from visual_memory.engine.model_registry import registry
from visual_memory.engine.text_recognition import TextRecognizer
from visual_memory.learning import ProjectionHead, ProjectionTrainer
from visual_memory.utils.image_utils import load_image
from visual_memory.utils.memory_monitor import MemoryMonitor
from visual_memory.utils.quality_utils import mean_luminance, estimate_text_likelihood
from visual_memory.utils.similarity_utils import cosine_similarity, find_match_dynamic_threshold

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_BENCHMARKS_DIR = _PROJECT_ROOT / "benchmarks"
_CHECKPOINT = _PROJECT_ROOT / "checkpoints" / "depth_pro.pt"
_RESULTS_SCHEMA_VERSION = 2

BLUR_THRESHOLD = 100.0
DOCUMENT_LABEL_KEYWORDS = {
    "receipt",
    "ticket",
    "document",
    "paper",
    "note",
    "label",
    "barcode",
    "passport",
    "license",
    "id",
    "card",
}


def _threshold_for_label(label: str, is_document: bool, thresholds: Dict[str, float], mode: str) -> float:
    if is_document:
        return float(thresholds["document"])
    return float(thresholds["personalized"] if mode == "personalized" else thresholds["baseline"])


def _margin_for_label(label: str, is_document: bool, settings: Settings) -> float:
    if is_document:
        return float(settings.get_scan_similarity_margin_document())
    return float(settings.get_scan_similarity_margin())


def _match_with_runtime_gates(
    query_embedding: torch.Tensor,
    database_embeddings: List[Tuple[str, torch.Tensor]],
    is_document_sample: bool,
    thresholds: Dict[str, float],
    settings: Settings,
    mode: str,
) -> tuple[Optional[str], float, float]:
    return find_match_dynamic_threshold(
        query_embedding,
        database_embeddings,
        lambda lbl: _threshold_for_label(lbl, is_document_sample, thresholds, mode=mode),
        lambda lbl: _margin_for_label(lbl, is_document_sample, settings),
    )


def _set_reproducible(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_depth_checkpoint(args: argparse.Namespace) -> Path:
    if args.depth_checkpoint:
        return args.depth_checkpoint.expanduser()
    env_override = os.environ.get("DEPTH_CHECKPOINT_PATH", "").strip()
    if env_override:
        return Path(env_override).expanduser()
    return _CHECKPOINT


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
    p.add_argument("--focal-length", type=float, default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--split-manifest",
        type=Path,
        default=_BENCHMARKS_DIR / "split_manifest.json",
        help="Pinned split manifest path (load if present, write if missing)",
    )
    p.add_argument(
        "--refresh-split-manifest",
        action="store_true",
        help="Regenerate split manifest instead of reusing an existing file",
    )
    p.add_argument("--train-per-label", type=int, default=6,
                   help="How many images per label go to train split (dataset has 12 per label)")
    p.add_argument("--no-train-augment", action="store_true",
                   help="Disable mirrored/blurred/darkened augmentations for train images")
    p.add_argument("--similarity-threshold", type=float, default=None,
                   help="Override retrieval threshold (default: Settings.similarity_threshold)")
    p.add_argument(
        "--baseline-threshold",
        type=float,
        default=None,
        help="Override baseline retrieval threshold (non-document samples)",
    )
    p.add_argument(
        "--personalized-threshold",
        type=float,
        default=None,
        help="Override personalized retrieval threshold (non-document samples)",
    )
    p.add_argument(
        "--document-threshold",
        type=float,
        default=None,
        help="Override retrieval threshold for document-like samples",
    )
    p.add_argument(
        "--regression-baseline",
        type=Path,
        default=None,
        help="Optional previous results.json path for drift gating",
    )
    p.add_argument(
        "--max-accuracy-drop-pp",
        type=float,
        default=3.0,
        help="Maximum allowed personalized accuracy drop (percentage points)",
    )
    p.add_argument(
        "--max-fp-increase-pp",
        type=float,
        default=2.0,
        help="Maximum allowed personalized false-positive increase on holdout set (pp)",
    )
    p.add_argument(
        "--max-fn-increase-pp",
        type=float,
        default=3.0,
        help="Maximum allowed personalized false-negative increase on test set (pp)",
    )
    p.add_argument("--no-depth", action="store_true")
    p.add_argument("--no-ocr", action="store_true")
    p.add_argument(
        "--depth-checkpoint",
        type=Path,
        default=None,
        help="Optional override path for depth_pro.pt checkpoint",
    )
    p.add_argument(
        "--no-fp-holdout-tests",
        action="store_true",
        help="Disable 60-case leave-one-label-out FP evaluation on test split",
    )
    p.add_argument(
        "--no-fp-expanded-tests",
        action="store_true",
        help="Disable expanded FP evaluation against all other-label images",
    )
    p.add_argument("--threshold-sweep-min", type=float, default=0.10)
    p.add_argument("--threshold-sweep-max", type=float, default=0.90)
    p.add_argument("--threshold-sweep-step", type=float, default=0.01)
    p.add_argument("--target-holdout-fp", type=float, default=0.40)
    p.add_argument("--min-personalized-accuracy", type=float, default=0.15)
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


def _embed_rows(
    rows: List[dict],
    images_dir: Path,
    no_ocr: bool,
    settings: Settings,
    monitor: Optional[MemoryMonitor] = None,
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
            "image_path": str(img_path),
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
        if monitor is not None and (i + 1) % 20 == 0:
            if monitor.suggest_throttle():
                monitor.log_memory_state(level="warning")
                time.sleep(5)
                gc.collect()
    return embedded


# phase 2: split into db set, triplet train set, and test set
#
# DB set (reference embeddings for cosine matching): 1ft_bright_clean images only.
# This mirrors a real teach session - user holds the item close under good lighting.
#
# Triplet train set: all 1ft images (4 per label). Gives enough positives per label
# for the projection head to train without leaking 3ft/6ft conditions into training.
#
# Test set: all 3ft and 6ft images - the actual scan conditions.

def _split(
    embedded: Dict[str, dict],
) -> Tuple[List[str], List[str], List[str]]:
    """Return (train_set, db_set, test_set).

    train_set: all 1ft images, used to build projection head triplets.
    db_set: 1ft_bright_clean subset of train_set, used as the reference DB for matching.
    test_set: 3ft and 6ft images, used to evaluate retrieval under real scan conditions.
    """
    train_set = [f for f in embedded if "_1ft_" in f]
    db_set = [f for f in train_set if "_1ft_bright_clean." in f]
    test_set = [f for f in embedded if "_1ft_" not in f]
    return train_set, db_set, test_set


def _condition_buckets(image_name: str, distance_ft: float, is_dark: bool) -> dict:
    stem = image_name.rsplit(".", 1)[0]
    tokens = stem.split("_")
    distance_bucket = next((t for t in tokens if t.endswith("ft")), "")
    lighting_bucket = next((t for t in tokens if t in {"bright", "dim", "dark"}), "")
    cleanliness_bucket = next((t for t in tokens if t in {"clean", "messy"}), "")
    if not distance_bucket:
        distance_bucket = f"{int(round(distance_ft))}ft"
    if not lighting_bucket:
        lighting_bucket = "dim" if is_dark else "bright"
    condition_bucket = "_".join(
        [x for x in [distance_bucket, lighting_bucket, cleanliness_bucket] if x]
    )
    return {
        "distance_bucket": distance_bucket,
        "lighting_bucket": lighting_bucket,
        "cleanliness_bucket": cleanliness_bucket,
        "condition_bucket": condition_bucket,
    }


def _is_document_like(label: str, prompt: str) -> bool:
    hay = f"{label} {prompt}".lower()
    return any(k in hay for k in DOCUMENT_LABEL_KEYWORDS)


def _split_fixed_60_60(
    embedded: Dict[str, dict],
    train_per_label: int,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Return (train_set, db_set, test_set) with strict 60/60 split.

    The dataset has 10 labels x 12 images each = 120.
    We sample `train_per_label` per label for train and use the remainder for test.
    This prevents cross-label leakage while preserving balanced class counts.
    """
    by_label: Dict[str, List[str]] = {}
    for fname, row in embedded.items():
        by_label.setdefault(row["label"], []).append(fname)

    rng = np.random.default_rng(seed)
    train_set: List[str] = []
    test_set: List[str] = []
    for label, files in sorted(by_label.items()):
        ordered = sorted(files)
        if len(ordered) < train_per_label + 1:
            raise ValueError(
                f"label {label} has only {len(ordered)} images; "
                f"need at least {train_per_label + 1} for train/test split"
            )
        perm = rng.permutation(len(ordered))
        train_idx = set(int(i) for i in perm[:train_per_label])
        train_label_files = [ordered[i] for i in range(len(ordered)) if i in train_idx]
        test_label_files = [ordered[i] for i in range(len(ordered)) if i not in train_idx]
        train_set.extend(train_label_files)
        test_set.extend(test_label_files)

    # Reference DB must contain only train images, one per label.
    db_set: List[str] = []
    for label in sorted(by_label.keys()):
        label_train = sorted([f for f in train_set if embedded[f]["label"] == label])
        if not label_train:
            raise ValueError(f"no train images for label {label}")
        preferred = [f for f in label_train if "_1ft_bright_clean." in f]
        db_set.append(preferred[0] if preferred else label_train[0])

    return sorted(train_set), sorted(db_set), sorted(test_set)


def _dataset_signature(embedded: Dict[str, dict]) -> dict:
    pairs = [f"{fname}:{embedded[fname]['label']}" for fname in sorted(embedded)]
    digest = hashlib.sha256("\n".join(pairs).encode("utf-8")).hexdigest()
    return {
        "count": len(pairs),
        "sha256": digest,
    }


def _write_split_manifest(
    path: Path,
    signature: dict,
    seed: int,
    train_per_label: int,
    train_set: List[str],
    db_set: List[str],
    test_set: List[str],
) -> None:
    payload = {
        "version": 1,
        "signature": signature,
        "seed": seed,
        "train_per_label": train_per_label,
        "train_set": sorted(train_set),
        "db_set": sorted(db_set),
        "test_set": sorted(test_set),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_split_manifest(path: Path, embedded: Dict[str, dict]) -> Tuple[List[str], List[str], List[str], dict]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid split manifest payload: {path}")

    train_set = payload.get("train_set") or []
    db_set = payload.get("db_set") or []
    test_set = payload.get("test_set") or []
    if not isinstance(train_set, list) or not isinstance(db_set, list) or not isinstance(test_set, list):
        raise ValueError(f"invalid split lists in manifest: {path}")

    all_names = set(embedded.keys())
    missing = sorted({str(x) for x in (train_set + db_set + test_set) if str(x) not in all_names})
    if missing:
        raise ValueError(
            f"split manifest references {len(missing)} missing images "
            f"(first: {missing[:3]}). Re-run with --refresh-split-manifest."
        )

    signature = payload.get("signature")
    info = {
        "signature": signature if isinstance(signature, dict) else {},
        "seed": int(payload.get("seed", 42)),
        "train_per_label": int(payload.get("train_per_label", 6)),
    }
    return sorted(train_set), sorted(db_set), sorted(test_set), info


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


def _augment_train_embedding(
    embedding: torch.Tensor,
    image_path: str,
    do_augment: bool,
) -> List[torch.Tensor]:
    """Return base embedding plus optional deterministic visual augmentations."""
    out = [embedding]
    if not do_augment:
        return out
    try:
        image = load_image(image_path)
    except Exception:
        return out

    variants = [
        ImageOps.mirror(image),
        image.filter(ImageFilter.GaussianBlur(radius=1.6)),
        ImageEnhance.Brightness(image).enhance(0.62),
    ]
    for variant in variants:
        try:
            v_emb = registry.img_embedder.embed(variant)
            out.append(make_combined_embedding(v_emb, None))
        except Exception:
            continue
    return out


def _build_triplets_augmented(
    train_set: List[str],
    embedded: Dict[str, dict],
    augment_train: bool,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Build richer triplets from train identities only.

    Augmentations are generated only from train images to avoid leakage from test images.
    """
    triplets: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    by_label: Dict[str, List[str]] = {}
    for fname in train_set:
        by_label.setdefault(embedded[fname]["label"], []).append(fname)

    for label, files in by_label.items():
        if len(files) < 2:
            continue
        negatives = [f for f in train_set if embedded[f]["label"] != label]
        if not negatives:
            continue
        for i, fname in enumerate(files):
            anchor = embedded[fname]
            positive_name = files[(i + 1) % len(files)]
            positive = embedded[positive_name]
            neg_name = negatives[i % len(negatives)]
            negative = embedded[neg_name]
            anchors = _augment_train_embedding(anchor["embedding"], anchor["image_path"], augment_train)
            positives = _augment_train_embedding(positive["embedding"], positive["image_path"], augment_train)
            for a_emb in anchors:
                for p_emb in positives[:2]:
                    triplets.append((a_emb, p_emb, negative["embedding"]))
    return triplets


def _train_head(
    train_set: List[str],
    embedded: Dict[str, dict],
    epochs: int,
    lr: float,
    save_path: Path,
    augment_train: bool,
) -> Tuple[ProjectionHead, float]:
    head = ProjectionHead()
    triplets = _build_triplets_augmented(train_set, embedded, augment_train=augment_train)
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
    thresholds: Dict[str, float],
    settings: Settings,
    monitor: Optional[MemoryMonitor] = None,
) -> List[dict]:
    head.eval()
    projected_db = [(lbl, head.project(e)) for lbl, e in database]
    results = []
    for i, fname in enumerate(test_set):
        data = embedded[fname]
        test_emb = data["embedding"]
        true_label = data["label"]
        is_document = bool(data.get("is_document", False))
        baseline_threshold = _threshold_for_label(true_label, is_document, thresholds, mode="baseline")
        personalized_threshold = _threshold_for_label(true_label, is_document, thresholds, mode="personalized")
        baseline_margin_threshold = _margin_for_label(true_label, is_document, settings)
        personalized_margin_threshold = _margin_for_label(true_label, is_document, settings)

        t0 = time.perf_counter()
        bl_label, bl_sim, bl_margin = _match_with_runtime_gates(
            test_emb,
            database,
            is_document,
            thresholds,
            settings,
            mode="baseline",
        )
        lat_bl = time.perf_counter() - t0

        t0 = time.perf_counter()
        pe_label, pe_sim, pe_margin = _match_with_runtime_gates(
            head.project(test_emb),
            projected_db,
            is_document,
            thresholds,
            settings,
            mode="personalized",
        )
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
            "distance_bucket": data.get("distance_bucket", ""),
            "lighting_bucket": data.get("lighting_bucket", ""),
            "cleanliness_bucket": data.get("cleanliness_bucket", ""),
            "condition_bucket": data.get("condition_bucket", ""),
            "is_document": int(is_document),
            "baseline_threshold_used": round(baseline_threshold, 6),
            "personalized_threshold_used": round(personalized_threshold, 6),
            "baseline_margin_threshold_used": round(baseline_margin_threshold, 6),
            "personalized_margin_threshold_used": round(personalized_margin_threshold, 6),
            "baseline_similarity_margin": round(bl_margin, 6),
            "personalized_similarity_margin": round(pe_margin, 6),
        })
        if monitor is not None and (i + 1) % 20 == 0:
            if monitor.suggest_throttle():
                monitor.log_memory_state(level="warning")
                time.sleep(5)
                gc.collect()
    return results


def _build_similarity_stats(
    test_set: List[str],
    embedded: Dict[str, dict],
    database: List[Tuple[str, torch.Tensor]],
    head: ProjectionHead,
) -> List[dict]:
    if not database or not test_set:
        return []
    head.eval()
    projected_db = [(lbl, head.project(emb)) for (lbl, emb) in database]
    stats: List[dict] = []
    for fname in test_set:
        row = embedded[fname]
        q = row["embedding"]
        qh = head.project(q)
        true_label = row["label"]
        best_true_bl = -1.0
        best_wrong_bl = -1.0
        for lbl, emb in database:
            sim = float(cosine_similarity(q, emb).item())
            if lbl == true_label:
                best_true_bl = max(best_true_bl, sim)
            else:
                best_wrong_bl = max(best_wrong_bl, sim)
        best_true_pe = -1.0
        best_wrong_pe = -1.0
        for lbl, emb in projected_db:
            sim = float(cosine_similarity(qh, emb).item())
            if lbl == true_label:
                best_true_pe = max(best_true_pe, sim)
            else:
                best_wrong_pe = max(best_wrong_pe, sim)
        stats.append(
            {
                "image": fname,
                "label": true_label,
                "distance_ft": row["distance_ft"],
                "is_dark": bool(row.get("is_dark", False)),
                "is_document": bool(row.get("is_document", False)),
                "distance_bucket": row.get("distance_bucket", ""),
                "lighting_bucket": row.get("lighting_bucket", ""),
                "cleanliness_bucket": row.get("cleanliness_bucket", ""),
                "condition_bucket": row.get("condition_bucket", ""),
                "best_true_bl": best_true_bl,
                "best_wrong_bl": max(best_wrong_bl, 0.0),
                "best_true_pe": best_true_pe,
                "best_wrong_pe": max(best_wrong_pe, 0.0),
            }
        )
    return stats


def _metrics_at_threshold(
    stats: List[dict],
    baseline_threshold: float,
    personalized_threshold: float,
    document_threshold: float,
) -> dict:
    n = max(len(stats), 1)
    bl_correct = 0
    pe_correct = 0
    bl_holdout_fp = 0
    pe_holdout_fp = 0
    weighted_recall_hits = 0.0
    weighted_recall_total = 0.0
    deprioritized_count = 0
    for s in stats:
        is_document = bool(s.get("is_document", False))
        is_deprioritized = (
            s.get("distance_bucket") == "6ft"
            and s.get("lighting_bucket") == "dim"
            and s.get("cleanliness_bucket", "") in {"messy", "clean"}
        )
        bl_t = document_threshold if is_document else baseline_threshold
        pe_t = document_threshold if is_document else personalized_threshold
        bl_ok = s["best_true_bl"] >= bl_t and s["best_true_bl"] >= s["best_wrong_bl"]
        pe_ok = s["best_true_pe"] >= pe_t and s["best_true_pe"] >= s["best_wrong_pe"]
        if bl_ok:
            bl_correct += 1
        if pe_ok:
            pe_correct += 1
        if s["best_wrong_bl"] >= bl_t:
            bl_holdout_fp += 1
        if s["best_wrong_pe"] >= pe_t:
            pe_holdout_fp += 1
        if s.get("distance_bucket") in {"1ft", "2ft", "3ft"} and s.get("lighting_bucket") == "bright":
            w = 0.2 if is_deprioritized else 1.0
            weighted_recall_total += w
            if pe_ok:
                weighted_recall_hits += w
        if is_deprioritized:
            deprioritized_count += 1
    return {
        "baseline_threshold": float(round(baseline_threshold, 3)),
        "personalized_threshold": float(round(personalized_threshold, 3)),
        "document_threshold": float(round(document_threshold, 3)),
        "baseline_accuracy": bl_correct / n,
        "personalized_accuracy": pe_correct / n,
        "baseline_holdout_fp_rate": bl_holdout_fp / n,
        "personalized_holdout_fp_rate": pe_holdout_fp / n,
        "priority_recall_1to3ft_bright": (
            weighted_recall_hits / weighted_recall_total if weighted_recall_total > 0 else 0.0
        ),
        "deprioritized_samples": deprioritized_count,
    }


def _validate_similarity_threshold(value: float, source: str) -> float:
    value = float(value)
    if not (0.0 < value <= 1.0):
        raise ValueError(f"{source} must be in (0, 1], got {value}")
    return value


def _tune_threshold_with_holdout(
    stats: List[dict],
    floor: float,
    ceiling: float,
    step: float,
    target_holdout_fp: float,
    min_personalized_accuracy: float,
) -> Tuple[dict, List[dict], dict]:
    floor = _validate_similarity_threshold(floor, "threshold sweep minimum")
    ceiling = _validate_similarity_threshold(ceiling, "threshold sweep maximum")
    if not stats:
        thr = float(round(floor, 3))
        return (
            {"baseline": thr, "personalized": thr, "document": thr},
            [],
            {
                "selection_reason": "fallback_no_stats",
                "selected_thresholds": {"baseline": thr, "personalized": thr, "document": thr},
            },
        )
    if ceiling < floor:
        floor, ceiling = ceiling, floor
    if step <= 0:
        step = 0.01
    candidates: List[float] = [
        _validate_similarity_threshold(float(round(v, 3)), "threshold sweep candidate")
        for v in np.arange(floor, ceiling + 1e-9, step)
    ]
    sweep: List[dict] = []
    for baseline_t in candidates:
        for personalized_t in candidates:
            for document_t in candidates:
                sweep.append(
                    _metrics_at_threshold(
                        stats,
                        baseline_threshold=baseline_t,
                        personalized_threshold=personalized_t,
                        document_threshold=document_t,
                    )
                )
    feasible = [
        r for r in sweep
        if r["personalized_holdout_fp_rate"] <= target_holdout_fp
        and r["personalized_accuracy"] >= min_personalized_accuracy
    ]
    if feasible:
        best = min(
            feasible,
            key=lambda r: (
                r["personalized_holdout_fp_rate"],
                -r["priority_recall_1to3ft_bright"],
                -r["personalized_accuracy"],
                -r["document_threshold"],
                -r["personalized_threshold"],
                -r["baseline_threshold"],
            ),
        )
        reason = "feasible"
    else:
        best = min(
            sweep,
            key=lambda r: (
                2.0 * max(0.0, r["personalized_holdout_fp_rate"] - target_holdout_fp)
                + max(0.0, min_personalized_accuracy - r["personalized_accuracy"])
                + 0.75 * max(0.0, 0.70 - r["priority_recall_1to3ft_bright"]),
                r["personalized_holdout_fp_rate"],
                -r["priority_recall_1to3ft_bright"],
                -r["personalized_accuracy"],
                -r["document_threshold"],
                -r["personalized_threshold"],
                -r["baseline_threshold"],
            ),
        )
        reason = "least_violation"
    selected_thresholds = {
        "baseline": best["baseline_threshold"],
        "personalized": best["personalized_threshold"],
        "document": best["document_threshold"],
    }
    tuning = {
        "selection_reason": reason,
        "selected_thresholds": selected_thresholds,
        "target_holdout_fp": target_holdout_fp,
        "min_personalized_accuracy": min_personalized_accuracy,
        "priority_policy": {
            "prioritize_low_fp": True,
            "prioritize_recall_1to3ft_bright": True,
            "deprioritize_6ft_dim_messy_clean": True,
        },
        "selected_metrics": {
            "baseline_accuracy": round(best["baseline_accuracy"], 6),
            "personalized_accuracy": round(best["personalized_accuracy"], 6),
            "baseline_holdout_fp_rate": round(best["baseline_holdout_fp_rate"], 6),
            "personalized_holdout_fp_rate": round(best["personalized_holdout_fp_rate"], 6),
            "priority_recall_1to3ft_bright": round(best["priority_recall_1to3ft_bright"], 6),
        },
        "n_candidates": len(sweep),
    }
    return selected_thresholds, sweep, tuning


# phase 6: grounding dino evaluation with full production fallback chain

def _detect_with_fallback(
    detector,
    image: Image.Image,
    prompt: str,
    settings: Settings,
) -> Tuple[Optional[dict], Optional[str]]:
    """Mirror RememberPipeline._detect_with_fallback: second-pass templates then Ollama.

    Returns (detection_or_None, used_alt_prompt_or_None).
    """
    detection = detector.detect(image, prompt)
    if detection is not None:
        return detection, None

    if not settings.detection_second_pass_enabled:
        return None, None

    for template in _SECOND_PASS_TEMPLATES:
        alt_prompt = template.format(prompt=prompt)
        detection = detector.detect(image, alt_prompt)
        if detection is not None:
            return detection, alt_prompt

    # Ollama third pass - same logic as RememberPipeline._detect_with_ollama_fallback
    try:
        from visual_memory.utils.ollama_utils import _chat
        n = max(1, settings.ollama_detection_retries)
        ollama_prompt = (
            f"A vision model failed to detect '{prompt}' in an image. "
            f"Give {n} alternative phrasings that a vision model might recognize better. "
            f"Reply with only the phrases, one per line, no numbering, no explanation."
        )
        suggestions_raw = _chat(ollama_prompt)
        if suggestions_raw:
            suggestions = [s.strip() for s in suggestions_raw.splitlines() if s.strip()][:n]
            for suggestion in suggestions:
                detection = detector.detect(image, suggestion)
                if detection is not None:
                    return detection, suggestion
    except Exception:
        pass

    return None, None


def _eval_detection(
    test_set: List[str],
    embedded: Dict[str, dict],
    settings: Settings,
    monitor: Optional[MemoryMonitor] = None,
) -> Dict[str, dict]:
    detector = registry.gdino_detector
    results: Dict[str, dict] = {}
    oom_seen = False
    for i, fname in enumerate(test_set):
        data = embedded[fname]
        image = load_image(data["image_path"])
        t0 = time.perf_counter()
        if oom_seen:
            det, used_prompt = None, None
        else:
            try:
                det, used_prompt = _detect_with_fallback(detector, image, data["dino_prompt"], settings)
            except torch.OutOfMemoryError:
                oom_seen = True
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                det, used_prompt = None, None
                print("  [warn] detection OOM; marking remaining detections as unavailable")
        lat_detect = time.perf_counter() - t0
        if det:
            results[fname] = {"detected": 1, "confidence": det["score"],
                              "box": det["box"], "lat_detect": lat_detect,
                              "second_pass_prompt": used_prompt}
        else:
            results[fname] = {"detected": 0, "confidence": 0.0,
                              "box": None, "lat_detect": lat_detect,
                              "second_pass_prompt": None}
        if monitor is not None and (i + 1) % 20 == 0:
            if monitor.suggest_throttle():
                monitor.log_memory_state(level="warning")
                time.sleep(5)
                gc.collect()
    return results


# phase 7: depth evaluation

def _load_depth_estimator(depth_checkpoint: Path):
    if not depth_checkpoint.exists():
        return None
    os.environ["DEPTH_CHECKPOINT_PATH"] = str(depth_checkpoint)
    if hasattr(registry, "_depth_estimator"):
        registry._depth_estimator = None
    return registry.depth_estimator


def _eval_depth(
    test_set: List[str],
    embedded: Dict[str, dict],
    dino_results: Dict[str, dict],
    focal_length: Optional[float],
    no_depth: bool,
    depth_checkpoint: Path,
    monitor: Optional[MemoryMonitor] = None,
) -> Dict[str, dict]:
    blank = {"predicted_ft": None, "abs_error": None, "pct_error": None, "lat_depth": 0.0}
    results = {fname: dict(blank) for fname in test_set}

    if no_depth:
        return results

    estimator = _load_depth_estimator(depth_checkpoint)
    if estimator is None:
        print(f"  [info] depth checkpoint not found at {depth_checkpoint}; skipping")
        return results

    for i, fname in enumerate(test_set):
        data = embedded[fname]
        det = dino_results[fname]
        gt_dist = data["distance_ft"]
        if not det["detected"] or gt_dist <= 0:
            continue
        image = load_image(data["image_path"])
        t0 = time.perf_counter()
        depth_map = estimator.estimate(image, focal_length_px=focal_length)
        pred_ft = estimator.get_depth_at_bbox(depth_map, det["box"])
        lat_depth = time.perf_counter() - t0
        abs_err = abs(pred_ft - gt_dist)
        results[fname] = {
            "predicted_ft": pred_ft,
            "abs_error": abs_err,
            "pct_error": abs_err / gt_dist * 100.0,
            "lat_depth": lat_depth,
        }
        if monitor is not None and (i + 1) % 20 == 0:
            if monitor.suggest_throttle():
                monitor.log_memory_state(level="warning")
                time.sleep(5)
                gc.collect()
    return results


def _eval_fp_holdout_from_test(
    test_set: List[str],
    embedded: Dict[str, dict],
    database: List[Tuple[str, torch.Tensor]],
    head: ProjectionHead,
    thresholds: Dict[str, float],
    settings: Settings,
) -> List[dict]:
    """
    Build 60 balanced FP tests from retrieval test images.

    For each test image, remove its true label from the candidate DB and check
    whether any wrong label still matches. This mirrors "remove from DB, then query"
    without requiring extra images.
    """
    head.eval()
    labels = sorted({lbl for lbl, _ in database})
    results: List[dict] = []
    for fname in test_set:
        row = embedded[fname]
        true_label = row["label"]
        filtered_db = [(lbl, emb) for (lbl, emb) in database if lbl != true_label]
        projected_db = [(lbl, head.project(emb)) for (lbl, emb) in filtered_db]
        emb = row["embedding"]
        is_document = bool(row.get("is_document", False))
        bl_t = _threshold_for_label(true_label, is_document, thresholds, mode="baseline")
        pe_t = _threshold_for_label(true_label, is_document, thresholds, mode="personalized")
        bl_margin_t = _margin_for_label(true_label, is_document, settings)
        pe_margin_t = _margin_for_label(true_label, is_document, settings)
        bl_lbl, bl_sim, bl_margin = _match_with_runtime_gates(
            emb,
            filtered_db,
            is_document,
            thresholds,
            settings,
            mode="baseline",
        )
        pe_lbl, pe_sim, pe_margin = _match_with_runtime_gates(
            head.project(emb),
            projected_db,
            is_document,
            thresholds,
            settings,
            mode="personalized",
        )
        results.append(
            {
                "image": fname,
                "source": "test_holdout",
                "true_label": true_label,
                "excluded_labels": [true_label] if true_label in labels else [],
                "baseline_fp": int(bl_lbl is not None),
                "baseline_match": bl_lbl or "",
                "baseline_sim": round(bl_sim, 6),
                "personalized_fp": int(pe_lbl is not None),
                "personalized_match": pe_lbl or "",
                "personalized_sim": round(pe_sim, 6),
                "distance_bucket": row.get("distance_bucket", ""),
                "lighting_bucket": row.get("lighting_bucket", ""),
                "cleanliness_bucket": row.get("cleanliness_bucket", ""),
                "is_document": int(is_document),
                "baseline_threshold_used": round(bl_t, 6),
                "personalized_threshold_used": round(pe_t, 6),
                "baseline_margin_threshold_used": round(bl_margin_t, 6),
                "personalized_margin_threshold_used": round(pe_margin_t, 6),
                "baseline_similarity_margin": round(bl_margin, 6),
                "personalized_similarity_margin": round(pe_margin, 6),
            }
        )
    return results


def _eval_fp_expanded_from_test(
    test_set: List[str],
    embedded: Dict[str, dict],
    head: ProjectionHead,
    thresholds: Dict[str, float],
    settings: Settings,
) -> List[dict]:
    """
    Expanded FP benchmark from retrieval test images.

    For each test image, evaluate against every embedded image from labels other than
    the true label. This gives a dense negative pool (roughly ~108 negatives/image
    with the 120-image benchmark set) and stress-tests false-positive behavior.
    """
    head.eval()
    all_items: List[Tuple[str, str, torch.Tensor]] = [
        (name, row["label"], row["embedding"]) for name, row in embedded.items()
    ]
    projected = {name: head.project(row["embedding"]) for name, row in embedded.items()}
    results: List[dict] = []
    for fname in test_set:
        row = embedded[fname]
        true_label = row["label"]
        q = row["embedding"]
        qh = projected[fname]
        is_document = bool(row.get("is_document", False))
        bl_t = _threshold_for_label(true_label, is_document, thresholds, mode="baseline")
        pe_t = _threshold_for_label(true_label, is_document, thresholds, mode="personalized")
        bl_margin_t = _margin_for_label(true_label, is_document, settings)
        pe_margin_t = _margin_for_label(true_label, is_document, settings)

        best_bl_sim = -1.0
        second_bl_sim = -1.0
        best_pe_sim = -1.0
        second_pe_sim = -1.0
        best_bl_match = ""
        best_pe_match = ""
        neg_count = 0
        for cand_name, cand_label, cand_emb in all_items:
            if cand_label == true_label:
                continue
            neg_count += 1
            bl_sim = float(cosine_similarity(q, cand_emb).item())
            if bl_sim > best_bl_sim:
                second_bl_sim = best_bl_sim
                best_bl_sim = bl_sim
                best_bl_match = cand_name
            elif bl_sim > second_bl_sim:
                second_bl_sim = bl_sim
            pe_sim = float(cosine_similarity(qh, projected[cand_name]).item())
            if pe_sim > best_pe_sim:
                second_pe_sim = best_pe_sim
                best_pe_sim = pe_sim
                best_pe_match = cand_name
            elif pe_sim > second_pe_sim:
                second_pe_sim = pe_sim

        baseline_margin = max(0.0, best_bl_sim - second_bl_sim) if neg_count > 1 else max(best_bl_sim, 0.0)
        personalized_margin = max(0.0, best_pe_sim - second_pe_sim) if neg_count > 1 else max(best_pe_sim, 0.0)
        baseline_fp = int(neg_count > 0 and best_bl_sim >= bl_t and baseline_margin >= bl_margin_t)
        personalized_fp = int(neg_count > 0 and best_pe_sim >= pe_t and personalized_margin >= pe_margin_t)
        results.append(
            {
                "image": fname,
                "source": "expanded_all_other_labels",
                "true_label": true_label,
                "excluded_labels": [true_label],
                "negative_pool_size": neg_count,
                "baseline_fp": baseline_fp,
                "baseline_match": best_bl_match if baseline_fp else "",
                "baseline_sim": round(max(best_bl_sim, 0.0), 6),
                "personalized_fp": personalized_fp,
                "personalized_match": best_pe_match if personalized_fp else "",
                "personalized_sim": round(max(best_pe_sim, 0.0), 6),
                "distance_bucket": row.get("distance_bucket", ""),
                "lighting_bucket": row.get("lighting_bucket", ""),
                "cleanliness_bucket": row.get("cleanliness_bucket", ""),
                "is_document": int(is_document),
                "baseline_threshold_used": round(bl_t, 6),
                "personalized_threshold_used": round(pe_t, 6),
                "baseline_margin_threshold_used": round(bl_margin_t, 6),
                "personalized_margin_threshold_used": round(pe_margin_t, 6),
                "baseline_similarity_margin": round(baseline_margin, 6),
                "personalized_similarity_margin": round(personalized_margin, 6),
            }
        )
    return results


# output

_CSV_FIELDS = [
    "image", "label", "ground_truth_distance_ft",
    "distance_bucket", "lighting_bucket", "cleanliness_bucket", "condition_bucket", "is_document",
    "baseline_similarity", "personalized_similarity", "similarity_gap",
    "baseline_threshold_used", "personalized_threshold_used",
    "baseline_correct", "personalized_correct",
    "dino_detected", "dino_confidence", "dino_second_pass_prompt",
    "predicted_distance_ft", "depth_absolute_error", "depth_percentage_error",
    "lat_embed_img_s", "lat_ocr_s", "lat_embed_txt_s",
    "lat_retrieve_bl_s", "lat_retrieve_pe_s", "lat_detect_s", "lat_depth_s",
    "lat_pipeline_prepare_s", "lat_pipeline_detect_s", "lat_pipeline_embed_s",
    "lat_pipeline_ocr_s", "lat_pipeline_match_s", "lat_pipeline_dedup_s",
    "lat_pipeline_depth_s", "lat_pipeline_db_s",
    "darkness_level", "is_dark", "blur_score", "is_blurry",
    "text_likelihood", "should_skip_ocr",
    "holdout_baseline_fp", "holdout_baseline_match", "holdout_baseline_sim",
    "holdout_personalized_fp", "holdout_personalized_match", "holdout_personalized_sim",
    "expanded_negative_pool_size",
    "expanded_baseline_fp", "expanded_baseline_match", "expanded_baseline_sim",
    "expanded_personalized_fp", "expanded_personalized_match", "expanded_personalized_sim",
]


def _merge_rows(
    retrieval: List[dict],
    dino: Dict[str, dict],
    depth: Dict[str, dict],
    fp_holdout: Optional[List[dict]] = None,
    fp_expanded: Optional[List[dict]] = None,
) -> List[dict]:
    holdout_by_image = {r.get("image", ""): r for r in (fp_holdout or [])}
    expanded_by_image = {r.get("image", ""): r for r in (fp_expanded or [])}
    rows = []
    for r in retrieval:
        fname = r["image"]
        d = dino.get(fname, {"detected": 0, "confidence": 0.0, "lat_detect": 0.0, "second_pass_prompt": None})
        dep = depth.get(fname, {"predicted_ft": None, "abs_error": None,
                                 "pct_error": None, "lat_depth": 0.0})
        hold = holdout_by_image.get(fname, {})
        expanded = expanded_by_image.get(fname, {})
        rows.append({
            "image": fname,
            "label": r["label"],
            "ground_truth_distance_ft": r["distance_ft"],
            "distance_bucket": r.get("distance_bucket", ""),
            "lighting_bucket": r.get("lighting_bucket", ""),
            "cleanliness_bucket": r.get("cleanliness_bucket", ""),
            "condition_bucket": r.get("condition_bucket", ""),
            "is_document": int(r.get("is_document", 0)),
            "baseline_similarity": round(r["baseline_similarity"], 6),
            "personalized_similarity": round(r["personalized_similarity"], 6),
            "similarity_gap": round(r["similarity_gap"], 6),
            "baseline_threshold_used": round(float(r.get("baseline_threshold_used", 0.0)), 6),
            "personalized_threshold_used": round(float(r.get("personalized_threshold_used", 0.0)), 6),
            "baseline_correct": r["baseline_correct"],
            "personalized_correct": r["personalized_correct"],
            "dino_detected": d["detected"],
            "dino_confidence": round(d["confidence"], 6),
            "dino_second_pass_prompt": d.get("second_pass_prompt") or "",
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
            # Derived stage timing fields aligned with live pipeline perf logs.
            "lat_pipeline_prepare_s": 0.0,
            "lat_pipeline_detect_s": round(d["lat_detect"], 4),
            "lat_pipeline_embed_s": round(r["lat_embed_img"] + r["lat_embed_txt"], 4),
            "lat_pipeline_ocr_s": round(r["lat_ocr"], 4),
            "lat_pipeline_match_s": round(r["lat_retrieve_pe"], 6),
            "lat_pipeline_dedup_s": 0.0,
            "lat_pipeline_depth_s": round(dep["lat_depth"], 4),
            "lat_pipeline_db_s": 0.0,
            "darkness_level": r["darkness_level"],
            "is_dark": r["is_dark"],
            "blur_score": r["blur_score"],
            "is_blurry": r["is_blurry"],
            "text_likelihood": r["text_likelihood"],
            "should_skip_ocr": r["should_skip_ocr"],
            "holdout_baseline_fp": int(hold.get("baseline_fp", 0)),
            "holdout_baseline_match": hold.get("baseline_match", "") or "",
            "holdout_baseline_sim": (
                round(float(hold.get("baseline_sim", 0.0)), 6)
                if hold else ""
            ),
            "holdout_personalized_fp": int(hold.get("personalized_fp", 0)),
            "holdout_personalized_match": hold.get("personalized_match", "") or "",
            "holdout_personalized_sim": (
                round(float(hold.get("personalized_sim", 0.0)), 6)
                if hold else ""
            ),
            "expanded_negative_pool_size": int(expanded.get("negative_pool_size", 0)),
            "expanded_baseline_fp": int(expanded.get("baseline_fp", 0)),
            "expanded_baseline_match": expanded.get("baseline_match", "") or "",
            "expanded_baseline_sim": (
                round(float(expanded.get("baseline_sim", 0.0)), 6)
                if expanded else ""
            ),
            "expanded_personalized_fp": int(expanded.get("personalized_fp", 0)),
            "expanded_personalized_match": expanded.get("personalized_match", "") or "",
            "expanded_personalized_sim": (
                round(float(expanded.get("personalized_sim", 0.0)), 6)
                if expanded else ""
            ),
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
    args: argparse.Namespace,
    final_loss: float,
    thresholds: Dict[str, float],
    db_set: List[str],
    train_set: List[str],
    settings: Settings,
    split_integrity: dict,
    benchmark_guard: dict,
    fp_holdout_results: List[dict],
    fp_expanded_results: List[dict],
    threshold_sweep: List[dict],
    threshold_tuning: dict,
) -> None:
    _BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _BENCHMARKS_DIR / "results.csv"
    json_path = _BENCHMARKS_DIR / "results.json"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "schema_version": _RESULTS_SCHEMA_VERSION,
        "split_strategy": "fixed_60_60_label_balanced_seeded",
        "seed": split_integrity.get("seed"),
        "train_per_label": split_integrity.get("train_per_label"),
        "epochs": args.epochs,
        "lr": args.lr,
        "similarity_threshold": float(round(thresholds["personalized"], 3)),
        "similarity_thresholds": {
            "baseline": float(round(thresholds["baseline"], 3)),
            "personalized": float(round(thresholds["personalized"], 3)),
            "document": float(round(thresholds["document"], 3)),
        },
        "threshold_overrides": {
            "similarity_threshold": args.similarity_threshold,
            "baseline_threshold": args.baseline_threshold,
            "personalized_threshold": args.personalized_threshold,
            "document_threshold": args.document_threshold,
        },
        "darkness_threshold": settings.darkness_threshold,
        "ocr_text_likelihood_threshold": settings.ocr_text_likelihood_threshold,
        "blur_threshold": BLUR_THRESHOLD,
        "detection_quality_low_max": settings.detection_quality_low_max,
        "detection_quality_high_min": settings.detection_quality_high_min,
        "n_db": len(db_set),
        "n_triplet_train": len(train_set),
        "n_test": len(rows),
        "n_fp_holdout": len(fp_holdout_results),
        "n_fp_expanded": len(fp_expanded_results),
        "final_triplet_loss": round(final_loss, 6),
        "split_integrity": split_integrity,
        "benchmark_guard": benchmark_guard,
        "threshold_sweep": threshold_sweep,
        "threshold_tuning": threshold_tuning,
        "pipeline_stage_latency_mode": "derived_from_component_phases",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(json_path, "w") as f:
        json.dump(
            {
                "metadata": metadata,
                "results": rows,
                "fp_holdout": fp_holdout_results,
                "fp_expanded": fp_expanded_results,
            },
            f,
            indent=2,
        )

    _archive_results(
        csv_path=csv_path,
        json_path=json_path,
        metadata=metadata,
        rows=rows,
        fp_holdout_results=fp_holdout_results,
        fp_expanded_results=fp_expanded_results,
        benchmark_guard=benchmark_guard,
        threshold_tuning=threshold_tuning,
    )

    print(f"\nResults written to: {csv_path}, {json_path}")


def _archive_results(
    csv_path: Path,
    json_path: Path,
    metadata: dict,
    rows: List[dict],
    fp_holdout_results: List[dict],
    fp_expanded_results: List[dict],
    benchmark_guard: dict,
    threshold_tuning: dict,
) -> None:
    archive_root = _BENCHMARKS_DIR / "baselines" / "full_benchmark"
    archive_root.mkdir(parents=True, exist_ok=True)
    run_id_base = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = run_id_base
    suffix = 1
    while (archive_root / run_id).exists():
        run_id = f"{run_id_base}_{suffix:02d}"
        suffix += 1
    run_dir = archive_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(csv_path, run_dir / "results.csv")
    shutil.copy2(json_path, run_dir / "results.json")

    n = max(len(rows), 1)
    bl_acc = sum(int(r.get("baseline_correct", 0)) for r in rows) / n
    pe_acc = sum(int(r.get("personalized_correct", 0)) for r in rows) / n

    hold_n = len(fp_holdout_results)
    hold_bl = (
        sum(int(r.get("baseline_fp", 0)) for r in fp_holdout_results) / max(hold_n, 1)
        if hold_n > 0 else None
    )
    hold_pe = (
        sum(int(r.get("personalized_fp", 0)) for r in fp_holdout_results) / max(hold_n, 1)
        if hold_n > 0 else None
    )

    exp_n = len(fp_expanded_results)
    exp_bl = (
        sum(int(r.get("baseline_fp", 0)) for r in fp_expanded_results) / max(exp_n, 1)
        if exp_n > 0 else None
    )
    exp_pe = (
        sum(int(r.get("personalized_fp", 0)) for r in fp_expanded_results) / max(exp_n, 1)
        if exp_n > 0 else None
    )

    if benchmark_guard.get("enabled"):
        good_run = bool(benchmark_guard.get("passed"))
    else:
        target_fp = float((threshold_tuning or {}).get("target_holdout_fp", 0.40))
        good_run = pe_acc >= bl_acc and (hold_pe is None or hold_pe <= target_fp)
    quality_label = "good" if good_run else "review"

    summary = {
        "run_id": run_id,
        "timestamp": metadata.get("timestamp"),
        "quality_label": quality_label,
        "good_run": bool(good_run),
        "selection_reason": (threshold_tuning or {}).get("selection_reason", ""),
        "thresholds": metadata.get("similarity_thresholds", {}),
        "accuracy_baseline_pct": round(bl_acc * 100.0, 3),
        "accuracy_personalized_pct": round(pe_acc * 100.0, 3),
        "fp_holdout_baseline_pct": round(hold_bl * 100.0, 3) if hold_bl is not None else None,
        "fp_holdout_personalized_pct": round(hold_pe * 100.0, 3) if hold_pe is not None else None,
        "fp_expanded_baseline_pct": round(exp_bl * 100.0, 3) if exp_bl is not None else None,
        "fp_expanded_personalized_pct": round(exp_pe * 100.0, 3) if exp_pe is not None else None,
        "benchmark_guard_passed": bool((benchmark_guard or {}).get("passed", False)),
        "benchmark_guard_enabled": bool((benchmark_guard or {}).get("enabled", False)),
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    index_path = archive_root / "run_index.csv"
    index_fields = [
        "run_id",
        "timestamp",
        "quality_label",
        "good_run",
        "selection_reason",
        "threshold_baseline",
        "threshold_personalized",
        "threshold_document",
        "accuracy_baseline_pct",
        "accuracy_personalized_pct",
        "fp_holdout_baseline_pct",
        "fp_holdout_personalized_pct",
        "fp_expanded_baseline_pct",
        "fp_expanded_personalized_pct",
        "benchmark_guard_enabled",
        "benchmark_guard_passed",
    ]
    row = {
        "run_id": run_id,
        "timestamp": metadata.get("timestamp", ""),
        "quality_label": quality_label,
        "good_run": int(bool(good_run)),
        "selection_reason": (threshold_tuning or {}).get("selection_reason", ""),
        "threshold_baseline": metadata.get("similarity_thresholds", {}).get("baseline", ""),
        "threshold_personalized": metadata.get("similarity_thresholds", {}).get("personalized", ""),
        "threshold_document": metadata.get("similarity_thresholds", {}).get("document", ""),
        "accuracy_baseline_pct": round(bl_acc * 100.0, 3),
        "accuracy_personalized_pct": round(pe_acc * 100.0, 3),
        "fp_holdout_baseline_pct": round(hold_bl * 100.0, 3) if hold_bl is not None else "",
        "fp_holdout_personalized_pct": round(hold_pe * 100.0, 3) if hold_pe is not None else "",
        "fp_expanded_baseline_pct": round(exp_bl * 100.0, 3) if exp_bl is not None else "",
        "fp_expanded_personalized_pct": round(exp_pe * 100.0, 3) if exp_pe is not None else "",
        "benchmark_guard_enabled": int(bool((benchmark_guard or {}).get("enabled", False))),
        "benchmark_guard_passed": int(bool((benchmark_guard or {}).get("passed", False))),
    }
    write_header = not index_path.exists()
    with open(index_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=index_fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _pct(value: float) -> float:
    return round(value * 100.0, 3)


def _regression_guard(
    rows: List[dict],
    fp_holdout_results: List[dict],
    fp_expanded_results: List[dict],
    baseline_path: Optional[Path],
    max_accuracy_drop_pp: float,
    max_fp_increase_pp: float,
    max_fn_increase_pp: float,
) -> dict:
    n = max(len(rows), 1)
    pe_acc = sum(int(r.get("personalized_correct", 0)) for r in rows) / n
    pe_fn = 1.0 - pe_acc
    holdout_n = len(fp_holdout_results)
    holdout_pe_fp = (
        sum(int(r.get("personalized_fp", 0)) for r in fp_holdout_results) / max(holdout_n, 1)
        if holdout_n > 0 else None
    )
    expanded_n = len(fp_expanded_results)
    expanded_pe_fp = (
        sum(int(r.get("personalized_fp", 0)) for r in fp_expanded_results) / max(expanded_n, 1)
        if expanded_n > 0 else None
    )
    pe_fp = expanded_pe_fp if expanded_pe_fp is not None else holdout_pe_fp

    summary = {
        "enabled": bool(baseline_path),
        "baseline_path": str(baseline_path) if baseline_path else "",
        "current": {
            "personalized_accuracy_pct": _pct(pe_acc),
            "personalized_fn_rate_pct": _pct(pe_fn),
            "personalized_fp_rate_pct": _pct(pe_fp) if pe_fp is not None else None,
            "personalized_fp_rate_holdout_pct": _pct(holdout_pe_fp) if holdout_pe_fp is not None else None,
            "personalized_fp_rate_expanded_pct": _pct(expanded_pe_fp) if expanded_pe_fp is not None else None,
        },
        "thresholds": {
            "max_accuracy_drop_pp": float(max_accuracy_drop_pp),
            "max_fp_increase_pp": float(max_fp_increase_pp),
            "max_fn_increase_pp": float(max_fn_increase_pp),
        },
        "baseline": None,
        "fp_gate_source": "expanded" if expanded_pe_fp is not None else ("holdout" if holdout_pe_fp is not None else "none"),
        "checks": [],
        "passed": True,
    }
    if not baseline_path:
        return summary
    if not baseline_path.exists():
        raise FileNotFoundError(f"regression baseline not found: {baseline_path}")

    with open(baseline_path, encoding="utf-8") as f:
        baseline = json.load(f)
    b_rows = baseline.get("results") or []
    b_neg_holdout = baseline.get("fp_holdout") or baseline.get("negatives") or []
    b_neg_expanded = baseline.get("fp_expanded") or []
    if not isinstance(b_rows, list) or not b_rows:
        raise ValueError(f"baseline results missing/invalid: {baseline_path}")

    bn = max(len(b_rows), 1)
    b_pe_acc = sum(int(r.get("personalized_correct", 0)) for r in b_rows) / bn
    b_pe_fn = 1.0 - b_pe_acc
    b_neg_holdout_n = len(b_neg_holdout)
    b_holdout_pe_fp = (
        sum(int(r.get("personalized_fp", 0)) for r in b_neg_holdout) / max(b_neg_holdout_n, 1)
        if b_neg_holdout_n > 0 else None
    )
    b_neg_expanded_n = len(b_neg_expanded)
    b_expanded_pe_fp = (
        sum(int(r.get("personalized_fp", 0)) for r in b_neg_expanded) / max(b_neg_expanded_n, 1)
        if b_neg_expanded_n > 0 else None
    )
    b_pe_fp = b_expanded_pe_fp if b_expanded_pe_fp is not None else b_holdout_pe_fp
    if expanded_pe_fp is not None and b_expanded_pe_fp is not None:
        fp_gate_source = "expanded"
        pe_fp = expanded_pe_fp
        b_pe_fp = b_expanded_pe_fp
    elif holdout_pe_fp is not None and b_holdout_pe_fp is not None:
        fp_gate_source = "holdout"
        pe_fp = holdout_pe_fp
        b_pe_fp = b_holdout_pe_fp
    else:
        fp_gate_source = "none"
        pe_fp = None
        b_pe_fp = None
    summary["fp_gate_source"] = fp_gate_source
    summary["current"]["personalized_fp_rate_pct"] = _pct(pe_fp) if pe_fp is not None else None
    summary["baseline"] = {
        "personalized_accuracy_pct": _pct(b_pe_acc),
        "personalized_fn_rate_pct": _pct(b_pe_fn),
        "personalized_fp_rate_pct": _pct(b_pe_fp) if b_pe_fp is not None else None,
        "personalized_fp_rate_holdout_pct": _pct(b_holdout_pe_fp) if b_holdout_pe_fp is not None else None,
        "personalized_fp_rate_expanded_pct": _pct(b_expanded_pe_fp) if b_expanded_pe_fp is not None else None,
        "n_test": len(b_rows),
        "n_fp_holdout": b_neg_holdout_n,
        "n_fp_expanded": b_neg_expanded_n,
    }

    acc_drop_pp = _pct(b_pe_acc - pe_acc)
    fn_increase_pp = _pct(pe_fn - b_pe_fn)
    fp_increase_pp = _pct((pe_fp - b_pe_fp)) if (pe_fp is not None and b_pe_fp is not None) else None

    acc_ok = acc_drop_pp <= max_accuracy_drop_pp
    fn_ok = fn_increase_pp <= max_fn_increase_pp
    fp_ok = True if fp_increase_pp is None else fp_increase_pp <= max_fp_increase_pp
    summary["checks"] = [
        {
            "name": "accuracy_drop_pp",
            "actual": acc_drop_pp,
            "limit": float(max_accuracy_drop_pp),
            "ok": acc_ok,
        },
        {
            "name": "fn_increase_pp",
            "actual": fn_increase_pp,
            "limit": float(max_fn_increase_pp),
            "ok": fn_ok,
        },
        {
            "name": "fp_increase_pp",
            "actual": fp_increase_pp,
            "limit": float(max_fp_increase_pp),
            "ok": fp_ok,
            "skipped": fp_increase_pp is None,
        },
    ]
    summary["passed"] = bool(acc_ok and fn_ok and fp_ok)
    return summary


def _enforce_memory_guard(monitor: MemoryMonitor, phase_name: str) -> None:
    if monitor.is_oom_risk(threshold=0.80):
        monitor.log_memory_state(level="critical")
        killed = monitor.cleanup_zombies(max_age_hours=1)
        if killed:
            print(f"  [memory] cleaned zombie pids before {phase_name}: {killed}")
        if monitor.is_oom_risk(threshold=0.85):
            raise RuntimeError(f"OOM risk too high before {phase_name}, aborting benchmark")
    elif monitor.suggest_throttle():
        monitor.log_memory_state(level="warning")


def _print_summary(
    retrieval: List[dict],
    dino: Dict[str, dict],
    depth: Dict[str, dict],
    final_loss: float,
    thresholds: Dict[str, float],
    no_depth: bool,
    fp_holdout_results: List[dict],
    fp_expanded_results: List[dict],
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

    print("\n=== Benchmark Summary ===")
    print(
        "Similarity thresholds: "
        f"baseline={thresholds['baseline']:.2f}, "
        f"personalized={thresholds['personalized']:.2f}, "
        f"document={thresholds['document']:.2f}"
    )
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
    if no_depth:
        print("                      skipped")
    elif not depth_evaluated:
        print(f"Evaluated on:         0 / {det_total} images")
    else:
        print(f"Evaluated on:         {len(depth_evaluated)} / {det_total} images")
        print(f"Mean abs. error:      {mean_abs:.1f} ft")
        print(f"Mean % error:         {mean_pct:.1f}%")
    print()
    if fp_holdout_results:
        n_hold = len(fp_holdout_results)
        bl_hold_fp = sum(r["baseline_fp"] for r in fp_holdout_results)
        pe_hold_fp = sum(r["personalized_fp"] for r in fp_holdout_results)
        hold_delta = pe_hold_fp - bl_hold_fp
        hold_sign = "+" if hold_delta > 0 else ""
        print("--- False Positives (60-case holdout from test split) ---")
        print(f"Baseline FP rate:     {bl_hold_fp} / {n_hold}  ({100*bl_hold_fp/max(n_hold,1):.1f}%)")
        print(f"Personalized FP rate: {pe_hold_fp} / {n_hold}  ({100*pe_hold_fp/max(n_hold,1):.1f}%)")
        print(f"Delta:                {hold_sign}{100*hold_delta/max(n_hold,1):.1f} pp")
        print()
    if fp_expanded_results:
        n_exp = len(fp_expanded_results)
        exp_pool_avg = (
            sum(int(r.get("negative_pool_size", 0)) for r in fp_expanded_results) / max(n_exp, 1)
        )
        bl_exp_fp = sum(int(r.get("baseline_fp", 0)) for r in fp_expanded_results)
        pe_exp_fp = sum(int(r.get("personalized_fp", 0)) for r in fp_expanded_results)
        exp_delta = pe_exp_fp - bl_exp_fp
        exp_sign = "+" if exp_delta > 0 else ""
        print("--- False Positives (expanded all-other-label negatives) ---")
        print(f"Avg negatives/query:  {exp_pool_avg:.1f}")
        print(f"Baseline FP rate:     {bl_exp_fp} / {n_exp}  ({100*bl_exp_fp/max(n_exp,1):.1f}%)")
        print(f"Personalized FP rate: {pe_exp_fp} / {n_exp}  ({100*pe_exp_fp/max(n_exp,1):.1f}%)")
        print(f"Delta:                {exp_sign}{100*exp_delta/max(n_exp,1):.1f} pp")
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

    # Pipeline stage latency from remember/scan logs (if available in retrieval rows)
    stage_phases = [
        ("pipeline_prepare", [r.get("lat_pipeline_prepare", 0.0) for r in retrieval if r.get("lat_pipeline_prepare", 0.0) > 0]),
        ("pipeline_detect",  [r.get("lat_pipeline_detect", 0.0) for r in retrieval if r.get("lat_pipeline_detect", 0.0) > 0]),
        ("pipeline_embed",   [r.get("lat_pipeline_embed", 0.0) for r in retrieval if r.get("lat_pipeline_embed", 0.0) > 0]),
        ("pipeline_ocr",     [r.get("lat_pipeline_ocr", 0.0) for r in retrieval if r.get("lat_pipeline_ocr", 0.0) > 0]),
        ("pipeline_match",   [r.get("lat_pipeline_match", 0.0) for r in retrieval if r.get("lat_pipeline_match", 0.0) > 0]),
        ("pipeline_dedup",   [r.get("lat_pipeline_dedup", 0.0) for r in retrieval if r.get("lat_pipeline_dedup", 0.0) > 0]),
        ("pipeline_depth",   [r.get("lat_pipeline_depth", 0.0) for r in retrieval if r.get("lat_pipeline_depth", 0.0) > 0]),
        ("pipeline_db",      [r.get("lat_pipeline_db", 0.0) for r in retrieval if r.get("lat_pipeline_db", 0.0) > 0]),
    ]
    for phase_name, vals in stage_phases:
        if not vals:
            continue
        mean, lo, hi, _ = _latency_stats(vals)
        print(f"  {phase_name:<14} mean {mean:.3f}s  [{lo:.3f}s - {hi:.3f}s]")


# main

def main() -> None:
    args = _parse_args()
    _set_reproducible(args.seed)
    settings = Settings()
    depth_checkpoint = _resolve_depth_checkpoint(args)
    sweep_min = _validate_similarity_threshold(float(args.threshold_sweep_min), "threshold sweep minimum")
    sweep_max = _validate_similarity_threshold(float(args.threshold_sweep_max), "threshold sweep maximum")
    if float(args.threshold_sweep_step) <= 0:
        raise ValueError(f"threshold sweep step must be > 0, got {args.threshold_sweep_step}")
    if args.similarity_threshold is not None:
        legacy_threshold = _validate_similarity_threshold(
            float(args.similarity_threshold),
            "manual similarity threshold",
        )
        thresholds: Dict[str, float] = {
            "baseline": legacy_threshold,
            "personalized": legacy_threshold,
            "document": legacy_threshold,
        }
    else:
        thresholds = {
            "baseline": _validate_similarity_threshold(
                float(settings.get_similarity_threshold_baseline()),
                "settings baseline similarity threshold",
            ),
            "personalized": _validate_similarity_threshold(
                float(settings.get_similarity_threshold_personalized()),
                "settings personalized similarity threshold",
            ),
            "document": _validate_similarity_threshold(
                float(settings.get_similarity_threshold_document()),
                "settings document similarity threshold",
            ),
        }
    monitor = MemoryMonitor()

    print(f"Dataset: {args.dataset}")
    print(f"Images:  {args.images}")
    print(f"Depth checkpoint: {depth_checkpoint}")
    print(f"Thresholds: baseline={thresholds['baseline']:.3f}, personalized={thresholds['personalized']:.3f}, "
          f"document={thresholds['document']:.3f}, darkness={settings.darkness_threshold:.1f}, "
          f"ocr_text_likelihood={settings.ocr_text_likelihood_threshold:.2f}, "
          f"blur={BLUR_THRESHOLD:.1f}")

    # Phase 1: Embed
    _enforce_memory_guard(monitor, "phase 1")
    print("\n[1/7] Loading and embedding images...")
    rows = _load_dataset(args.dataset)
    embedded = _embed_rows(rows, args.images, args.no_ocr, settings, monitor=monitor)
    if not embedded:
        print("No images found. Check --images path.", file=sys.stderr)
        sys.exit(1)
    for fname, row in embedded.items():
        buckets = _condition_buckets(
            image_name=fname,
            distance_ft=float(row.get("distance_ft", 0.0)),
            is_dark=bool(row.get("is_dark", False)),
        )
        row.update(buckets)
        row["is_document"] = _is_document_like(str(row.get("label", "")), str(row.get("dino_prompt", "")))

    # Phase 2: Split
    _enforce_memory_guard(monitor, "phase 2")
    print("\n[2/7] Splitting: fixed 60 train / 60 test (label-balanced, seeded)...")
    ds_sig = _dataset_signature(embedded)
    manifest_path = args.split_manifest
    used_manifest = False
    manifest_seed = args.seed
    manifest_train_per_label = args.train_per_label
    if manifest_path.exists() and not args.refresh_split_manifest:
        train_set, db_set, test_set, manifest_info = _load_split_manifest(manifest_path, embedded)
        manifest_sig = manifest_info.get("signature") or {}
        if manifest_sig and manifest_sig != ds_sig:
            raise ValueError(
                "split manifest signature mismatch; dataset changed. "
                "Re-run with --refresh-split-manifest."
            )
        manifest_seed = int(manifest_info.get("seed", manifest_seed))
        manifest_train_per_label = int(manifest_info.get("train_per_label", manifest_train_per_label))
        used_manifest = True
        print(f"  using split manifest: {manifest_path}")
    else:
        train_set, db_set, test_set = _split_fixed_60_60(
            embedded,
            train_per_label=args.train_per_label,
            seed=args.seed,
        )
        _write_split_manifest(
            manifest_path,
            ds_sig,
            args.seed,
            args.train_per_label,
            train_set,
            db_set,
            test_set,
        )
        print(f"  wrote split manifest: {manifest_path}")
    train_labels = sorted({embedded[f]["label"] for f in train_set})
    test_labels = sorted({embedded[f]["label"] for f in test_set})
    file_overlap = sorted(set(train_set).intersection(test_set))
    db_not_in_train = sorted(set(db_set).difference(train_set))
    db_in_test = sorted(set(db_set).intersection(test_set))
    if file_overlap:
        raise ValueError(
            "split leakage detected: train/test image overlap found "
            f"({len(file_overlap)} files)"
        )
    if db_not_in_train:
        raise ValueError(
            "split integrity error: DB includes images not present in train split "
            f"({len(db_not_in_train)} files)"
        )
    if db_in_test:
        raise ValueError(
            "split leakage detected: DB contains test images "
            f"({len(db_in_test)} files)"
        )
    split_integrity = {
        "strategy": "fixed_60_60_label_balanced_seeded",
        "manifest_path": str(manifest_path),
        "manifest_used": used_manifest,
        "seed": manifest_seed,
        "train_per_label": manifest_train_per_label,
        "dataset_signature": ds_sig,
        "train_count": len(train_set),
        "test_count": len(test_set),
        "train_labels": train_labels,
        "test_labels": test_labels,
        "label_overlap": sorted(set(train_labels).intersection(test_labels)),
        "db_labels": sorted({embedded[f]["label"] for f in db_set}),
        "train_test_file_overlap_count": len(file_overlap),
        "db_not_in_train_count": len(db_not_in_train),
        "db_in_test_count": len(db_in_test),
    }
    print(
        f"  db (teach): {len(db_set)}, triplet train: {len(train_set)}, "
        f"test (scan): {len(test_set)}"
    )
    print(
        f"  split integrity: train={len(train_set)} test={len(test_set)} "
        f"label_overlap={len(split_integrity['label_overlap'])}"
    )
    if len(train_set) != 60 or len(test_set) != 60:
        print("Split is not 60/60; check --train-per-label and dataset cardinality.", file=sys.stderr)
        sys.exit(1)
    if not test_set:
        print("No 3ft/6ft images found. Check dataset and image filenames.", file=sys.stderr)
        sys.exit(1)
    if not db_set:
        print("No 1ft_bright_clean images found. Check dataset and image filenames.", file=sys.stderr)
        sys.exit(1)

    # Phase 3: Database
    _enforce_memory_guard(monitor, "phase 3")
    print("\n[3/7] Building reference database from 1ft_bright_clean images...")
    database = _build_database(db_set, embedded)
    print(f"  {len(database)} entries")

    # Phase 4: Train
    _enforce_memory_guard(monitor, "phase 4")
    print(f"\n[4/7] Training projection head ({args.epochs} epochs, train split only)...")
    head_path = _BENCHMARKS_DIR / "projection_head_bench.pt"
    head, final_loss = _train_head(
        train_set,
        embedded,
        args.epochs,
        args.lr,
        head_path,
        augment_train=not args.no_train_augment,
    )
    print(f"  final loss: {final_loss:.4f}")
    threshold_sweep: List[dict] = []
    threshold_tuning: dict = {}
    similarity_stats = _build_similarity_stats(test_set, embedded, database, head)
    if (
        args.similarity_threshold is not None
        or args.baseline_threshold is not None
        or args.personalized_threshold is not None
        or args.document_threshold is not None
    ):
        if args.baseline_threshold is not None:
            thresholds["baseline"] = _validate_similarity_threshold(float(args.baseline_threshold), "manual baseline threshold")
        if args.personalized_threshold is not None:
            thresholds["personalized"] = _validate_similarity_threshold(
                float(args.personalized_threshold), "manual personalized threshold"
            )
        if args.document_threshold is not None:
            thresholds["document"] = _validate_similarity_threshold(float(args.document_threshold), "manual document threshold")
        threshold_tuning = {
            "selection_reason": "manual_override",
            "selected_thresholds": {
                "baseline": float(round(thresholds["baseline"], 3)),
                "personalized": float(round(thresholds["personalized"], 3)),
                "document": float(round(thresholds["document"], 3)),
            },
            "manual_override": True,
        }
        candidates = [
            _validate_similarity_threshold(float(round(v, 3)), "threshold sweep candidate")
            for v in np.arange(sweep_min, sweep_max + 1e-9, args.threshold_sweep_step)
        ]
        threshold_sweep = [
            _metrics_at_threshold(
                stats=similarity_stats,
                baseline_threshold=t,
                personalized_threshold=t,
                document_threshold=t,
            )
            for t in candidates
        ]
        print(
            "  using manual threshold override: "
            f"baseline={thresholds['baseline']:.3f}, "
            f"personalized={thresholds['personalized']:.3f}, "
            f"document={thresholds['document']:.3f}"
        )
    else:
        thresholds, threshold_sweep, threshold_tuning = _tune_threshold_with_holdout(
            stats=similarity_stats,
            floor=sweep_min,
            ceiling=sweep_max,
            step=float(args.threshold_sweep_step),
            target_holdout_fp=float(args.target_holdout_fp),
            min_personalized_accuracy=float(args.min_personalized_accuracy),
        )
        print(
            "  tuned thresholds from sweep: "
            f"baseline={thresholds['baseline']:.3f}, "
            f"personalized={thresholds['personalized']:.3f}, "
            f"document={thresholds['document']:.3f} "
            f"(reason={threshold_tuning.get('selection_reason', 'unknown')})"
        )

    # Phase 5: Retrieval
    _enforce_memory_guard(monitor, "phase 5")
    print("\n[5/7] Evaluating retrieval (3ft/6ft test set against 1ft_bright_clean DB)...")
    retrieval_results = _eval_retrieval(test_set, embedded, database, head, thresholds, settings, monitor=monitor)

    # Phase 6: Detection
    _enforce_memory_guard(monitor, "phase 6")
    print("\n[6/7] Evaluating GroundingDINO detection (with full fallback chain)...")
    dino_results = _eval_detection(test_set, embedded, settings, monitor=monitor)

    # Phase 7: Depth
    _enforce_memory_guard(monitor, "phase 7")
    print("\n[7/7] Evaluating depth estimation...")
    depth_results = _eval_depth(
        test_set,
        embedded,
        dino_results,
        args.focal_length,
        args.no_depth,
        depth_checkpoint,
        monitor=monitor,
    )

    fp_holdout_results: List[dict] = []
    if not args.no_fp_holdout_tests:
        print("\n[fp] Evaluating 60-case holdout FP tests from detection split...")
        fp_holdout_results = _eval_fp_holdout_from_test(
            test_set, embedded, database, head, thresholds, settings
        )
        print(f"  {len(fp_holdout_results)} holdout FP cases evaluated")
    fp_expanded_results: List[dict] = []
    if not args.no_fp_expanded_tests:
        print("\n[fp] Evaluating expanded FP tests against all other-label samples...")
        fp_expanded_results = _eval_fp_expanded_from_test(
            test_set, embedded, head, thresholds, settings
        )
        if fp_expanded_results:
            avg_neg = sum(int(r.get("negative_pool_size", 0)) for r in fp_expanded_results) / len(fp_expanded_results)
            print(f"  {len(fp_expanded_results)} expanded FP cases evaluated (avg negatives/query: {avg_neg:.1f})")

    # Output
    merged = _merge_rows(
        retrieval_results,
        dino_results,
        depth_results,
        fp_holdout=fp_holdout_results,
        fp_expanded=fp_expanded_results,
    )
    benchmark_guard = _regression_guard(
        rows=merged,
        fp_holdout_results=fp_holdout_results,
        fp_expanded_results=fp_expanded_results,
        baseline_path=args.regression_baseline,
        max_accuracy_drop_pp=args.max_accuracy_drop_pp,
        max_fp_increase_pp=args.max_fp_increase_pp,
        max_fn_increase_pp=args.max_fn_increase_pp,
    )
    _print_summary(retrieval_results, dino_results, depth_results,
                   final_loss, thresholds, args.no_depth, fp_holdout_results, fp_expanded_results)
    _write_output(
        merged,
        args,
        final_loss,
        thresholds,
        db_set,
        train_set,
        settings,
        split_integrity=split_integrity,
        benchmark_guard=benchmark_guard,
        fp_holdout_results=fp_holdout_results,
        fp_expanded_results=fp_expanded_results,
        threshold_sweep=threshold_sweep,
        threshold_tuning=threshold_tuning,
    )
    if benchmark_guard["enabled"]:
        print("\n[guard] Benchmark regression checks:")
        for check in benchmark_guard["checks"]:
            if check.get("skipped"):
                print(f"  - {check['name']}: skipped (insufficient holdout baseline)")
                continue
            status = "PASS" if check["ok"] else "FAIL"
            print(
                f"  - {check['name']}: {status} "
                f"(actual={check['actual']} limit={check['limit']})"
            )
        if not benchmark_guard["passed"]:
            print("[guard] Regression gate failed.", file=sys.stderr)
            sys.exit(2)


if __name__ == "__main__":
    main()
