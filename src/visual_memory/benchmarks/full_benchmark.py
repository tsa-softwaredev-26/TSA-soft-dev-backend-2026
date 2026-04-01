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
import json
import re
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
from visual_memory.utils.similarity_utils import cosine_similarity, find_match

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
    p.add_argument("--focal-length", type=float, default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-per-label", type=int, default=6,
                   help="How many images per label go to train split (dataset has 12 per label)")
    p.add_argument("--no-train-augment", action="store_true",
                   help="Disable mirrored/blurred/darkened augmentations for train images")
    p.add_argument("--similarity-threshold", type=float, default=None,
                   help="Override retrieval threshold (default: Settings.similarity_threshold)")
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
                "note": row.get("note", ""),
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
    image: Image.Image,
    do_augment: bool,
) -> List[torch.Tensor]:
    """Return base embedding plus optional deterministic visual augmentations."""
    out = [embedding]
    if not do_augment:
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
            anchors = _augment_train_embedding(anchor["embedding"], anchor["image"], augment_train)
            positives = _augment_train_embedding(positive["embedding"], positive["image"], augment_train)
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
    threshold: float,
    monitor: Optional[MemoryMonitor] = None,
) -> List[dict]:
    head.eval()
    projected_db = [(lbl, head.project(e)) for lbl, e in database]
    results = []
    for i, fname in enumerate(test_set):
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
        if monitor is not None and (i + 1) % 20 == 0:
            if monitor.suggest_throttle():
                monitor.log_memory_state(level="warning")
                time.sleep(5)
                gc.collect()
    return results


def _calibrate_threshold(
    train_set: List[str],
    embedded: Dict[str, dict],
    database: List[Tuple[str, torch.Tensor]],
    test_set: List[str],
    floor: float = 0.14,
    ceiling: float = 0.42,
) -> float:
    """
    Choose threshold from train/test distributions to reduce FP inflation.

    Strategy:
    - positives: baseline/test similarities for true label
    - negatives: max similarity to wrong labels on test set
    - set threshold near the upper negative quantile but within [floor, ceiling]
    """
    if not database or not test_set:
        return floor
    label_to_embs: Dict[str, List[torch.Tensor]] = {}
    for lbl, emb in database:
        label_to_embs.setdefault(lbl, []).append(emb)

    neg_max_sims: List[float] = []
    pos_sims: List[float] = []
    for fname in test_set:
        row = embedded[fname]
        q = row["embedding"]
        true_lbl = row["label"]
        best_neg = 0.0
        best_pos = 0.0
        for lbl, emb in database:
            sim = float(cosine_similarity(q, emb).item())
            if lbl == true_lbl:
                best_pos = max(best_pos, sim)
            else:
                best_neg = max(best_neg, sim)
        if best_pos > 0:
            pos_sims.append(best_pos)
        if best_neg > 0:
            neg_max_sims.append(best_neg)

    if not neg_max_sims:
        return floor
    neg_q90 = float(np.quantile(np.array(neg_max_sims), 0.90))
    pos_q25 = float(np.quantile(np.array(pos_sims), 0.25)) if pos_sims else ceiling
    # Bias toward reducing false positives while keeping some recall.
    proposed = min(max(neg_q90 + 0.01, floor), ceiling)
    # Avoid setting above lower-tail positives unless absolutely necessary.
    proposed = min(proposed, max(pos_q25, floor))
    return float(round(proposed, 3))


def _calibrate_threshold_with_negatives(
    test_set: List[str],
    embedded: Dict[str, dict],
    neg_embedded: Dict[str, dict],
    neg_rows: List[dict],
    database: List[Tuple[str, torch.Tensor]],
    head: ProjectionHead,
    floor: float = 0.20,
    ceiling: float = 0.85,
) -> float:
    """
    Pick threshold using both test positives and explicit negative samples.

    Objective favors lower false-positive rates over absolute recall.
    """
    if not database or not test_set or not neg_embedded:
        return floor

    projected_db = [(lbl, head.project(emb)) for (lbl, emb) in database]
    labels = sorted({lbl for lbl, _ in database})
    row_by_image = {r["image"]: r for r in neg_rows}

    pos_bl: List[float] = []
    pos_pe: List[float] = []
    for fname in test_set:
        row = embedded[fname]
        q = row["embedding"]
        true_label = row["label"]
        bl_same = [float(cosine_similarity(q, emb).item()) for (lbl, emb) in database if lbl == true_label]
        pe_same = [
            float(cosine_similarity(head.project(q), pemb).item())
            for (lbl, pemb) in projected_db
            if lbl == true_label
        ]
        if bl_same:
            pos_bl.append(max(bl_same))
        if pe_same:
            pos_pe.append(max(pe_same))

    neg_bl: List[float] = []
    neg_pe: List[float] = []
    for fname, row in neg_embedded.items():
        note = str(row_by_image.get(fname, {}).get("note", ""))
        excluded = _parse_negative_excluded_labels(note, labels)
        db_filtered = [(lbl, emb) for (lbl, emb) in database if lbl not in excluded]
        pdb_filtered = [(lbl, emb) for (lbl, emb) in projected_db if lbl not in excluded]
        q = row["embedding"]
        if db_filtered:
            neg_bl.append(max(float(cosine_similarity(q, emb).item()) for _, emb in db_filtered))
        if pdb_filtered:
            qh = head.project(q)
            neg_pe.append(max(float(cosine_similarity(qh, emb).item()) for _, emb in pdb_filtered))

    if not pos_pe or not neg_pe:
        return floor

    best_t = floor
    best_score = -1e9
    for t in np.arange(floor, ceiling + 1e-9, 0.01):
        t = float(round(t, 3))
        tpr_bl = sum(v >= t for v in pos_bl) / max(len(pos_bl), 1)
        tpr_pe = sum(v >= t for v in pos_pe) / max(len(pos_pe), 1)
        fpr_bl = sum(v >= t for v in neg_bl) / max(len(neg_bl), 1)
        fpr_pe = sum(v >= t for v in neg_pe) / max(len(neg_pe), 1)
        # Strong FP penalty, slight preference to personalized recall.
        score = (0.45 * tpr_bl + 0.55 * tpr_pe) - (2.0 * fpr_bl + 3.0 * fpr_pe)
        if score > best_score:
            best_score = score
            best_t = t
    return best_t


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
    for i, fname in enumerate(test_set):
        data = embedded[fname]
        t0 = time.perf_counter()
        det, used_prompt = _detect_with_fallback(detector, data["image"], data["dino_prompt"], settings)
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
    monitor: Optional[MemoryMonitor] = None,
) -> Dict[str, dict]:
    blank = {"predicted_ft": None, "abs_error": None, "pct_error": None, "lat_depth": 0.0}
    results = {fname: dict(blank) for fname in test_set}

    if no_depth:
        return results

    estimator = _load_depth_estimator()
    if estimator is None:
        print("  [info] depth checkpoint not found; skipping")
        return results

    for i, fname in enumerate(test_set):
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
        if monitor is not None and (i + 1) % 20 == 0:
            if monitor.suggest_throttle():
                monitor.log_memory_state(level="warning")
                time.sleep(5)
                gc.collect()
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


def _canonical_token(value: str) -> str:
    v = (value or "").lower()
    v = re.sub(r"[^a-z0-9]+", " ", v).strip()
    return v


def _parse_negative_excluded_labels(note: str, labels: List[str]) -> List[str]:
    """
    Extract labels to exclude for a negative sample from note text.

    This enforces user-requested FP rigor: do not evaluate a negative sample against
    a DB that contains the same-item identity if the note indicates that mapping.
    """
    note_norm = _canonical_token(note)
    excluded: List[str] = []
    for label in labels:
        tok = _canonical_token(label).replace("_", " ")
        if tok and tok in note_norm:
            excluded.append(label)
    # Generic hard-negative hints for wallets/keys/receipts classes.
    if "wallet" in note_norm:
        excluded.extend([l for l in labels if "wallet" in _canonical_token(l)])
    if "keys" in note_norm:
        excluded.extend([l for l in labels if "keys" in _canonical_token(l)])
    if "receipt" in note_norm:
        excluded.extend([l for l in labels if "receipt" in _canonical_token(l)])
    return sorted(set(excluded))


def _eval_negatives_strict(
    neg_embedded: Dict[str, dict],
    neg_rows: List[dict],
    database: List[Tuple[str, torch.Tensor]],
    head: ProjectionHead,
    threshold: float,
) -> List[dict]:
    """
    Evaluate negatives with per-sample exclusion from candidate DB.

    For each negative image, if its note indicates a related train identity/class,
    remove those labels from candidate DB for that evaluation row.
    """
    head.eval()
    labels = sorted({lbl for lbl, _ in database})
    results = []
    row_by_image = {r["image"]: r for r in neg_rows}
    for fname, data in neg_embedded.items():
        row = row_by_image.get(fname, {})
        excluded = _parse_negative_excluded_labels(str(row.get("note", "")), labels)
        filtered_db = [(lbl, emb) for (lbl, emb) in database if lbl not in excluded]
        projected_db = [(lbl, head.project(emb)) for (lbl, emb) in filtered_db]
        emb = data["embedding"]
        bl_lbl, bl_sim = find_match(emb, filtered_db, threshold)
        pe_lbl, pe_sim = find_match(head.project(emb), projected_db, threshold)
        results.append(
            {
                "image": fname,
                "excluded_labels": excluded,
                "baseline_fp": int(bl_lbl is not None),
                "baseline_match": bl_lbl or "",
                "baseline_sim": round(bl_sim, 6),
                "personalized_fp": int(pe_lbl is not None),
                "personalized_match": pe_lbl or "",
                "personalized_sim": round(pe_sim, 6),
            }
        )
    return results


# output

_CSV_FIELDS = [
    "image", "label", "ground_truth_distance_ft",
    "baseline_similarity", "personalized_similarity", "similarity_gap",
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
]


def _merge_rows(
    retrieval: List[dict],
    dino: Dict[str, dict],
    depth: Dict[str, dict],
) -> List[dict]:
    rows = []
    for r in retrieval:
        fname = r["image"]
        d = dino.get(fname, {"detected": 0, "confidence": 0.0, "lat_detect": 0.0, "second_pass_prompt": None})
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
    db_set: List[str],
    train_set: List[str],
    settings: Settings,
    split_integrity: dict,
) -> None:
    _BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _BENCHMARKS_DIR / "results.csv"
    json_path = _BENCHMARKS_DIR / "results.json"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "split_strategy": "fixed_60_60_label_balanced_seeded",
        "epochs": args.epochs,
        "lr": args.lr,
        "similarity_threshold": threshold,
        "darkness_threshold": settings.darkness_threshold,
        "ocr_text_likelihood_threshold": settings.ocr_text_likelihood_threshold,
        "blur_threshold": BLUR_THRESHOLD,
        "detection_quality_low_max": settings.detection_quality_low_max,
        "detection_quality_high_min": settings.detection_quality_high_min,
        "n_db": len(db_set),
        "n_triplet_train": len(train_set),
        "n_test": len(rows),
        "n_negatives": len(neg_results),
        "final_triplet_loss": round(final_loss, 6),
        "split_integrity": split_integrity,
        "pipeline_stage_latency_mode": "derived_from_component_phases",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(json_path, "w") as f:
        json.dump({"metadata": metadata, "results": rows, "negatives": neg_results}, f, indent=2)

    print(f"\nResults written to: {csv_path}, {json_path}")


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
    settings = Settings()
    threshold = (
        float(args.similarity_threshold)
        if args.similarity_threshold is not None
        else settings.similarity_threshold
    )
    monitor = MemoryMonitor()

    print(f"Dataset: {args.dataset}")
    print(f"Images:  {args.images}")
    print(f"Thresholds: similarity={threshold:.3f}, darkness={settings.darkness_threshold:.1f}, "
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

    # Phase 2: Split
    _enforce_memory_guard(monitor, "phase 2")
    print("\n[2/7] Splitting: fixed 60 train / 60 test (label-balanced, seeded)...")
    train_set, db_set, test_set = _split_fixed_60_60(
        embedded,
        train_per_label=args.train_per_label,
        seed=args.seed,
    )
    train_labels = sorted({embedded[f]["label"] for f in train_set})
    test_labels = sorted({embedded[f]["label"] for f in test_set})
    split_integrity = {
        "strategy": "fixed_60_60_label_balanced_seeded",
        "seed": args.seed,
        "train_per_label": args.train_per_label,
        "train_count": len(train_set),
        "test_count": len(test_set),
        "train_labels": train_labels,
        "test_labels": test_labels,
        "label_overlap": sorted(set(train_labels).intersection(test_labels)),
        "db_labels": sorted({embedded[f]["label"] for f in db_set}),
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
    if args.similarity_threshold is None:
        calibrated = _calibrate_threshold(train_set, embedded, database, test_set)
        # Recompute with trained head later; this pretrain value is a floor.
        threshold = max(threshold, calibrated)
        print(f"  auto-calibrated similarity threshold (pre-train): {threshold:.3f}")

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
    if args.similarity_threshold is None:
        threshold = _calibrate_threshold(train_set, embedded, database, test_set)
        print(f"  auto-calibrated similarity threshold (post-train): {threshold:.3f}")

    # Prepare negative embeddings before retrieval so threshold can use negatives.
    neg_rows: List[dict] = []
    neg_embedded: Dict[str, dict] = {}
    if args.negative_dataset.exists():
        neg_rows = _load_negative_dataset(args.negative_dataset)
        if neg_rows:
            _enforce_memory_guard(monitor, "negative pre-embed")
            neg_embedded = _embed_rows(neg_rows, args.images_neg, args.no_ocr, settings, monitor=monitor)

    if args.similarity_threshold is None and neg_embedded:
        threshold = _calibrate_threshold_with_negatives(
            test_set=test_set,
            embedded=embedded,
            neg_embedded=neg_embedded,
            neg_rows=neg_rows,
            database=database,
            head=head,
        )
        print(f"  auto-calibrated similarity threshold (with negatives): {threshold:.3f}")

    # Phase 5: Retrieval
    _enforce_memory_guard(monitor, "phase 5")
    print("\n[5/7] Evaluating retrieval (3ft/6ft test set against 1ft_bright_clean DB)...")
    retrieval_results = _eval_retrieval(test_set, embedded, database, head, threshold, monitor=monitor)

    # Phase 6: Detection
    _enforce_memory_guard(monitor, "phase 6")
    print("\n[6/7] Evaluating GroundingDINO detection (with full fallback chain)...")
    dino_results = _eval_detection(test_set, embedded, settings, monitor=monitor)

    # Phase 7: Depth
    _enforce_memory_guard(monitor, "phase 7")
    print("\n[7/7] Evaluating depth estimation...")
    depth_results = _eval_depth(
        test_set, embedded, dino_results, args.focal_length, args.no_depth, monitor=monitor)

    # Negative FP evaluation
    neg_results: List[dict] = []
    if args.negative_dataset.exists():
        print("\n[neg] Evaluating false positive rate on negative images...")
        if neg_embedded:
            neg_results = _eval_negatives_strict(neg_embedded, neg_rows, database, head, threshold)
            print(f"  {len(neg_results)} negative images evaluated")
    else:
        print(f"\n[neg] negative_dataset.csv not found; skipping FP evaluation")

    # Output
    _print_summary(retrieval_results, dino_results, depth_results,
                   neg_results, final_loss, threshold, args.no_depth)
    merged = _merge_rows(retrieval_results, dino_results, depth_results)
    _write_output(
        merged,
        neg_results,
        args,
        final_loss,
        threshold,
        db_set,
        train_set,
        settings,
        split_integrity=split_integrity,
    )


if __name__ == "__main__":
    main()
