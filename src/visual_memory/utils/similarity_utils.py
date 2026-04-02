"""Similarity utilities for embedding search and box filtering."""

from typing import Optional, Tuple, List, Dict, Any, Callable
import torch
import torch.nn as nn
import re

_cos_sim = nn.CosineSimilarity(dim=1, eps=1e-8)
_DOCUMENT_LABEL_PATTERN = re.compile(
    r"\b("
    r"receipt|document|invoice|bill|statement|form|contract|letter|memo|note|"
    r"paper|passport|license|id|card|menu|ticket|coupon|label|text"
    r")s?\b",
    flags=re.IGNORECASE,
)

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return cosine similarity between two (1, dim) tensors."""
    return _cos_sim(a, b)

def find_match(
    query_embedding: torch.Tensor,
    database_embeddings: List[Tuple[str, torch.Tensor]],
    threshold: float,
) -> Tuple[Optional[str], float]:
    """
    Return best match above similarity threshold.
    Linear search over database.
    """

    if not database_embeddings:
        return None, 0.0

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    best_path = None
    best_similarity = -1.0

    for path, db_embedding in database_embeddings:
        sim = cosine_similarity(query_embedding, db_embedding)
        if sim > best_similarity:
            best_similarity = sim
            best_path = path

    if best_similarity < threshold:
        return None, 0.0

    return best_path, best_similarity.item()


def find_match_dynamic_threshold(
    query_embedding: torch.Tensor,
    database_embeddings: List[Tuple[str, torch.Tensor]],
    threshold_for_path: Callable[[str], float],
    margin_for_path: Optional[Callable[[str], float]] = None,
) -> Tuple[Optional[str], float, float]:
    """Return best match where each candidate can have its own threshold and top1-top2 margin."""
    if not database_embeddings:
        return None, 0.0, 0.0

    ranked: List[Tuple[float, str, float, float]] = []

    for path, db_embedding in database_embeddings:
        threshold = float(threshold_for_path(path))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0 and 1.")
        margin = float(margin_for_path(path)) if margin_for_path is not None else 0.0
        if margin < 0.0:
            raise ValueError("Margin must be >= 0.")
        sim = float(cosine_similarity(query_embedding, db_embedding))
        ranked.append((sim, path, threshold, margin))

    eligible = [row for row in ranked if row[0] >= row[2]]
    if not eligible:
        return None, 0.0, 0.0

    eligible.sort(key=lambda row: row[0], reverse=True)
    best_similarity, best_path, _best_threshold, best_margin = eligible[0]
    second_similarity = max((row[0] for row in ranked if row[1] != best_path), default=0.0)
    similarity_margin = best_similarity - second_similarity

    if similarity_margin < best_margin:
        return None, 0.0, similarity_margin

    return best_path, best_similarity, similarity_margin


def is_document_like_label(label: str) -> bool:
    """Return True when label likely refers to a document/text-heavy object."""
    if not label:
        return False
    normalized = str(label).replace("_", " ").replace("-", " ")
    return _DOCUMENT_LABEL_PATTERN.search(normalized) is not None

def iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection-over-Union between two boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    union = area1 + area2 - inter_area
    return 0.0 if union == 0 else inter_area / union

def deduplicate_matches(
    matches: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Remove overlapping or duplicate detections.
    Keeps highest similarity match first.
    """

    if not matches:
        return []

    matches = sorted(matches, key=lambda m: m["similarity"], reverse=True)
    kept = []

    for match in matches:
        duplicate = False

        for k in kept:
            if iou(match["box"], k["box"]) > iou_threshold:
                if match["label"] == k["label"]:
                    duplicate = True
                    break

        if not duplicate:
            kept.append(match)

    return kept
