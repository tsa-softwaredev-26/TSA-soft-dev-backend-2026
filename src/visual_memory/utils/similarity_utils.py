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
) -> Tuple[Optional[str], float]:
    """Return best match where each candidate can have its own threshold."""
    if not database_embeddings:
        return None, 0.0

    best_path = None
    best_similarity = -1.0

    for path, db_embedding in database_embeddings:
        threshold = float(threshold_for_path(path))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0 and 1.")
        sim = float(cosine_similarity(query_embedding, db_embedding))
        if sim >= threshold and sim > best_similarity:
            best_similarity = sim
            best_path = path

    if best_path is None:
        return None, 0.0

    return best_path, best_similarity


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
