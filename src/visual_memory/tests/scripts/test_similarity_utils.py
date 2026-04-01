"""
Unit tests for similarity_utils: cosine_similarity, find_match, iou,
deduplicate_matches.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

from visual_memory.tests.scripts.test_harness import TestRunner
from visual_memory.utils.similarity_utils import (
    cosine_similarity,
    find_match,
    find_match_dynamic_threshold,
    iou,
    deduplicate_matches,
    is_document_like_label,
)

_runner = TestRunner("similarity_utils")


def _norm(seed: int, dim: int = 512) -> torch.Tensor:
    torch.manual_seed(seed)
    return F.normalize(torch.randn(1, dim), dim=1)


def test_cosine_similarity_identical():
    v = _norm(0)
    sim = cosine_similarity(v, v)
    assert abs(float(sim) - 1.0) < 1e-5, f"identical vectors should have sim=1.0, got {float(sim)}"


def test_cosine_similarity_orthogonal():
    a = torch.zeros(1, 4)
    a[0, 0] = 1.0
    b = torch.zeros(1, 4)
    b[0, 1] = 1.0
    sim = float(cosine_similarity(a, b))
    assert abs(sim) < 1e-5, f"orthogonal vectors should have sim~0, got {sim}"


def test_find_match_below_threshold_returns_none():
    query = _norm(0)
    db = [("label_a", _norm(1)), ("label_b", _norm(2))]
    # Use a very high threshold to ensure no match
    label, sim = find_match(query, db, threshold=0.9999)
    assert label is None
    assert sim == 0.0


def test_find_match_identical_returns_match():
    query = _norm(0)
    db = [("wallet", query.clone())]
    label, sim = find_match(query, db, threshold=0.5)
    assert label == "wallet"
    assert sim > 0.9


def test_find_match_empty_db():
    query = _norm(0)
    label, sim = find_match(query, [], threshold=0.3)
    assert label is None
    assert sim == 0.0


def test_find_match_returns_best():
    query = _norm(0)
    # First entry is very similar (same seed), second is different
    db = [("keys", _norm(99)), ("wallet", _norm(0))]
    label, sim = find_match(query, db, threshold=0.5)
    assert label == "wallet"


def test_find_match_dynamic_threshold_prefers_best_eligible():
    query = _norm(0)
    db = [("wallet", _norm(0)), ("receipt", _norm(1))]
    label, sim = find_match_dynamic_threshold(
        query,
        db,
        lambda path: 0.95 if path == "wallet" else 0.0,
    )
    assert label == "wallet"
    assert sim > 0.95


def test_iou_identical_boxes():
    box = [0.0, 0.0, 100.0, 100.0]
    result = iou(box, box)
    assert abs(result - 1.0) < 1e-5, f"identical boxes should have IoU=1.0, got {result}"


def test_iou_non_overlapping():
    box1 = [0.0, 0.0, 10.0, 10.0]
    box2 = [20.0, 20.0, 30.0, 30.0]
    result = iou(box1, box2)
    assert result == 0.0, f"non-overlapping boxes should have IoU=0, got {result}"


def test_iou_partial_overlap():
    box1 = [0.0, 0.0, 10.0, 10.0]
    box2 = [5.0, 5.0, 15.0, 15.0]
    result = iou(box1, box2)
    assert 0.0 < result < 1.0, f"partial overlap should have 0 < IoU < 1, got {result}"


def test_deduplicate_keeps_highest_similarity():
    matches = [
        {"label": "wallet", "similarity": 0.9, "box": [0, 0, 50, 50]},
        {"label": "wallet", "similarity": 0.6, "box": [5, 5, 55, 55]},  # overlapping
    ]
    result = deduplicate_matches(matches, iou_threshold=0.5)
    assert len(result) == 1
    assert result[0]["similarity"] == 0.9


def test_deduplicate_different_labels_both_kept():
    matches = [
        {"label": "wallet", "similarity": 0.9, "box": [0, 0, 50, 50]},
        {"label": "keys", "similarity": 0.8, "box": [5, 5, 55, 55]},  # overlapping but different label
    ]
    result = deduplicate_matches(matches, iou_threshold=0.5)
    assert len(result) == 2


def test_deduplicate_empty():
    result = deduplicate_matches([], iou_threshold=0.5)
    assert result == []


def test_deduplicate_no_overlap():
    matches = [
        {"label": "wallet", "similarity": 0.9, "box": [0, 0, 10, 10]},
        {"label": "wallet", "similarity": 0.8, "box": [50, 50, 60, 60]},  # far away
    ]
    result = deduplicate_matches(matches, iou_threshold=0.5)
    assert len(result) == 2


def test_is_document_like_label_positive():
    assert is_document_like_label("receipt") is True
    assert is_document_like_label("tax-document") is True
    assert is_document_like_label("paper_note") is True


def test_is_document_like_label_negative():
    assert is_document_like_label("wallet") is False
    assert is_document_like_label("keys") is False


for name, fn in [
    ("sim:cosine_identical", test_cosine_similarity_identical),
    ("sim:cosine_orthogonal", test_cosine_similarity_orthogonal),
    ("sim:find_match_below_threshold", test_find_match_below_threshold_returns_none),
    ("sim:find_match_identical", test_find_match_identical_returns_match),
    ("sim:find_match_empty_db", test_find_match_empty_db),
    ("sim:find_match_returns_best", test_find_match_returns_best),
    ("sim:find_match_dynamic_threshold", test_find_match_dynamic_threshold_prefers_best_eligible),
    ("sim:iou_identical", test_iou_identical_boxes),
    ("sim:iou_non_overlapping", test_iou_non_overlapping),
    ("sim:iou_partial_overlap", test_iou_partial_overlap),
    ("sim:dedup_keeps_highest", test_deduplicate_keeps_highest_similarity),
    ("sim:dedup_different_labels_both_kept", test_deduplicate_different_labels_both_kept),
    ("sim:dedup_empty", test_deduplicate_empty),
    ("sim:dedup_no_overlap", test_deduplicate_no_overlap),
    ("sim:document_label_positive", test_is_document_like_label_positive),
    ("sim:document_label_negative", test_is_document_like_label_negative),
]:
    _runner.run(name, fn)

sys.exit(_runner.summary())
