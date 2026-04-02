"""CPU-only tests for benchmark threshold and detector helpers."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["ENABLE_DEPTH"] = "0"
os.environ["ENABLE_OCR"] = "0"

from visual_memory.benchmarks import full_benchmark as fb
from visual_memory.benchmarks import format_results as fr
from visual_memory.config import Settings
from visual_memory.database.store import DatabaseStore
from visual_memory.tests.scripts.test_harness import TestRunner

_runner = TestRunner("full_benchmark")


def _make_args(**overrides):
    base = dict(
        similarity_threshold=None,
        baseline_threshold=None,
        personalized_threshold=None,
        document_threshold=None,
        auto_tune_thresholds=False,
        threshold_sweep_min=0.10,
        threshold_sweep_max=0.90,
        threshold_sweep_step=0.01,
        target_holdout_fp=0.40,
        min_personalized_accuracy=0.15,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_persisted_ml_settings_are_applied():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "bench.db"
        db = DatabaseStore(db_path)
        db.save_ml_settings(
            {
                "similarity_threshold_baseline": 0.61,
                "similarity_threshold_personalized": 0.62,
                "similarity_threshold_document": 0.63,
                "scan_similarity_margin": 0.11,
                "scan_similarity_margin_document": 0.07,
            }
        )

        settings = Settings(db_path=str(db_path))
        applied = fb._apply_persisted_ml_settings(settings)

        assert applied["scan_similarity_margin"] == 0.11
        assert settings.get_similarity_threshold_baseline() == 0.61
        assert settings.get_similarity_threshold_personalized() == 0.62
        assert settings.get_similarity_threshold_document() == 0.63
        assert settings.get_scan_similarity_margin() == 0.11
        assert settings.get_scan_similarity_margin_document() == 0.07


def test_threshold_selection_defaults_to_production_settings():
    settings = Settings()
    settings.similarity_threshold_baseline = 0.61
    settings.similarity_threshold_personalized = 0.62
    settings.similarity_threshold_document = 0.63

    thresholds, sweep, tuning = fb._resolve_thresholds(_make_args(), settings)

    assert thresholds == {"baseline": 0.61, "personalized": 0.62, "document": 0.63}
    assert sweep == []
    assert tuning["selection_reason"] == "production_settings"
    assert tuning["auto_tune_requested"] is False
    assert tuning["selected_thresholds"] == {"baseline": 0.61, "personalized": 0.62, "document": 0.63}


def test_auto_tune_thresholds_requires_opt_in():
    settings = Settings()
    calls = {"count": 0}

    def fake_tune(*args, **kwargs):
        calls["count"] += 1
        return (
            {"baseline": 0.7, "personalized": 0.71, "document": 0.72},
            [{"selection_reason": "fake"}],
            {"selection_reason": "fake", "selected_thresholds": {"baseline": 0.7, "personalized": 0.71, "document": 0.72}},
        )

    original = fb._tune_threshold_with_holdout
    fb._tune_threshold_with_holdout = fake_tune
    try:
        thresholds, sweep, tuning = fb._resolve_thresholds(
            _make_args(auto_tune_thresholds=False),
            settings,
        )
        assert calls["count"] == 0
        assert thresholds["baseline"] == settings.get_similarity_threshold_baseline()
        assert tuning["selection_reason"] == "production_settings"
        assert sweep == []

        thresholds, sweep, tuning = fb._resolve_thresholds(
            _make_args(auto_tune_thresholds=True),
            settings,
            similarity_stats=[{"baseline_accuracy": 1.0}],
        )
        assert calls["count"] == 1
        assert thresholds == {"baseline": 0.7, "personalized": 0.71, "document": 0.72}
        assert sweep == [{"selection_reason": "fake"}]
        assert tuning["selection_reason"] == "fake"
    finally:
        fb._tune_threshold_with_holdout = original


def test_yoloe_summary_aggregates_detection_metrics():
    embedded = {
        "wallet_1.jpg": {"label": "wallet", "condition_bucket": "1ft_bright_clean"},
        "receipt_6.jpg": {"label": "receipt", "condition_bucket": "6ft_dim_messy"},
    }
    yoloe = {
        "wallet_1.jpg": {
            "detected": 1,
            "box_count": 2,
            "max_confidence": 0.84,
            "mean_confidence": 0.73,
            "lat_detect": 0.11,
        },
        "receipt_6.jpg": {
            "detected": 0,
            "box_count": 0,
            "max_confidence": 0.0,
            "mean_confidence": 0.0,
            "lat_detect": 0.15,
        },
    }

    summary = fb._summarize_yoloe_results(yoloe, embedded)

    assert summary["available"] is True
    assert summary["evaluated"] == 2
    assert summary["detected"] == 1
    assert summary["by_label"]["wallet"]["detected"] == 1
    assert summary["by_label"]["receipt"]["detected"] == 0
    assert summary["by_condition"]["1ft_bright_clean"]["detected"] == 1
    assert summary["by_condition"]["6ft_dim_messy"]["n"] == 1
    assert abs(summary["mean_box_count"] - 1.0) < 1e-6
    assert abs(summary["mean_max_confidence"] - 0.42) < 1e-6


def test_formatter_includes_threshold_and_yoloe_sections():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        results_path = tmp / "results.csv"
        metadata_path = tmp / "results.json"
        output_path = tmp / "BENCHMARKS.md"

        row_a = {field: "" for field in fb._CSV_FIELDS}
        row_a.update(
            {
                "image": "wallet_1.jpg",
                "label": "wallet",
                "ground_truth_distance_ft": 1.0,
                "distance_bucket": "1ft",
                "lighting_bucket": "bright",
                "cleanliness_bucket": "clean",
                "condition_bucket": "1ft_bright_clean",
                "is_document": 0,
                "baseline_similarity": 0.55,
                "personalized_similarity": 0.65,
                "similarity_gap": 0.10,
                "baseline_threshold_used": 0.61,
                "personalized_threshold_used": 0.62,
                "baseline_correct": 0,
                "personalized_correct": 1,
                "yoloe_detected": 1,
                "yoloe_box_count": 2,
                "yoloe_max_confidence": 0.84,
                "yoloe_mean_confidence": 0.73,
                "yoloe_confidence_threshold": 0.35,
                "yoloe_iou_threshold": 0.45,
                "yoloe_lat_detect_s": 0.11,
                "dino_detected": 1,
                "dino_confidence": 0.59,
                "dino_second_pass_prompt": "",
                "predicted_distance_ft": 1.2,
                "depth_absolute_error": 0.2,
                "depth_percentage_error": 20.0,
                "lat_embed_img_s": 0.1,
                "lat_ocr_s": 0.0,
                "lat_embed_txt_s": 0.0,
                "lat_yoloe_s": 0.11,
                "lat_retrieve_bl_s": 0.01,
                "lat_retrieve_pe_s": 0.01,
                "lat_detect_s": 0.2,
                "lat_depth_s": 0.3,
                "darkness_level": 40.0,
                "is_dark": 0,
                "blur_score": 150.0,
                "is_blurry": 0,
                "text_likelihood": 0.2,
                "should_skip_ocr": 1,
                "holdout_baseline_fp": 0,
                "holdout_baseline_match": "",
                "holdout_baseline_sim": "",
                "holdout_personalized_fp": 0,
                "holdout_personalized_match": "",
                "holdout_personalized_sim": "",
                "expanded_negative_pool_size": 0,
                "expanded_baseline_fp": 0,
                "expanded_baseline_match": "",
                "expanded_baseline_sim": "",
                "expanded_personalized_fp": 0,
                "expanded_personalized_match": "",
                "expanded_personalized_sim": "",
            }
        )
        row_b = {field: "" for field in fb._CSV_FIELDS}
        row_b.update(
            {
                "image": "receipt_6.jpg",
                "label": "receipt",
                "ground_truth_distance_ft": 6.0,
                "distance_bucket": "6ft",
                "lighting_bucket": "dim",
                "cleanliness_bucket": "messy",
                "condition_bucket": "6ft_dim_messy",
                "is_document": 1,
                "baseline_similarity": 0.35,
                "personalized_similarity": 0.45,
                "similarity_gap": 0.10,
                "baseline_threshold_used": 0.61,
                "personalized_threshold_used": 0.62,
                "baseline_correct": 0,
                "personalized_correct": 0,
                "yoloe_detected": 0,
                "yoloe_box_count": 0,
                "yoloe_max_confidence": 0.0,
                "yoloe_mean_confidence": 0.0,
                "yoloe_confidence_threshold": 0.35,
                "yoloe_iou_threshold": 0.45,
                "yoloe_lat_detect_s": 0.15,
                "dino_detected": 0,
                "dino_confidence": 0.0,
                "dino_second_pass_prompt": "",
                "predicted_distance_ft": "",
                "depth_absolute_error": "",
                "depth_percentage_error": "",
                "lat_embed_img_s": 0.2,
                "lat_ocr_s": 0.0,
                "lat_embed_txt_s": 0.0,
                "lat_yoloe_s": 0.15,
                "lat_retrieve_bl_s": 0.01,
                "lat_retrieve_pe_s": 0.01,
                "lat_detect_s": 0.2,
                "lat_depth_s": 0.0,
                "darkness_level": 20.0,
                "is_dark": 1,
                "blur_score": 80.0,
                "is_blurry": 1,
                "text_likelihood": 0.05,
                "should_skip_ocr": 1,
                "holdout_baseline_fp": 0,
                "holdout_baseline_match": "",
                "holdout_baseline_sim": "",
                "holdout_personalized_fp": 0,
                "holdout_personalized_match": "",
                "holdout_personalized_sim": "",
                "expanded_negative_pool_size": 0,
                "expanded_baseline_fp": 0,
                "expanded_baseline_match": "",
                "expanded_baseline_sim": "",
                "expanded_personalized_fp": 0,
                "expanded_personalized_match": "",
                "expanded_personalized_sim": "",
            }
        )

        with open(results_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fb._CSV_FIELDS)
            writer.writeheader()
            writer.writerow(row_a)
            writer.writerow(row_b)

        metadata = {
            "metadata": {
                "timestamp": "2026-04-01T00:00:00Z",
                "similarity_threshold": 0.62,
                "similarity_thresholds": {
                    "baseline": 0.61,
                    "personalized": 0.62,
                    "document": 0.63,
                },
                "threshold_strategy": "production_settings",
                "threshold_settings_source": "persisted_db",
                "epochs": 1,
                "lr": 0.0001,
                "final_triplet_loss": 0.01,
                "seed": 42,
                "n_triplet_train": 2,
                "n_test": 2,
                "n_fp_holdout": 0,
                "n_fp_expanded": 0,
                "darkness_threshold": 30.0,
                "blur_threshold": 100.0,
                "ocr_text_likelihood_threshold": 0.30,
                "yoloe_summary": {
                    "available": True,
                    "evaluated": 2,
                    "detected": 1,
                    "detection_rate": 0.5,
                    "mean_box_count": 1.0,
                    "mean_max_confidence": 0.42,
                    "mean_mean_confidence": 0.365,
                    "mean_latency_s": 0.13,
                    "by_label": {},
                    "by_condition": {},
                },
            }
        }
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        fr.generate(results_path, metadata_path, output_path)
        output = output_path.read_text(encoding="utf-8")
        assert "Threshold strategy" in output
        assert "Detection (YOLOE)" in output
        assert "YOLOE By Condition" in output


for name, fn in [
    ("benchmark:persisted_ml_settings", test_persisted_ml_settings_are_applied),
    ("benchmark:thresholds_default_to_production", test_threshold_selection_defaults_to_production_settings),
    ("benchmark:auto_tune_requires_opt_in", test_auto_tune_thresholds_requires_opt_in),
    ("benchmark:yoloe_summary", test_yoloe_summary_aggregates_detection_metrics),
    ("benchmark:formatter_includes_yoloe", test_formatter_includes_threshold_and_yoloe_sections),
]:
    _runner.run(name, fn)

sys.exit(_runner.summary())
