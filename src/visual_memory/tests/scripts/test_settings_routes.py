"""
Integration tests for GET/PATCH /settings and GET/PATCH /user-settings.
"""
from __future__ import annotations

import os
import sys

os.environ["ENABLE_DEPTH"] = "0"

import visual_memory.api.pipelines as _pm
from visual_memory.tests.scripts.test_harness import (
    TestRunner, make_test_app, assert_status,
)
from visual_memory.api.routes.settings_route import settings_bp
from visual_memory.api.routes.user_settings_route import user_settings_bp
from visual_memory.api.routes.feedback import feedback_bp

_runner = TestRunner("settings")


client, db, cleanup = make_test_app([settings_bp, user_settings_bp, feedback_bp])


def test_get_settings_fields():
    resp = client.get("/settings")
    assert_status(resp, 200)
    data = resp.get_json()
    for field in ("enable_learning", "min_feedback_for_training", "projection_head_weight",
                  "projection_head_ramp_at", "projection_head_ramp_power", "projection_head_epochs",
                  "triplet_margin", "triplet_positive_weight", "triplet_negative_weight",
                  "triplet_hard_negative_boost", "similarity_threshold",
                  "similarity_threshold_baseline", "similarity_threshold_personalized",
                  "similarity_threshold_document", "scan_similarity_margin",
                  "scan_similarity_margin_document", "remember_max_prototypes_per_label",
                  "head_trained", "triplet_count", "feedback_counts"):
        assert field in data, f"missing field: {field}"


def test_patch_settings_enable_learning():
    resp = client.patch("/settings", json={"enable_learning": False})
    assert_status(resp, 200)
    data = resp.get_json()
    assert data.get("enable_learning") is False


def test_patch_settings_invalid_head_weight():
    resp = client.patch("/settings", json={"projection_head_weight": 1.5})
    assert_status(resp, 400)
    data = resp.get_json()
    assert "errors" in data


def test_patch_settings_invalid_head_weight_negative():
    resp = client.patch("/settings", json={"projection_head_weight": -0.1})
    assert_status(resp, 400)


def test_patch_settings_invalid_min_feedback():
    resp = client.patch("/settings", json={"min_feedback_for_training": 0})
    assert_status(resp, 400)


def test_patch_settings_unrecognized_field_ignored():
    resp = client.patch("/settings", json={"unknown_field_xyz": True})
    assert_status(resp, 200)


def test_patch_settings_valid_head_weight():
    resp = client.patch("/settings", json={"projection_head_weight": 0.5})
    assert_status(resp, 200)
    data = resp.get_json()
    assert data.get("projection_head_weight") == 0.5


def test_patch_settings_valid_new_ml_fields():
    payload = {
        "projection_head_ramp_power": 1.5,
        "triplet_margin": 0.3,
        "triplet_positive_weight": 1.2,
        "triplet_negative_weight": 0.9,
        "triplet_hard_negative_boost": 0.25,
    }
    resp = client.patch("/settings", json=payload)
    assert_status(resp, 200)
    data = resp.get_json()
    for key, value in payload.items():
        assert data.get(key) == value


def test_patch_settings_invalid_new_ml_fields():
    resp = client.patch("/settings", json={"projection_head_ramp_power": 0.0})
    assert_status(resp, 400)
    resp = client.patch("/settings", json={"triplet_positive_weight": 0.0})
    assert_status(resp, 400)
    resp = client.patch("/settings", json={"triplet_negative_weight": -1.0})
    assert_status(resp, 400)
    resp = client.patch("/settings", json={"triplet_hard_negative_boost": -0.1})
    assert_status(resp, 400)


def test_patch_settings_persists_new_ml_fields():
    payload = {
        "projection_head_ramp_power": 1.25,
        "triplet_margin": 0.25,
        "triplet_positive_weight": 1.1,
        "triplet_negative_weight": 1.05,
        "triplet_hard_negative_boost": 0.1,
    }
    resp = client.patch("/settings", json=payload)
    assert_status(resp, 200)
    saved = db.load_ml_settings() or {}
    for key, value in payload.items():
        assert saved.get(key) == value


def test_patch_settings_similarity_thresholds():
    payload = {
        "similarity_threshold": 0.71,
        "similarity_threshold_baseline": 0.72,
        "similarity_threshold_personalized": 0.73,
        "similarity_threshold_document": 0.74,
    }
    resp = client.patch("/settings", json=payload)
    assert_status(resp, 200)
    data = resp.get_json()
    for key, value in payload.items():
        assert data.get(key) == value
    saved = db.load_ml_settings() or {}
    for key, value in payload.items():
        assert saved.get(key) == value


def test_patch_settings_margin_and_prototype_fields():
    payload = {
        "scan_similarity_margin": 0.04,
        "scan_similarity_margin_document": 0.02,
        "remember_max_prototypes_per_label": 5,
    }
    resp = client.patch("/settings", json=payload)
    assert_status(resp, 200)
    data = resp.get_json()
    for key, value in payload.items():
        assert data.get(key) == value
    saved = db.load_ml_settings() or {}
    for key, value in payload.items():
        assert saved.get(key) == value


def test_patch_settings_invalid_similarity_threshold():
    resp = client.patch("/settings", json={"similarity_threshold_document": 1.1})
    assert_status(resp, 400)
    data = resp.get_json()
    assert "errors" in data


def test_patch_settings_invalid_margin_and_prototype_fields():
    resp = client.patch("/settings", json={"scan_similarity_margin": -0.01})
    assert_status(resp, 400)
    resp = client.patch("/settings", json={"scan_similarity_margin_document": -0.1})
    assert_status(resp, 400)
    resp = client.patch("/settings", json={"remember_max_prototypes_per_label": 0})
    assert_status(resp, 400)


def test_patch_settings_invalid_json():
    resp = client.patch(
        "/settings",
        data='{"enable_learning":',
        content_type="application/json",
    )
    assert_status(resp, 400)


def test_get_user_settings_fields():
    resp = client.get("/user-settings")
    assert_status(resp, 200)
    data = resp.get_json()
    for field in ("performance_mode", "voice_speed", "learning_enabled", "button_layout", "performance_config"):
        assert field in data, f"missing field: {field}"
    assert "depth_enabled" in data["performance_config"]
    assert "target_latency" in data["performance_config"]
    assert "vlm_enabled" in data["performance_config"]
    assert "vlm_timeout_seconds" in data["performance_config"]


def test_patch_user_settings_performance_mode():
    resp = client.patch("/user-settings", json={"performance_mode": "fast"})
    assert_status(resp, 200)
    data = resp.get_json()
    assert data.get("performance_mode") == "fast"


def test_patch_user_settings_invalid_performance_mode():
    resp = client.patch("/user-settings", json={"performance_mode": "turbo"})
    assert_status(resp, 400)


def test_patch_user_settings_invalid_voice_speed():
    resp = client.patch("/user-settings", json={"voice_speed": 3.0})
    assert_status(resp, 400)


def test_patch_user_settings_valid_voice_speed():
    resp = client.patch("/user-settings", json={"voice_speed": 1.5})
    assert_status(resp, 200)
    data = resp.get_json()
    assert data.get("voice_speed") == 1.5


def test_patch_user_settings_learning_enabled():
    resp = client.patch("/user-settings", json={"learning_enabled": False})
    assert_status(resp, 200)
    data = resp.get_json()
    assert data.get("learning_enabled") is False


def test_patch_user_settings_button_layout():
    resp = client.patch("/user-settings", json={"button_layout": "swapped"})
    assert_status(resp, 200)
    data = resp.get_json()
    assert data.get("button_layout") == "swapped"


def test_patch_user_settings_all_fields_and_persistence():
    payload = {
        "performance_mode": "accurate",
        "voice_speed": 1.25,
        "learning_enabled": False,
        "button_layout": "swapped",
    }
    resp = client.patch("/user-settings", json=payload)
    assert_status(resp, 200)
    data = resp.get_json()
    for key, value in payload.items():
        assert data.get(key) == value

    saved = db.load_user_settings() or {}
    for key, value in payload.items():
        assert saved.get(key) == value


def test_patch_user_settings_invalid_json():
    resp = client.patch(
        "/user-settings",
        data='{"performance_mode":',
        content_type="application/json",
    )
    assert_status(resp, 400)


def test_fast_mode_disables_second_pass():
    resp = client.patch("/user-settings", json={"performance_mode": "fast"})
    assert_status(resp, 200)
    s = _pm._settings
    assert s.detection_second_pass_enabled is False
    assert s.llm_query_fallback_enabled is False


def test_accurate_mode_enables_second_pass():
    resp = client.patch("/user-settings", json={"performance_mode": "accurate"})
    assert_status(resp, 200)
    s = _pm._settings
    assert s.detection_second_pass_enabled is True
    assert s.llm_query_fallback_enabled is True


def test_balanced_mode_enables_llm_fallback():
    resp = client.patch("/user-settings", json={"performance_mode": "balanced"})
    assert_status(resp, 200)
    s = _pm._settings
    assert s.detection_second_pass_enabled is True
    assert s.llm_query_fallback_enabled is True


for name, fn in [
    ("settings:get_fields", test_get_settings_fields),
    ("settings:patch_enable_learning", test_patch_settings_enable_learning),
    ("settings:patch_invalid_head_weight_high", test_patch_settings_invalid_head_weight),
    ("settings:patch_invalid_head_weight_negative", test_patch_settings_invalid_head_weight_negative),
    ("settings:patch_invalid_min_feedback", test_patch_settings_invalid_min_feedback),
    ("settings:patch_unrecognized_field_ignored", test_patch_settings_unrecognized_field_ignored),
    ("settings:patch_valid_head_weight", test_patch_settings_valid_head_weight),
    ("settings:patch_valid_new_ml_fields", test_patch_settings_valid_new_ml_fields),
    ("settings:patch_invalid_new_ml_fields", test_patch_settings_invalid_new_ml_fields),
    ("settings:patch_persists_new_ml_fields", test_patch_settings_persists_new_ml_fields),
    ("settings:patch_similarity_thresholds", test_patch_settings_similarity_thresholds),
    ("settings:patch_margin_and_prototype_fields", test_patch_settings_margin_and_prototype_fields),
    ("settings:patch_invalid_similarity_threshold", test_patch_settings_invalid_similarity_threshold),
    ("settings:patch_invalid_margin_and_prototype_fields", test_patch_settings_invalid_margin_and_prototype_fields),
    ("settings:patch_invalid_json", test_patch_settings_invalid_json),
    ("user_settings:get_fields", test_get_user_settings_fields),
    ("user_settings:patch_performance_mode", test_patch_user_settings_performance_mode),
    ("user_settings:patch_invalid_performance_mode", test_patch_user_settings_invalid_performance_mode),
    ("user_settings:patch_invalid_voice_speed", test_patch_user_settings_invalid_voice_speed),
    ("user_settings:patch_valid_voice_speed", test_patch_user_settings_valid_voice_speed),
    ("user_settings:patch_learning_enabled", test_patch_user_settings_learning_enabled),
    ("user_settings:patch_button_layout", test_patch_user_settings_button_layout),
    ("user_settings:patch_all_fields_and_persist", test_patch_user_settings_all_fields_and_persistence),
    ("user_settings:patch_invalid_json", test_patch_user_settings_invalid_json),
    ("user_settings:fast_mode_disables_second_pass", test_fast_mode_disables_second_pass),
    ("user_settings:accurate_mode_enables_second_pass", test_accurate_mode_enables_second_pass),
    ("user_settings:balanced_mode_enables_llm_fallback", test_balanced_mode_enables_llm_fallback),
]:
    _runner.run(name, fn)

cleanup()
sys.exit(_runner.summary())
