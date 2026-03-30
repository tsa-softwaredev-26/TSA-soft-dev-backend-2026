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
                  "projection_head_ramp_at", "projection_head_epochs",
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
    ("user_settings:get_fields", test_get_user_settings_fields),
    ("user_settings:patch_performance_mode", test_patch_user_settings_performance_mode),
    ("user_settings:patch_invalid_performance_mode", test_patch_user_settings_invalid_performance_mode),
    ("user_settings:patch_invalid_voice_speed", test_patch_user_settings_invalid_voice_speed),
    ("user_settings:patch_valid_voice_speed", test_patch_user_settings_valid_voice_speed),
    ("user_settings:fast_mode_disables_second_pass", test_fast_mode_disables_second_pass),
    ("user_settings:accurate_mode_enables_second_pass", test_accurate_mode_enables_second_pass),
    ("user_settings:balanced_mode_enables_llm_fallback", test_balanced_mode_enables_llm_fallback),
]:
    _runner.run(name, fn)

cleanup()
sys.exit(_runner.summary())
