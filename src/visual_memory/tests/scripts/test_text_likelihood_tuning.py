"""Validation tests for text-likelihood threshold tuning (0.30 + rescue path)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Test file is at PROJECT_ROOT/src/visual_memory/tests/scripts/test_*.py
# parents[4] = PROJECT_ROOT
_test_file = Path(__file__).resolve()
PROJECT_ROOT = _test_file.parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visual_memory.tests.scripts.test_harness import TestRunner
from visual_memory.utils.quality_utils import estimate_text_likelihood
from visual_memory.config import Settings

_runner = TestRunner("text_likelihood_tuning")


def _high_contrast_pattern(size: tuple = (128, 128), frequency: int = 3) -> Image.Image:
    """Simulate text-like high-contrast edges."""
    arr = np.zeros((size[0], size[1]), dtype=np.uint8)
    arr[::frequency, :] = 255
    arr_rgb = np.stack([arr, arr, arr], axis=2)
    return Image.fromarray(arr_rgb, "RGB")


def _solid_image(value: int = 150, size: tuple = (128, 128)) -> Image.Image:
    """Simulate uniform textless object."""
    arr = np.full((size[0], size[1], 3), value, dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def test_threshold_and_rescue_defaults():
    """Verify threshold and rescue defaults are set for production tuning."""
    settings = Settings()
    assert settings.ocr_text_likelihood_threshold == 0.30, \
        f"Expected threshold 0.30, got {settings.ocr_text_likelihood_threshold}"
    assert settings.ocr_text_likelihood_rescue_threshold == 0.10, \
        f"Expected rescue threshold 0.10, got {settings.ocr_text_likelihood_rescue_threshold}"
    assert settings.ocr_text_likelihood_rescue_min_luminance == 40.0, \
        f"Expected rescue min luminance 40.0, got {settings.ocr_text_likelihood_rescue_min_luminance}"
    assert settings.ocr_text_likelihood_rescue_min_blur_score == 130.0, \
        f"Expected rescue min blur score 130.0, got {settings.ocr_text_likelihood_rescue_min_blur_score}"


def test_text_like_image_scores_above_threshold():
    """High-contrast text-like images should score well (above 0.30)."""
    img = _high_contrast_pattern(frequency=3)
    score = estimate_text_likelihood(img)
    assert score >= 0.30, \
        f"Text-like image should score >= 0.30 for OCR, got {score}"


def test_uniform_image_scores_below_threshold():
    """Uniform solid images have no edges, should score 0."""
    img = _solid_image(150)
    score = estimate_text_likelihood(img)
    assert score < 0.30, \
        f"Uniform image should score < 0.30, got {score}"


def test_threshold_config_is_float():
    """Verify threshold is a valid float between 0.0 and 1.0."""
    settings = Settings()
    threshold = settings.ocr_text_likelihood_threshold
    assert isinstance(threshold, float), \
        f"Threshold should be float, got {type(threshold)}"
    assert 0.0 <= threshold <= 1.0, \
        f"Threshold should be in [0.0, 1.0], got {threshold}"


def test_settings_file_is_readable():
    """Verify settings file exists and contains the threshold."""
    settings_path = PROJECT_ROOT / "src" / "visual_memory" / "config" / "settings.py"
    assert settings_path.exists(), f"Settings file not found at {settings_path}"
    
    content = settings_path.read_text()
    assert "ocr_text_likelihood_threshold" in content, \
        "Settings file should define ocr_text_likelihood_threshold"
    assert "ocr_text_likelihood_rescue_threshold" in content, \
        "Settings file should define ocr_text_likelihood_rescue_threshold"


for name, fn in [
    ("text_tuning:threshold_and_rescue_defaults", test_threshold_and_rescue_defaults),
    ("text_tuning:text_like_above_threshold", test_text_like_image_scores_above_threshold),
    ("text_tuning:uniform_below_threshold", test_uniform_image_scores_below_threshold),
    ("text_tuning:threshold_is_valid_float", test_threshold_config_is_float),
    ("text_tuning:settings_file_valid", test_settings_file_is_readable),
]:
    _runner.run(name, fn)

sys.exit(_runner.summary())
