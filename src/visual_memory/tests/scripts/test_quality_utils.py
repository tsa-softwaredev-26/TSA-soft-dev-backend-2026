"""
Unit tests for quality_utils: mean_luminance, estimate_text_likelihood.
Pure PIL/numpy - no models.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from visual_memory.tests.scripts.test_harness import TestRunner
from visual_memory.utils.quality_utils import mean_luminance, estimate_text_likelihood

_runner = TestRunner("quality_utils")


def _solid(value: int, size: tuple = (64, 64)) -> Image.Image:
    return Image.new("RGB", size, color=(value, value, value))


def _gradient(size: tuple = (64, 64)) -> Image.Image:
    arr = np.linspace(0, 255, size[0] * size[1], dtype=np.uint8).reshape(size)
    arr_rgb = np.stack([arr, arr, arr], axis=2)
    return Image.fromarray(arr_rgb, "RGB")


def _text_like(size: tuple = (128, 128)) -> Image.Image:
    """High-contrast checkerboard pattern simulating text edges."""
    arr = np.zeros((size[0], size[1]), dtype=np.uint8)
    arr[::4, :] = 255  # horizontal lines every 4 pixels
    arr_rgb = np.stack([arr, arr, arr], axis=2)
    return Image.fromarray(arr_rgb, "RGB")


def test_black_image_luminance_near_zero():
    img = _solid(0)
    lum = mean_luminance(img)
    assert lum < 5.0, f"expected <5, got {lum}"


def test_white_image_luminance_near_255():
    img = _solid(255)
    lum = mean_luminance(img)
    assert lum > 250.0, f"expected >250, got {lum}"


def test_gray_image_luminance_near_128():
    img = _solid(128)
    lum = mean_luminance(img)
    assert 120.0 <= lum <= 136.0, f"expected ~128, got {lum}"


def test_luminance_returns_float():
    img = _solid(100)
    result = mean_luminance(img)
    assert isinstance(result, float)


def test_text_like_image_high_likelihood():
    img = _text_like()
    score = estimate_text_likelihood(img)
    # High-contrast edge pattern should score above the default threshold (0.10)
    assert score > 0.1, f"expected >0.1, got {score}"


def test_smooth_image_low_likelihood():
    img = _solid(128)
    score = estimate_text_likelihood(img)
    # Uniform solid image has no edges, should score 0
    assert score == 0.0, f"expected 0.0, got {score}"


def test_text_likelihood_returns_float_in_range():
    img = _gradient()
    score = estimate_text_likelihood(img)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0, f"score out of range: {score}"


def test_very_low_contrast_returns_zero():
    # Near-uniform image -> contrast < 10, short-circuits to 0.0
    arr = np.full((64, 64, 3), 128, dtype=np.uint8)
    arr[0, 0] = 130  # tiny variation
    img = Image.fromarray(arr, "RGB")
    score = estimate_text_likelihood(img)
    assert score == 0.0, f"expected 0.0 for low-contrast, got {score}"


for name, fn in [
    ("quality:black_luminance", test_black_image_luminance_near_zero),
    ("quality:white_luminance", test_white_image_luminance_near_255),
    ("quality:gray_luminance", test_gray_image_luminance_near_128),
    ("quality:luminance_returns_float", test_luminance_returns_float),
    ("quality:text_high_likelihood", test_text_like_image_high_likelihood),
    ("quality:smooth_low_likelihood", test_smooth_image_low_likelihood),
    ("quality:likelihood_in_range", test_text_likelihood_returns_float_in_range),
    ("quality:low_contrast_zero", test_very_low_contrast_returns_zero),
]:
    _runner.run(name, fn)

sys.exit(_runner.summary())
