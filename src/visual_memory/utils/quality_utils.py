"""
Image quality checks used by both RememberPipeline and ScanPipeline.

All functions are pure numpy/PIL - no model loading, no extra dependencies.
"""
import numpy as np
from PIL import Image


def mean_luminance(image: Image.Image) -> float:
    """
    Mean pixel value of the grayscale image (0-255 scale).
    Lower = darker. A dark room with no lights typically reads < 30.
    """
    return float(np.array(image.convert("L"), dtype=np.float32).mean())


def estimate_text_likelihood(image: Image.Image) -> float:
    """
    Fast heuristic (< 2ms) for whether a crop likely contains readable text.

    Text regions have thin, high-contrast edges that produce a high mean
    absolute Laplacian relative to the image's overall dynamic range.
    Images of plain objects (fabric, metal, leather) score low; product
    labels, packaging, and text-bearing surfaces score high.

    Returns 0.0 (no text) to 1.0 (high text likelihood). Below
    settings.ocr_text_likelihood_threshold, pipelines skip OCR service calls.
    """
    small = np.array(image.convert("L").resize((128, 128)), dtype=np.float32)
    lap = np.abs(
        np.roll(small, 1, 0) + np.roll(small, -1, 0) +
        np.roll(small, 1, 1) + np.roll(small, -1, 1) - 4.0 * small
    )
    contrast = float(small.max() - small.min())
    if contrast < 10.0:
        return 0.0
    return min(1.0, float(lap.mean()) / (contrast * 0.35))
