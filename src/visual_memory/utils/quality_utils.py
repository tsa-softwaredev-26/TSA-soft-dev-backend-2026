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


def blur_score(image: Image.Image) -> float:
    """
    Laplacian variance sharpness score.

    Higher scores are sharper; low scores indicate blur.
    """
    gray = np.array(image.convert("L"), dtype=np.float32)
    lap = (
        np.roll(gray, 1, 0) + np.roll(gray, -1, 0) +
        np.roll(gray, 1, 1) + np.roll(gray, -1, 1) - 4.0 * gray
    )
    return float(lap.var())


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


def should_run_ocr(
    text_likelihood: float,
    *,
    lower_threshold: float,
    upper_threshold: float,
    luminance: float | None = None,
    blur_score: float | None = None,
    rescue_threshold: float = 0.0,
    rescue_min_luminance: float = 0.0,
    rescue_min_blur_score: float = 0.0,
) -> bool:
    """
    OCR gate with a bounded main band and an optional bright+sharp rescue path.

    Main path keeps OCR to text-like crops in [lower_threshold, upper_threshold].
    Rescue path allows OCR for lower-likelihood crops only when both image quality
    checks pass, which helps receipts in good lighting at short range.
    """
    v = float(text_likelihood)
    lo = float(lower_threshold)
    hi = float(upper_threshold)
    if lo <= v <= hi:
        return True
    if v < float(rescue_threshold) or v > hi:
        return False
    if luminance is None or blur_score is None:
        return False
    return (
        float(luminance) >= float(rescue_min_luminance)
        and float(blur_score) >= float(rescue_min_blur_score)
    )
