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

    edge20 = float((lap > 20.0).mean())
    edge30 = float((lap > 30.0).mean())
    edge_ratio = edge30 / (edge20 + 1e-6)
    dark_pix = float((small < 70.0).mean())
    bright_pix = float((small > 185.0).mean())
    tr_h = float(np.mean(np.abs(np.diff((small > float(small.mean())).astype(np.int8), axis=1))))

    # Calibrated on validation_images to keep text and textless classes separable.
    z_dark = (dark_pix - 0.335113525390625) / 0.2534453210974587
    z_mean = (float(lap.mean()) - 18.86269962086397) / 5.933804487238114
    z_edge_ratio = (edge_ratio - 0.710287982583579) / 0.07591043361376067
    z_edge20 = (edge20 - 0.2532187069163603) / 0.08748948814872878
    z_tr_h = (tr_h - 0.052443984483557214) / 0.02713921427375173
    z_bright = (bright_pix - 0.17881864659926472) / 0.1558409051027704
    raw = (
        0.36388 * (z_dark ** 2)
        + 0.15803 * (z_mean * z_edge_ratio)
        - 0.12188 * (z_edge20 * z_tr_h)
        + 0.22956 * z_bright
        + 0.27073
    )
    glare_penalty = max(0.0, edge20 - 0.45) * max(0.0, bright_pix - 0.18) * 100.0
    raw -= glare_penalty

    # Keep decision margin centered around the default 0.30 OCR threshold.
    score = 0.3 + (raw - 0.519029235) * 0.5
    return max(0.0, min(1.0, float(score)))


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
