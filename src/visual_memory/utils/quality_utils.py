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
