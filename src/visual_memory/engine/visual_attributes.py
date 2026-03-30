from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _dominant_color_name(rgb: np.ndarray) -> str:
    r, g, b = [int(v) for v in rgb]
    spread = max(r, g, b) - min(r, g, b)
    peak = max(r, g, b)

    if spread < 28:
        if peak < 70:
            return "black"
        if peak > 205:
            return "white"
        return "gray"

    if r > 170 and g > 150 and b < 115:
        return "yellow"
    if r > 150 and b > 145 and g < 125:
        return "purple"
    if r > g and r > b:
        return "red" if r > 140 else "brown"
    if g > r and g > b:
        return "green"
    if b > r and b > g:
        return "blue"
    return "multicolor"


def _shape_from_aspect_ratio(aspect_ratio: float) -> str:
    if aspect_ratio > 1.3:
        return "wide"
    if aspect_ratio < 0.8:
        return "tall"
    return "square"


def _texture_from_laplacian(gray: np.ndarray) -> str:
    gray_f = gray.astype(np.float32)
    lap = (
        np.roll(gray_f, 1, 0)
        + np.roll(gray_f, -1, 0)
        + np.roll(gray_f, 1, 1)
        + np.roll(gray_f, -1, 1)
        - 4.0 * gray_f
    )
    variance = float(lap.var())
    if variance < 100.0:
        return "smooth"
    if variance < 500.0:
        return "detailed"
    return "textured"


def extract_visual_attributes(image_path: str | Path) -> dict:
    """Extract low-cost visual attributes for description fallback."""
    path = Path(image_path)
    image = Image.open(path).convert("RGB")
    rgb = np.array(image)

    if rgb.size == 0:
        return {}

    avg_rgb = rgb.reshape(-1, 3).mean(axis=0)
    height, width = rgb.shape[:2]
    aspect_ratio = float(width / max(height, 1))
    gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    return {
        "dominant_color": _dominant_color_name(avg_rgb),
        "shape": _shape_from_aspect_ratio(aspect_ratio),
        "texture": _texture_from_laplacian(gray),
        "aspect_ratio": round(aspect_ratio, 2),
    }


def describe_from_attributes(label: str, attributes: dict) -> str:
    if not attributes:
        return f"This is your {label}."

    color = attributes.get("dominant_color")
    shape = attributes.get("shape")
    texture = attributes.get("texture")

    parts = [f"This is your {label}."]
    if color:
        parts.append(f"It is mostly {color}.")
    if shape:
        parts.append(f"It has a {shape} shape.")
    if texture:
        parts.append(f"It looks {texture}.")
    return " ".join(parts)
