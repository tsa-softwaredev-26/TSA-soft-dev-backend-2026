"""
Image utility functions for loading and processing images.
"""

from typing import List, Tuple, Optional
from pathlib import Path
from PIL import Image, ImageOps


def crop_object(image: Image.Image, box: List[float]) -> Image.Image:
    """
    Crop an object from an image given a bounding box.

    Args:
        image: PIL Image
        box: [x1, y1, x2, y2]

    Returns:
        Cropped PIL Image

    Raises:
        ValueError: If box is invalid
    """

    if len(box) != 4:
        raise ValueError(f"Box must contain 4 coordinates, got {len(box)}")

    try:
        x1, y1, x2, y2 = [int(c) for c in box]
    except Exception:
        raise ValueError(f"Box contains non-numeric values: {box}")

    # Clamp to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.width, x2)
    y2 = min(image.height, y2)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bounding box after clamping: {box}")

    return image.crop((x1, y1, x2, y2))


def load_image(file_path: str) -> Image.Image:
    """
    Load a single image from disk.

    - Fixes EXIF orientation
    - Converts to RGB
    - Raises errors if loading fails

    Raises:
        FileNotFoundError
        RuntimeError
    """

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")

    except Exception as e:
        raise RuntimeError(f"Failed to load image '{file_path}': {e}")


def load_folder_images(folder_path: str) -> List[Tuple[str, Image.Image]]:
    """
    Load all valid images from a folder.

    Filters:
        - Hidden files
        - macOS metadata files
        - Non-image extensions

    Raises:
        FileNotFoundError
        NotADirectoryError
    """

    path = Path(folder_path)

    if not path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder_path}")

    allowed_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".heic"}

    images: List[Tuple[str, Image.Image]] = []

    for file_path in path.iterdir():

        # Skip hidden + macOS metadata
        if file_path.name.startswith("."):
            continue
        if file_path.name.startswith("._"):
            continue

        if file_path.suffix.lower() not in allowed_extensions:
            continue

        img = load_image(str(file_path))
        images.append((str(file_path), img))

    return images


def refine_crop_with_scan_detector(
    image: Image.Image,
    fallback_box: List[float],
    detector: Optional[object],
) -> tuple[Image.Image, List[float], bool]:
    """
    Refine crop using scan-style prompt-free detector by selecting the best-score box.

    Falls back to fallback_box crop when detector is unavailable or returns no valid box.
    Returns (crop, box_used, refined_flag).
    """
    if detector is None:
        return crop_object(image, fallback_box), fallback_box, False
    try:
        batch = detector.detect_all_batch([image])
    except Exception:
        return crop_object(image, fallback_box), fallback_box, False
    if not batch:
        return crop_object(image, fallback_box), fallback_box, False
    boxes, scores = batch[0]
    if not boxes or not scores:
        return crop_object(image, fallback_box), fallback_box, False
    try:
        best_idx = max(range(len(scores)), key=lambda i: float(scores[i]))
        best_box = boxes[best_idx]
        return crop_object(image, best_box), best_box, True
    except Exception:
        return crop_object(image, fallback_box), fallback_box, False
