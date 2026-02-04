"""Image utility functions for loading and processing images."""
from typing import Optional, List, Tuple
from pathlib import Path
from PIL import Image


def crop_object(image: Image.Image, box: List[float]) -> Image.Image:
    """
    Crop an object from an image given a bounding box.
    
    Args:
        image: PIL Image object
        box: Bounding box as [x1, y1, x2, y2]
        
    Returns:
        Cropped PIL Image
        
    Raises:
        ValueError: If box coordinates are invalid
    """
    if len(box) != 4:
        raise ValueError(f"Box must have 4 coordinates, got {len(box)}")
    
    x1, y1, x2, y2 = [int(coord) for coord in box]
    
    # Clamp coordinates to image bounds
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, image.width), min(y2, image.height)
    
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid box coordinates: {box}")
    
    return image.crop((x1, y1, x2, y2))


def load_image(file_path: str) -> Optional[Image.Image]:
    """
    Load a single image from a given path.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        PIL Image object in RGB mode, or None if loading fails
    """
    path = Path(file_path)
    
    if not path.is_file():
        print(f"Warning: File '{file_path}' does not exist.")
        return None
    
    try:
        img = Image.open(file_path).convert("RGB")
        return img
    except Exception as e:
        print(f"Warning: Could not load '{file_path}': {e}")
        return None


def load_folder_images(folder_path: str) -> List[Tuple[str, Image.Image]]:
    """
    Load all images from a folder.
    
    Args:
        folder_path: Folder containing images
        
    Returns:
        List of tuples: (file_path, PIL Image)
        
    Note:
        Supported formats: .jpg, .jpeg, .png, .heic, .heif
    """
    path = Path(folder_path)
    
    if not path.exists():
        print(f"Warning: Folder '{folder_path}' not found.")
        return []
    
    if not path.is_dir():
        print(f"Warning: '{folder_path}' is not a directory.")
        return []
    
    # Supported image extensions
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.bmp', '.gif', '.webp'}
    
    images = []
    for file_path in path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            img = load_image(str(file_path))
            if img is not None:
                images.append((str(file_path), img))
    
    return images
