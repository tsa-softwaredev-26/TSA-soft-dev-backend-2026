import os
from PIL import Image


def crop_object(image, box):
    """
    Crop an object from an image given a bounding box.
    
    Args:
        image: PIL Image object
        box: Bounding box as (x1, y1, x2, y2)
        
    Returns:
        Cropped PIL Image
    """
    x1, y1, x2, y2 = [int(coord) for coord in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, image.width), min(y2, image.height)
    return image.crop((x1, y1, x2, y2))

def load_image(file_path):
    """
    Load a single image from a given path.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        PIL Image object or None if loading fails
    """
    if not os.path.isfile(file_path):
        print(f"Warning: File '{file_path}' does not exist.")
        return None
    
    try:
        img = Image.open(file_path).convert("RGB")
        return img
    except Exception as e:
        print(f"Warning: Could not load '{file_path}': {e}")
        return None


def load_folder_images(folder_path):
    """
    Load all images from a folder.
    
    Args:
        folder_path: Folder containing images        
    Returns:
        List of tuples: (file_path, PIL Image)
    """

    if not os.path.exists(folder_path):
        print(f"Warning: Folder '{folder_path}' not found.")
        return []

    images = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        _, ext = os.path.splitext(filename.lower())
        img = load_image(file_path)
        if img is not None:
            images.append((file_path, img))
    return images
