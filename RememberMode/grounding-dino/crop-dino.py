import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import pillow_heif
from pathlib import Path

# Register HEIF opener with PIL
pillow_heif.register_heif_opener()


def load_image_safe(image_path):
    """
    Safely load an image from any format (HEIC, JPG, PNG, etc.) and convert to RGB.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        PIL Image in RGB mode
    """
    img = Image.open(image_path)
    
    # Convert to RGB to prevent color mode issues
    # This handles RGBA, P, L, CMYK, etc.
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    return img


def crop_object(image, box):
    """
    Crop an object from an image given a bounding box.
    
    Args:
        image: PIL Image object
        box: Bounding box as [x1, y1, x2, y2] in pixel coordinates
        
    Returns:
        Cropped PIL Image
    """
    x1, y1, x2, y2 = [int(coord) for coord in box]
    
    # Ensure coordinates are within image bounds
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, image.width), min(y2, image.height)
    
    return image.crop((x1, y1, x2, y2))


def detect_and_crop_with_dino(
    image_path, 
    prompt,
    model_id="IDEA-Research/grounding-dino-tiny",
    box_threshold=0.4,
    text_threshold=0.3,
    output_path=None
):
    """
    Detect a prompted object in an image using Grounding DINO and crop it.
    
    Args:
        image_path: Path to the input image (supports HEIC, JPG, PNG, etc.)
        prompt: Text prompt for the object to detect (e.g., "cat", "dog", "person")
        model_id: Hugging Face model ID for Grounding DINO
        box_threshold: Confidence threshold for detection
        text_threshold: Text similarity threshold
        output_path: Optional path to save the cropped image. If None, auto-generates filename.
        
    Returns:
        PIL Image of the cropped object, or None if no detection
    """
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and processor
    print("Loading Grounding DINO model...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    
    # Load and process the image
    print(f"Loading image: {image_path}")
    image = load_image_safe(image_path)
    
    # Prepare text labels (wrap in list as expected by the model)
    text_labels = [[prompt]]
    
    # Run detection
    print(f"Detecting '{prompt}'...")
    inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]]
    )
    
    result = results[0]
    
    if len(result["boxes"]) == 0:
        print(f"No '{prompt}' detected in the image.")
        return None
    
    # Get the best detection (highest confidence)
    best_idx = result["scores"].argmax()
    best_box = result["boxes"][best_idx]
    best_score = result["scores"][best_idx].item()
    best_label = result["labels"][best_idx]
    
    print(f"Detected '{best_label}' with confidence: {best_score:.3f}")
    print(f"Bounding box: {[round(x, 2) for x in best_box.tolist()]}")
    
    # Crop the detected object
    cropped_image = crop_object(image, best_box.tolist())
    
    # Ensure cropped image is in RGB mode
    if cropped_image.mode != 'RGB':
        cropped_image = cropped_image.convert('RGB')
    
    # Generate output path if not provided
    if output_path is None:
        input_path = Path(image_path)
        output_path = input_path.parent / f"cropped_{prompt.replace(' ', '_')}.png"
    
    # Save as PNG or JPG with proper settings
    output_ext = Path(output_path).suffix.lower()
    if output_ext in ['.jpg', '.jpeg']:
        cropped_image.save(output_path, 'JPEG', quality=95)
    else:
        cropped_image.save(output_path, 'PNG')
    
    print(f"Cropped image saved to: {output_path}")
    
    return cropped_image


if __name__ == "__main__":
    # Works with test.png, test.jpg, test.heic, or any image format
    # You can change this to your actual filename
    image_path = "/Users/joeroche/Developer/VisualMemory/test/images/Wallet.heic"  # Change to "test.heic" or "image.jpg" as needed
    
    # Ask user for the prompt
    prompt = "camo wallet"
    
    # Output will be saved in the same directory as the input
    # Format: cropped_<prompt>.png (e.g., cropped_cat.png)
    detect_and_crop_with_dino(image_path, prompt)