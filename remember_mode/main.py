from pathlib import Path
import pillow_heif
from utils import load_image, crop_object
from object_detection import GroundingDinoDetector
from embed_image import ImageEmbedder

pillow_heif.register_heif_opener() # For iphone image support
CURRENT_DIR = Path(__file__).parent

# Configure your test image and prompt
DEMO_IMAGE_PATH = CURRENT_DIR / "demo_images" / "depth_image.png"
PROMPT = "small round earbud case"

def main():
    """
    Run a demo detection on a sample image to get a croppped object using the text prompt above.
    """
    #Customize thresholds here:
    # detector = GroundingDinoDetector(
    #     box_threshold=0.35,  # Lower = more detections but less confident
    #     text_threshold=0.25   # Lower = more lenient text matching
    # )
    detector = GroundingDinoDetector()
    
    image = load_image(DEMO_IMAGE_PATH)

    # Detect object and get bounding box
    detection = detector.detect(image, PROMPT)
    

    #Remove debug print when called from Kotlin
    if detection:
        box = detection['box']
        label = detection['label']
        score = detection['score']
        print("\n Detection successful!")
        print(f"  Confidence: {score:.3f}")
        print(f"  Bounding box: {box}")
        cropped_detection = crop_object(image, box)
        # ImageEmbedder().embed(cropped_detection)
        print("Opening cropped detection...")
        cropped_detection.show()
    else:
        print("\n No object detected matching the prompt.")
        




if __name__ == "__main__":
    main()




