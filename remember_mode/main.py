from remember_mode.prompt_based_detector import GroundingDinoDetector
from pathlib import Path
CURRENT_DIR = Path(__file__).parent

# Configure your test image and prompt
DEMO_IMAGE_PATH = CURRENT_DIR / "demo_images" / "Wallet.heic"
PROMPT = "camo wallet"

def main():
    """
    Run a demo detection on a sample image.
    
    Change the image_path and prompt variables to test with your own images.
    """
    # Initialize detector with default settings
    # You can customize thresholds here if needed:
    # detector = GroundingDinoDetector(
    #     box_threshold=0.35,  # Lower = more detections but less confident
    #     text_threshold=0.25   # Lower = more lenient text matching
    # )
    detector = GroundingDinoDetector()
    
    
    # Detect object and get bounding box
    detection = detector.detect(DEMO_IMAGE_PATH, PROMPT)
    
    if detection:
        print("\n✓ Detection successful!")
        print(f"  Label: {detection['label']}")
        print(f"  Confidence: {detection['score']:.3f}")
        print(f"  Bounding box: {detection['box']}")
    else:
        print("\n✗ No object detected matching the prompt.")


if __name__ == "__main__":
    main()