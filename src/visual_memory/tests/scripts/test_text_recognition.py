"""
CLI test for TextRecognizer and CLIPEmbedder (text embedding).

Usage:
    python -m visual_memory.tests.scripts.test_text_recognition [image_path]

If no image_path is given, uses magnesium.heic (supplement bottle with visible text).
"""

import sys
import json
import argparse
from pathlib import Path

import pillow_heif
pillow_heif.register_heif_opener()

from visual_memory.engine.text_recognition import TextRecognizer
from visual_memory.engine.embedding import CLIPTextEmbedder
from visual_memory.utils import load_image


SCRIPTS_DIR = Path(__file__).resolve().parent   # tests/scripts/
TESTS_DIR = SCRIPTS_DIR.parent                  # tests/
INPUT_DIR = TESTS_DIR / "input_images"
DEFAULT_IMAGE = INPUT_DIR / "magnesium.heic"


def run_text_recognition_tests(image_path: Path) -> dict:
    """
    Run text recognition tests on the given image.

    Returns dict with keys: recognizer_ok, embedder_ok, errors
    """
    results = {"recognizer_ok": False, "embedder_ok": False, "errors": []}

    image = load_image(str(image_path))

    # Test 1: TextRecognizer
    try:
        recognizer = TextRecognizer()
        ocr = recognizer.recognize(image)
        assert isinstance(ocr, dict), "recognize() must return a dict"
        assert "text" in ocr and "confidence" in ocr and "segments" in ocr, "Missing keys in result"
        results["recognizer_ok"] = True
        results["ocr_text"] = ocr["text"]
        results["ocr_confidence"] = ocr["confidence"]
        results["segment_count"] = len(ocr["segments"])
    except Exception as e:
        results["errors"].append(f"TextRecognizer: {e}")
        return results

    # Test 2: CLIPTextEmbedder text embedding
    try:
        embedder = CLIPTextEmbedder()
        sample_text = ocr["text"] if ocr["text"] else "test document"
        embedding = embedder.embed_text(sample_text)
        assert embedding.shape[0] == 1, f"Expected batch dim 1, got {embedding.shape[0]}"
        assert embedding.shape[1] == 512, f"Expected 512-dim embedding, got {embedding.shape[1]}"
        results["embedder_ok"] = True
        results["embedding_shape"] = list(embedding.shape)
    except Exception as e:
        results["errors"].append(f"CLIPTextEmbedder: {e}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Test TextRecognizer and CLIPEmbedder.")
    parser.add_argument("image_path", nargs="?", default=str(DEFAULT_IMAGE), help="Path to image.")
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image_path)

    if not image_path.exists():
        print(json.dumps({"error": f"Image not found: {image_path}"}))
        sys.exit(1)

    results = run_text_recognition_tests(image_path)
    print(json.dumps(results, indent=2))

    if results["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
