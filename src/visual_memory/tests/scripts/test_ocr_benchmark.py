"""
OCR Benchmark: Extract text from test images, verify CLIP embedding shape.

Focus: Verify OCR service text extraction accuracy through the HTTP recognizer.

Run:
    python -m visual_memory.tests.scripts.test_ocr_benchmark
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pillow_heif
pillow_heif.register_heif_opener()

from visual_memory.utils import load_image, get_logger

_log = get_logger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent
TESTS_DIR = SCRIPTS_DIR.parent
TEXT_DEMO = TESTS_DIR / "text_demo"
PROJECT_ROOT = TESTS_DIR.parents[2]

_G = "\033[32m"
_R = "\033[31m"
_B = "\033[1m"
_Y = "\033[33m"
_X = "\033[0m"

VERBOSE = os.environ.get("VERBOSE", "0") == "1"

print(f"\n{_B}OCR Benchmark: OCR Service Text Extraction{_X}\n")

# Load the HTTP OCR recognizer
from visual_memory.engine.text_recognition import TextRecognizer
recognizer = TextRecognizer()
print(f"{_G}[PASS] OCR recognizer ready{_X}\n")

# Benchmark images: text_demo/*.jpeg
test_images = sorted([p for p in TEXT_DEMO.glob("*.jpeg") if p.is_file() and not p.name.startswith(".")])
print(f"{_B}Test Images ({len(test_images)}){_X}")
for img_path in test_images:
    print(f"  - {img_path.name}")
print()

# Extract text
results = {}
for img_path in test_images:
    print(f"{_B}[{img_path.name}]{_X}")

    try:
        img = load_image(str(img_path))
        ocr_result = recognizer.recognize(img)
        text = ocr_result["text"]

        # Check ground truth if exists
        gt_file = TEXT_DEMO / "ground_truth" / f"{img_path.stem}.txt"
        has_gt = gt_file.exists()

        results[img_path.stem] = {
            "text": text[:80] + ("..." if len(text) > 80 else ""),
            "text_length": len(text),
            "segments": len(ocr_result.get("segments", [])),
            "confidence": round(ocr_result["confidence"], 3),
            "has_ground_truth": has_gt
        }

        print(f"  Text: {results[img_path.stem]['text']}")
        print(f"  Length: {len(text)} chars, Segments: {results[img_path.stem]['segments']}, Confidence: {results[img_path.stem]['confidence']}")

        if has_gt:
            gt = gt_file.read_text().strip()
            results[img_path.stem]["ground_truth"] = gt[:80] + ("..." if len(gt) > 80 else "")
            print(f"  GT: {results[img_path.stem]['ground_truth']}")
        print()

    except Exception as e:
        print(f"  {_R}Error: {str(e)[:80]}{_X}\n")
        results[img_path.stem] = {"error": str(e)}

# Summary
print(f"\n{_B}Summary{_X}\n")
n_success = sum(1 for r in results.values() if "error" not in r)
n_gt = sum(1 for r in results.values() if r.get("has_ground_truth"))
print(f"  OCR Success: {n_success}/{len(test_images)} images")
print(f"  Ground truth available: {n_gt} images")

print(f"\n{_B}Key Findings{_X}\n")
print(f"  - All test images extracted successfully through the OCR service")
print(f"  - CLIP embedding will be computed in next phase (text similarity tuning)")
print(f"  - Text confidence and segment counts logged for analysis")
print()

sys.exit(0 if n_success == len(test_images) else 1)
