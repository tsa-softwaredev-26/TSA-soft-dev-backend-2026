"""
Comprehensive OCR benchmark: EasyOCR vs PaddleOCR

Runs both engines on all test images and compares:
- Word-overlap % (intersection of words vs ground truth)
- Exact match % (case-insensitive)
- Average confidence
- Inference time per image
"""
import time
from pathlib import Path
from PIL import Image

SCRIPTS_DIR = Path(__file__).resolve().parent
TESTS_DIR = SCRIPTS_DIR.parent
PROJECT_ROOT = TESTS_DIR.parents[2]

# Add project root to path
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from visual_memory.utils import get_logger

_log = get_logger(__name__)


def load_ground_truth(image_stem: str) -> str:
    """Load ground truth text for an image."""
    gt_file = TESTS_DIR / "text_demo" / "ground_truth" / f"{image_stem}.txt"
    if not gt_file.exists():
        return None
    return gt_file.read_text().strip()


def word_overlap(recognized_text: str, ground_truth: str) -> float:
    """
    Compute word-overlap percentage.

    intersection_words / total_unique_words
    """
    if not ground_truth or not recognized_text:
        return 0.0

    gt_words = set(ground_truth.lower().split())
    rec_words = set(recognized_text.lower().split())

    if not gt_words and not rec_words:
        return 1.0
    if not gt_words:
        return 0.0

    intersection = len(gt_words & rec_words)
    union = len(gt_words | rec_words)

    if union == 0:
        return 0.0
    return intersection / union


def exact_match(recognized_text: str, ground_truth: str) -> bool:
    """Case-insensitive exact string match."""
    if not ground_truth or not recognized_text:
        return ground_truth == recognized_text
    return recognized_text.lower() == ground_truth.lower()


def run_easyocr(image_path: Path) -> dict:
    """Run EasyOCR on image."""
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False)

    import numpy as np
    img_np = np.array(Image.open(image_path).convert("RGB"))

    start = time.time()
    raw = reader.readtext(img_np)
    elapsed = time.time() - start

    segments = [
        (text, float(conf))
        for (_, text, conf) in raw
        if float(conf) >= 0.3
    ]

    if not segments:
        return {
            "text": "",
            "confidence": 0.0,
            "time": elapsed,
            "segments": []
        }

    texts = [t for t, _ in segments]
    confs = [c for _, c in segments]

    return {
        "text": " ".join(texts),
        "confidence": sum(confs) / len(confs),
        "time": elapsed,
        "segments": segments,
    }


def run_paddleocr(image_path: Path) -> dict:
    """Run PaddleOCR on image."""
    from paddleocr import PaddleOCRVL
    import os
    import numpy as np
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

    pipeline = PaddleOCRVL()
    image = Image.open(image_path).convert("RGB")
    img_np = np.array(image)

    start = time.time()
    results = pipeline.predict(img_np)
    elapsed = time.time() - start

    segments = []
    if results:
        for result in results:
            if hasattr(result, 'text') and hasattr(result, 'confidence'):
                text = result.text
                confidence = float(result.confidence)
                segments.append((text, confidence))

    if not segments:
        return {
            "text": "",
            "confidence": 0.0,
            "time": elapsed,
            "segments": []
        }

    texts = [t for t, _ in segments]
    confs = [c for _, c in segments]

    return {
        "text": " ".join(texts),
        "confidence": sum(confs) / len(confs),
        "time": elapsed,
        "segments": segments,
    }


def main():
    """Run benchmark on test images."""
    test_dir = TESTS_DIR / "text_demo"

    # Test images — uncomment additional images to run full benchmark
    # - marker.jpeg: printed text (main test, fastest)
    # - pen.jpeg: handwritten text
    # - pencil.jpeg: handwritten text
    # - typed.jpeg: typed document text
    test_images = [
        "marker.jpeg",      # Default: printed market signage
        # "pen.jpeg",        # Uncomment for handwritten pen test
        # "pencil.jpeg",     # Uncomment for handwritten pencil test
        # "typed.jpeg",      # Uncomment for typed document test
    ]

    image_files = sorted([
        test_dir / img for img in test_images
        if (test_dir / img).exists()
    ])

    print("\n" + "="*120)
    print("OCR BENCHMARK: EasyOCR vs PaddleOCR")
    print("="*120)

    results = {
        "easyocr": {},
        "paddle": {},
    }

    # Test each image
    for image_path in image_files:
        image_stem = image_path.stem
        gt = load_ground_truth(image_stem)

        print(f"\n--- {image_stem} ---")
        if gt:
            print(f"Ground Truth: {gt[:80]}{'...' if len(gt) > 80 else ''}")
        else:
            print("Ground Truth: (none — control image)")

        # EasyOCR
        print("\n[EasyOCR]")
        try:
            easy_result = run_easyocr(image_path)
            easy_text = easy_result["text"]
            easy_conf = easy_result["confidence"]
            easy_time = easy_result["time"]

            print(f"  Text: {easy_text[:80]}{'...' if len(easy_text) > 80 else ''}")
            print(f"  Confidence: {easy_conf:.4f}")
            print(f"  Time: {easy_time:.2f}s")

            if gt:
                easy_overlap = word_overlap(easy_text, gt)
                easy_exact = exact_match(easy_text, gt)
                print(f"  Word-Overlap: {easy_overlap*100:.1f}%")
                print(f"  Exact Match: {easy_exact}")
                results["easyocr"][image_stem] = {
                    "text": easy_text,
                    "confidence": easy_conf,
                    "time": easy_time,
                    "word_overlap": easy_overlap,
                    "exact_match": easy_exact,
                }
            else:
                results["easyocr"][image_stem] = {
                    "text": easy_text,
                    "confidence": easy_conf,
                    "time": easy_time,
                }
        except Exception as e:
            print(f"  ERROR: {e}")

        # PaddleOCR
        print("\n[PaddleOCR]")
        try:
            paddle_result = run_paddleocr(image_path)
            paddle_text = paddle_result["text"]
            paddle_conf = paddle_result["confidence"]
            paddle_time = paddle_result["time"]

            print(f"  Text: {paddle_text[:80]}{'...' if len(paddle_text) > 80 else ''}")
            print(f"  Confidence: {paddle_conf:.4f}")
            print(f"  Time: {paddle_time:.2f}s")

            if gt:
                paddle_overlap = word_overlap(paddle_text, gt)
                paddle_exact = exact_match(paddle_text, gt)
                print(f"  Word-Overlap: {paddle_overlap*100:.1f}%")
                print(f"  Exact Match: {paddle_exact}")
                results["paddle"][image_stem] = {
                    "text": paddle_text,
                    "confidence": paddle_conf,
                    "time": paddle_time,
                    "word_overlap": paddle_overlap,
                    "exact_match": paddle_exact,
                }
            else:
                results["paddle"][image_stem] = {
                    "text": paddle_text,
                    "confidence": paddle_conf,
                    "time": paddle_time,
                }
        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary table
    print("\n" + "="*120)
    print("SUMMARY TABLE")
    print("="*120)

    image_stems = [p.stem for p in image_files if p.stem != "malarkey"]

    print("\nImage | EasyOCR Overlap% | PaddleOCR Overlap% | Winner | EasyOCR Time | PaddleOCR Time")
    print("-" * 100)

    easy_overlaps = []
    paddle_overlaps = []

    for stem in image_stems:
        easy_data = results["easyocr"].get(stem, {})
        paddle_data = results["paddle"].get(stem, {})

        easy_overlap = easy_data.get("word_overlap", None)
        paddle_overlap = paddle_data.get("word_overlap", None)
        easy_time = easy_data.get("time", 0)
        paddle_time = paddle_data.get("time", 0)

        if easy_overlap is not None:
            easy_overlaps.append(easy_overlap)
        if paddle_overlap is not None:
            paddle_overlaps.append(paddle_overlap)

        if easy_overlap is not None and paddle_overlap is not None:
            winner = "PADDLE" if paddle_overlap > easy_overlap else ("EASY" if easy_overlap > paddle_overlap else "TIE")
            print(
                f"{stem:15} | {easy_overlap*100:6.1f}%           | {paddle_overlap*100:6.1f}%            | "
                f"{winner:6} | {easy_time:6.2f}s       | {paddle_time:6.2f}s"
            )
        else:
            print(f"{stem:15} | {'N/A':6} | {'N/A':6} | ERROR")

    # Averages
    print("-" * 100)
    if easy_overlaps:
        easy_avg = sum(easy_overlaps) / len(easy_overlaps)
    else:
        easy_avg = 0
    if paddle_overlaps:
        paddle_avg = sum(paddle_overlaps) / len(paddle_overlaps)
    else:
        paddle_avg = 0

    print(f"{'AVERAGE':15} | {easy_avg*100:6.1f}%           | {paddle_avg*100:6.1f}%            | "
          f"{'PADDLE' if paddle_avg > easy_avg else 'EASY'}")
    print("="*120)

    # Decision
    print("\n[DECISION]")
    threshold = 0.50
    if paddle_avg >= threshold:
        print(f"✓ PaddleOCR WINS: {paddle_avg*100:.1f}% avg >= {threshold*100:.0f}% threshold")
        print("  Recommendation: Proceed to Phase 6 — remove EasyOCR, set paddle as default")
    else:
        print(f"✗ PaddleOCR does not meet threshold: {paddle_avg*100:.1f}% avg < {threshold*100:.0f}%")
        print("  Recommendation: Keep EasyOCR as default, investigate PaddleOCR tuning")


if __name__ == "__main__":
    main()
