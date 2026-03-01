"""
EasyOCR accuracy test on text_demo/ images.

Run:
    python -m visual_memory.tests.scripts.test_ocr_comparison

Runs EasyOCR on every image in tests/text_demo/.
Loads ground truth from tests/text_demo/ground_truth/<stem>.txt if present.

Word-overlap % = |words_predicted ∩ words_ground_truth| / |words_ground_truth|
malarkey.jpeg has no ground truth — raw output only.

Note: DeepSeek-OCR support is available in deepseek_recognizer.py but deferred
(requires GPU + pinned venv). See ARCHITECTURE.md Future Plans.
"""

from __future__ import annotations
from pathlib import Path

import pillow_heif
pillow_heif.register_heif_opener()

from visual_memory.utils import load_image
from visual_memory.engine.text_recognition.recognizer import EasyOCRRecognizer

_SCRIPTS_DIR = Path(__file__).resolve().parent   # tests/scripts/
_TESTS_DIR = _SCRIPTS_DIR.parent                 # tests/
_TEXT_DEMO_DIR = _TESTS_DIR / "text_demo"
_GT_DIR = _TEXT_DEMO_DIR / "ground_truth"

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".heic", ".heif", ".webp"}


def _word_overlap(predicted: str, ground_truth: str) -> float:
    """Return fraction of ground-truth words found in predicted text."""
    gt_words = set(ground_truth.lower().split())
    if not gt_words:
        return 0.0
    pred_words = set(predicted.lower().split())
    return len(gt_words & pred_words) / len(gt_words)


def _load_ground_truth(stem: str) -> str | None:
    gt_path = _GT_DIR / f"{stem}.txt"
    if gt_path.exists():
        return gt_path.read_text(encoding="utf-8").strip()
    return None


def main():
    images = sorted(
        p for p in _TEXT_DEMO_DIR.iterdir()
        if p.suffix.lower() in _IMAGE_EXTENSIONS and not p.name.startswith(".")
    )

    if not images:
        print(f"No test images found in {_TEXT_DEMO_DIR}")
        return

    print("Loading EasyOCR recognizer...")
    easy = EasyOCRRecognizer()

    rows = []
    for img_path in images:
        print(f"\nProcessing {img_path.name} ...")
        img = load_image(str(img_path))
        gt = _load_ground_truth(img_path.stem)

        result = easy.recognize(img)
        words = len(result["text"].split())
        gt_words = len(gt.split()) if gt else None
        overlap = _word_overlap(result["text"], gt) if gt else None

        rows.append({
            "image": img_path.name,
            "words": words,
            "conf": result["confidence"],
            "gt_words": gt_words,
            "overlap_pct": round(overlap * 100, 1) if overlap is not None else None,
        })

        print(f"  EasyOCR ({words:>3} words, conf={result['confidence']:.2f}): {result['text'][:120]}")
        if gt:
            print(f"  Ground truth ({gt_words} words): {gt[:120]}")
            print(f"  Overlap: {overlap*100:.1f}%")
        else:
            print("  (no ground truth)")

    # Summary table
    scored = [r for r in rows if r["overlap_pct"] is not None]
    avg = sum(r["overlap_pct"] for r in scored) / len(scored) if scored else 0.0

    print("\n" + "=" * 60)
    print(f"{'Image':<25} {'Words':>6} {'Conf':>6} {'GT':>5} {'Overlap%':>9}")
    print("-" * 60)
    for r in rows:
        pct = f"{r['overlap_pct']:.1f}" if r["overlap_pct"] is not None else "n/a"
        gt_w = str(r["gt_words"]) if r["gt_words"] is not None else "n/a"
        print(f"{r['image']:<25} {r['words']:>6} {r['conf']:>6.2f} {gt_w:>5} {pct:>9}")
    print("-" * 60)
    print(f"{'Average (scored images)':<25} {'':>6} {'':>6} {'':>5} {avg:>8.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
