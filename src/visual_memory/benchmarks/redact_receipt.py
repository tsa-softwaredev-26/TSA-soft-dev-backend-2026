"""Receipt image redactor using PaddleOCR for text detection.

Workflow for each receipt (run once per receipt id, a or b):
  1. OCR the 1ft_bright_clean image and print all detected text blocks.
  2. You type the text strings you want redacted (name, card number, etc.).
  3. Script OCRs every image in the receipt's set, finds matching blocks,
     and overwrites them with black rectangles.
  4. You type the ground-truth text (what the receipt says post-redaction).
  5. Script evaluates OCR accuracy on the best image and saves ground truth.

Redacted images are saved in-place (overwrite). Run on COPIES if you want
to keep originals.

Usage:
    python -m visual_memory.benchmarks.redact_receipt a
    python -m visual_memory.benchmarks.redact_receipt b
    python -m visual_memory.benchmarks.redact_receipt a --images-dir /path/to/images
    python -m visual_memory.benchmarks.redact_receipt a --dry-run
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_BENCHMARKS_DIR = _PROJECT_ROOT / "benchmarks"

# Ordered from best to worst quality — used to pick the reference image for
# showing detected text and for final OCR accuracy evaluation.
_CONDITION_ORDER = [
    "1ft_bright_clean",
    "1ft_bright_messy",
    "3ft_bright_clean",
    "1ft_dim_clean",
    "3ft_bright_messy",
    "1ft_dim_messy",
    "3ft_dim_clean",
    "3ft_dim_messy",
    "6ft_bright_clean",
    "6ft_bright_messy",
    "6ft_dim_clean",
    "6ft_dim_messy",
]


# ---- PaddleOCR with bounding boxes ----

def _suppress_paddle_logs() -> None:
    for name in ("ppocr", "paddle", "paddleocr", "PaddleOCR"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _load_ocr():
    _suppress_paddle_logs()
    from paddleocr import PaddleOCR
    return PaddleOCR(use_angle_cls=True, lang="en", show_log=False)


def _ocr_boxes(ocr, img_path: str) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
    """Run OCR on an image. Returns list of (text, confidence, (x1,y1,x2,y2))."""
    raw = ocr.ocr(img_path, cls=True)
    if not raw:
        return []
    # raw may be list-of-pages (paddleocr 2.x) or direct list (3.x)
    page = raw[0] if (raw and isinstance(raw[0], list)) else raw
    if page is None:
        return []
    results = []
    for line in page:
        if not line:
            continue
        try:
            box_pts, (text, conf) = line
        except (TypeError, ValueError):
            continue
        xs = [p[0] for p in box_pts]
        ys = [p[1] for p in box_pts]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        results.append((text, float(conf), (x1, y1, x2, y2)))
    return results


# ---- Text matching ----

def _normalize(s: str) -> str:
    return re.sub(r"\s+", "", s.lower())


def _is_match(ocr_text: str, query: str) -> bool:
    """Return True if query appears in ocr_text (normalized substring match)."""
    return _normalize(query) in _normalize(ocr_text)


def _find_boxes_to_redact(
    blocks: List[Tuple[str, float, Tuple[int, int, int, int]]],
    queries: List[str],
) -> List[Tuple[int, int, int, int]]:
    """Return all bounding boxes from blocks whose text matches any query."""
    redact_boxes = []
    for text, conf, box in blocks:
        for q in queries:
            if _is_match(text, q):
                redact_boxes.append(box)
                break
    return redact_boxes


# ---- Image redaction ----

def _apply_redaction(img_path: Path, boxes: List[Tuple[int, int, int, int]]) -> None:
    """Overwrite each bounding box with a filled black rectangle."""
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for (x1, y1, x2, y2) in boxes:
        # Expand by 4px on each side to cover any sub-pixel antialiasing
        draw.rectangle(
            [max(0, x1 - 4), max(0, y1 - 4), x2 + 4, y2 + 4],
            fill=(0, 0, 0),
        )
    img.save(img_path, quality=95)


# ---- Word overlap (OCR accuracy) ----

def _word_overlap(pred: str, ground_truth: str) -> float:
    pred_words = set(pred.lower().split())
    gt_words = set(ground_truth.lower().split())
    if not gt_words:
        return 0.0
    return len(pred_words & gt_words) / len(gt_words)


# ---- Interactive prompts ----

def _prompt_queries() -> List[str]:
    print()
    print("Enter the text you want redacted. Each line is one search string.")
    print("The script will black out any OCR block containing that string")
    print("(case-insensitive, whitespace-insensitive).")
    print("Examples:  John Smith   |   4242   |   555-867-5309")
    print("Enter a blank line when done.")
    queries = []
    while True:
        line = input("  Redact> ").strip()
        if not line:
            break
        queries.append(line)
    return queries


def _prompt_ground_truth() -> str:
    print()
    print("Enter the ground-truth text for this receipt (after redaction).")
    print("This is saved to benchmarks/ground_truth/ and used to score OCR accuracy.")
    print("Tip: paste what you can actually read on the receipt. Use newlines for line breaks.")
    print("Enter 'END' on its own line when done.")
    lines = []
    while True:
        line = input("  GT> ")
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


# ---- Main ----

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Redact personal info from receipt images")
    p.add_argument("receipt_id", choices=["a", "b"],
                   help="Which receipt to process (a or b)")
    p.add_argument("--images-dir", type=Path,
                   default=_BENCHMARKS_DIR / "images",
                   help="Directory containing benchmark images")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be redacted without modifying images")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    receipt_id = f"receipt_{args.receipt_id}"
    images_dir = args.images_dir

    print(f"\nReceipt: {receipt_id}")
    print(f"Images dir: {images_dir}")

    # --- Locate all 12 images for this receipt ---
    all_images: List[Path] = []
    missing: List[str] = []
    for cond in _CONDITION_ORDER:
        fname = f"{receipt_id}_{cond}.jpg"
        p = images_dir / fname
        if p.exists():
            all_images.append(p)
        else:
            missing.append(fname)

    if not all_images:
        print(f"\nNo images found for {receipt_id} in {images_dir}")
        print("Expected filenames like: receipt_a_1ft_bright_clean.jpg")
        sys.exit(1)

    if missing:
        print(f"\n[warn] {len(missing)} images not found (will be skipped):")
        for m in missing:
            print(f"  {m}")

    print(f"\nFound {len(all_images)} images to process.")

    # --- Step 1: Show OCR results from the best available image ---
    best_img = all_images[0]
    print(f"\nRunning OCR on reference image: {best_img.name} ...")
    print("(Loading PaddleOCR — this may take a moment on first run)")

    ocr = _load_ocr()
    ref_blocks = _ocr_boxes(ocr, str(best_img))

    if not ref_blocks:
        print("[warn] OCR detected no text in reference image.")
        print("Check that the image is clear and the receipt text is readable.")
    else:
        print(f"\nDetected {len(ref_blocks)} text blocks:\n")
        print(f"  {'#':>3}  {'Conf':>5}  Text")
        print(f"  {'-'*3}  {'-'*5}  {'-'*50}")
        for i, (text, conf, _box) in enumerate(ref_blocks):
            print(f"  {i:>3}  {conf:.2f}   {text}")

    # --- Step 2: Collect strings to redact ---
    queries = _prompt_queries()
    if not queries:
        print("No redaction strings entered. Exiting.")
        sys.exit(0)

    # Show what would be matched in the reference image
    preview_boxes = _find_boxes_to_redact(ref_blocks, queries)
    if not preview_boxes:
        print(f"\n[warn] None of the entered strings matched any OCR block")
        print("in the reference image. Check spelling and try again.")
        cont = input("Continue anyway? (y/N) ").strip().lower()
        if cont != "y":
            sys.exit(0)
    else:
        print(f"\nWill redact {len(preview_boxes)} block(s) in reference image.")

    if args.dry_run:
        print("\n[dry-run] No files modified.")
        print("Blocks that would be redacted in reference image:")
        for text, conf, box in ref_blocks:
            for q in queries:
                if _is_match(text, q):
                    print(f"  '{text}' at {box}")
                    break
        sys.exit(0)

    # --- Step 3: Process all images ---
    print(f"\nProcessing {len(all_images)} images...\n")
    total_redacted = 0
    skipped = []

    for img_path in all_images:
        blocks = _ocr_boxes(ocr, str(img_path))
        boxes = _find_boxes_to_redact(blocks, queries)
        if boxes:
            _apply_redaction(img_path, boxes)
            print(f"  [ok]   {img_path.name}  ({len(boxes)} block(s) redacted)")
            total_redacted += len(boxes)
        else:
            skipped.append(img_path.name)
            print(f"  [skip] {img_path.name}  (no matching text detected)")

    print(f"\nDone. {total_redacted} block(s) redacted across {len(all_images)} images.")
    if skipped:
        print(f"\n[warn] No matches found in {len(skipped)} image(s):")
        for s in skipped:
            print(f"  {s}")
        print("These images may need manual review (distant/dim shots with poor OCR).")

    # --- Step 4: Ground truth text ---
    ground_truth = _prompt_ground_truth()

    gt_dir = _BENCHMARKS_DIR / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_path = gt_dir / f"{receipt_id}.txt"
    gt_path.write_text(ground_truth, encoding="utf-8")
    print(f"\nGround truth saved: {gt_path}")

    # --- Step 5: OCR accuracy on best image (post-redaction) ---
    if ground_truth.strip():
        print(f"\nEvaluating OCR accuracy on: {best_img.name}")
        post_blocks = _ocr_boxes(ocr, str(best_img))
        pred_text = " ".join(t for t, _c, _b in post_blocks)
        score = _word_overlap(pred_text, ground_truth)
        print(f"\nOCR word overlap (post-redaction): {score:.1%}")
        print(f"  Predicted: {pred_text[:200]}{'...' if len(pred_text) > 200 else ''}")
        gt_preview = ground_truth[:200] + ("..." if len(ground_truth) > 200 else "")
        print(f"  Expected:  {gt_preview}")

    print(f"\nDone. To run the full benchmark:")
    print(f"  python -m visual_memory.benchmarks.full_benchmark \\")
    print(f"      --dataset benchmarks/dataset.csv \\")
    print(f"      --images benchmarks/images")


if __name__ == "__main__":
    main()
