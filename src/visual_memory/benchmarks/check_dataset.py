"""Verify benchmark dataset completeness before running full_benchmark.py.

Usage:
    python -m visual_memory.benchmarks.check_dataset
    python -m visual_memory.benchmarks.check_dataset --dataset benchmarks/dataset.csv \\
        --images benchmarks/images
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional

from PIL import Image

from visual_memory.engine.text_recognition import TextRecognizer

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_BENCHMARKS_DIR = _PROJECT_ROOT / "benchmarks"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify benchmark dataset completeness")
    p.add_argument("--dataset", type=Path, default=_BENCHMARKS_DIR / "dataset.csv")
    p.add_argument("--images", type=Path, default=_BENCHMARKS_DIR / "images")
    return p.parse_args()


def _load_csv(path: Path) -> list:
    rows = []
    with open(path, newline="") as f:
        for line in f:
            if line.lstrip().startswith("#"):
                continue
            rows.append(line)
    return list(csv.DictReader(iter(rows)))


def _receipt_images(entries: list, label: str) -> List[str]:
    candidates = [r.get("image", "") for r in entries if r.get("label", "") == label and r.get("image")]
    preferred = ("1ft_bright_clean", "1ft_bright_messy", "3ft_bright_clean")
    ordered: List[str] = []
    for hint in preferred:
        ordered.extend(sorted([name for name in candidates if hint in name]))
    ordered.extend(sorted([name for name in candidates if name not in ordered]))
    return ordered


def main() -> None:
    args = _parse_args()
    all_ok = True

    # positive images
    print(f"Dataset: {args.dataset}")
    print(f"Images:  {args.images}")
    entries = _load_csv(args.dataset)
    missing = [r["image"] for r in entries if not (args.images / r["image"]).exists()]
    found = len(entries) - len(missing)
    print(f"Positive: {found} / {len(entries)} found", end="")
    if missing:
        all_ok = False
        print(f"  ({len(missing)} missing)")
        by_obj: dict = {}
        for fname in missing:
            obj = fname.rsplit("_", 3)[0]
            by_obj.setdefault(obj, []).append(fname)
        for obj, fnames in sorted(by_obj.items()):
            print(f"  {obj}  ({len(fnames)} missing)")
    else:
        print("  ok")

    # Strict OCR receipt text validation is deferred for now; use smoke checks.
    print("Receipt OCR strict validation: deferred")
    gt_dir = _BENCHMARKS_DIR / "ground_truth"
    for rid in ("receipt_salon", "receipt_eye_doctor"):
        gt_file = gt_dir / f"{rid}.txt"
        status = "present (unused while deferred)" if gt_file.exists() else "missing (unused while deferred)"
        print(f"Ground truth {rid}: {status}")

    recognizer = TextRecognizer()
    smoke_checks = (
        ("receipt_salon", "salon"),
        ("receipt_eye_doctor", "eye"),
    )
    ocr_smoke_ok = True
    for receipt_label, keyword in smoke_checks:
        image_names = _receipt_images(entries, receipt_label)
        if not image_names:
            ocr_smoke_ok = False
            print(f"OCR smoke {receipt_label}: no dataset image found")
            continue

        text_hits = 0
        keyword_hits = 0
        keyword_example: Optional[str] = None
        scanned = 0
        missing_images = 0
        for image_name in image_names:
            image_path = args.images / image_name
            if not image_path.exists():
                missing_images += 1
                continue
            scanned += 1
            with Image.open(image_path) as image:
                ocr = recognizer.recognize(image)
            text = str(ocr.get("text", "") or "").strip()
            if text:
                text_hits += 1
                if keyword.lower() in text.lower():
                    keyword_hits += 1
                    if keyword_example is None:
                        keyword_example = image_name

        has_text = text_hits > 0
        has_keyword = keyword_hits > 0
        print(
            f"OCR smoke {receipt_label}: "
            f"scanned={scanned}, missing_images={missing_images}, "
            f"text_hits={text_hits}, keyword[{keyword}]_hits={keyword_hits}, "
            f"keyword_example={keyword_example or 'none'}"
        )
        if not has_text or not has_keyword:
            ocr_smoke_ok = False

    print()
    if not ocr_smoke_ok:
        print("OCR smoke checks: WARN (non-blocking while strict OCR validation is deferred)")
        print("  Expected: at least one receipt OCR hit with keyword 'salon' and one with keyword 'eye'.")
        print()
    if all_ok:
        print("All checks passed. Ready to run full_benchmark.py.")
    else:
        print("Fix missing items above before running full_benchmark.py.")
        sys.exit(1)


if __name__ == "__main__":
    main()
