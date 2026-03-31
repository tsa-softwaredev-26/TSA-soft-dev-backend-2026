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

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_BENCHMARKS_DIR = _PROJECT_ROOT / "benchmarks"
_DEFAULT_NEG_IMAGES = _PROJECT_ROOT / "src" / "visual_memory" / "tests" / "input_images"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify benchmark dataset completeness")
    p.add_argument("--dataset", type=Path, default=_BENCHMARKS_DIR / "dataset.csv")
    p.add_argument("--images", type=Path, default=_BENCHMARKS_DIR / "images")
    p.add_argument("--negative-dataset", type=Path,
                   default=_BENCHMARKS_DIR / "negative_dataset.csv")
    p.add_argument("--images-neg", type=Path, default=_DEFAULT_NEG_IMAGES)
    return p.parse_args()


def _load_csv(path: Path) -> list:
    rows = []
    with open(path, newline="") as f:
        for line in f:
            if line.lstrip().startswith("#"):
                continue
            rows.append(line)
    return list(csv.DictReader(iter(rows)))


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

    # negative images
    if args.negative_dataset.exists():
        neg_entries = _load_csv(args.negative_dataset)
        neg_missing = [r["image"] for r in neg_entries
                       if not (args.images_neg / r["image"]).exists()]
        neg_found = len(neg_entries) - len(neg_missing)
        print(f"Negative: {neg_found} / {len(neg_entries)} found", end="")
        if neg_missing:
            all_ok = False
            print(f"  ({len(neg_missing)} missing in {args.images_neg})")
            for m in neg_missing:
                print(f"    {m}")
        else:
            print("  ok")
    else:
        print(f"Negative: negative_dataset.csv not found; skipping")

    # receipt ground truth
    gt_dir = _BENCHMARKS_DIR / "ground_truth"
    for rid in ("receipt_salon", "receipt_eye_doctor"):
        gt_file = gt_dir / f"{rid}.txt"
        status = "ok" if gt_file.exists() else "MISSING; run: python -m visual_memory.benchmarks.redact_receipt"
        print(f"Ground truth {rid}: {status}")
        if not gt_file.exists():
            all_ok = False

    print()
    if all_ok:
        print("All checks passed. Ready to run full_benchmark.py.")
    else:
        print("Fix missing items above before running full_benchmark.py.")
        sys.exit(1)


if __name__ == "__main__":
    main()
