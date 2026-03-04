"""Verify that all images listed in dataset.csv are present in the images directory.

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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify benchmark dataset completeness")
    p.add_argument("--dataset", type=Path, default=_BENCHMARKS_DIR / "dataset.csv")
    p.add_argument("--images", type=Path, default=_BENCHMARKS_DIR / "images")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    rows = []
    with open(args.dataset, newline="") as f:
        for line in f:
            if line.lstrip().startswith("#"):
                continue
            rows.append(line)

    reader = csv.DictReader(iter(rows))
    entries = list(reader)

    missing = []
    found = 0
    for row in entries:
        p = args.images / row["image"]
        if p.exists():
            found += 1
        else:
            missing.append(row["image"])

    total = len(entries)
    print(f"Dataset: {args.dataset}")
    print(f"Images:  {args.images}")
    print(f"\nFound:   {found} / {total}")

    if missing:
        print(f"Missing: {len(missing)}\n")
        # Group by object prefix for readability
        by_obj: dict = {}
        for fname in missing:
            obj = fname.rsplit("_", 3)[0]
            by_obj.setdefault(obj, []).append(fname)
        for obj, fnames in sorted(by_obj.items()):
            print(f"  {obj}  ({len(fnames)} missing)")
            for f in fnames:
                print(f"    {f}")
        sys.exit(1)
    else:
        print("\nAll images present. Ready to run full_benchmark.py.")

    # Receipt ground truth check
    gt_dir = _BENCHMARKS_DIR / "ground_truth"
    for rid in ("a", "b"):
        gt_file = gt_dir / f"receipt_{rid}.txt"
        status = "ok" if gt_file.exists() else "MISSING - run redact_receipt.py"
        print(f"  ground_truth/receipt_{rid}.txt: {status}")


if __name__ == "__main__":
    main()
