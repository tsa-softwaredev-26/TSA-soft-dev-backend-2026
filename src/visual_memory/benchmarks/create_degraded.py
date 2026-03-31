"""Create degraded dataset variants with controlled quality degradation.

Phase 6: Generate degraded images and corresponding CSV for studying performance
degradation curves across image quality levels (blur, compression, noise).

Usage:
    python -m visual_memory.benchmarks.create_degraded \
        --dataset benchmarks/dataset.csv \
        --images benchmarks/images \
        --output benchmarks/degraded \
        --degradation-levels blur:1-5 compression:30-90 noise:0.01-0.1

This creates:
- benchmarks/degraded/ directory with degraded image variants
- benchmarks/dataset_degraded.csv with mapping to degraded images
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import cv2


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_BENCHMARKS_DIR = _PROJECT_ROOT / "benchmarks"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create degraded image dataset variants")
    p.add_argument("--dataset", type=Path, default=_BENCHMARKS_DIR / "dataset.csv",
                   help="Input dataset CSV")
    p.add_argument("--images", type=Path, default=_BENCHMARKS_DIR / "images",
                   help="Input images directory")
    p.add_argument("--output", type=Path, default=_BENCHMARKS_DIR / "degraded",
                   help="Output directory for degraded images and CSV")
    p.add_argument("--blur-levels", type=str, default="1,2,3,4,5",
                   help="Comma-separated blur kernel sizes")
    p.add_argument("--compression-levels", type=str, default="30,50,70,90",
                   help="Comma-separated JPEG quality levels")
    p.add_argument("--noise-levels", type=str, default="0.01,0.02,0.05,0.1",
                   help="Comma-separated Gaussian noise std devs")
    p.add_argument("--dry-run", action="store_true",
                   help="Parse args only, don't write files")
    return p.parse_args()


def _load_dataset(csv_path: Path) -> List[dict]:
    """Load dataset CSV and parse rows."""
    rows = []
    with open(csv_path, newline="") as f:
        filtered = (line for line in f if not line.lstrip().startswith("#"))
        for row in csv.DictReader(filtered):
            rows.append({
                "image": row["image"],
                "label": row["label"],
                "distance_ft": float(row["ground_truth_distance_ft"]),
                "dino_prompt": row["dino_prompt"],
            })
    return rows


def _apply_blur(img: Image.Image, kernel_size: int) -> Image.Image:
    """Apply Gaussian blur with specified kernel size."""
    if kernel_size <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=kernel_size))


def _apply_compression(img_path: Path, quality: int) -> Image.Image:
    """Load image and compress to specified JPEG quality."""
    img = Image.open(img_path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    # Simulate compression by save/load cycle
    import io
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def _apply_noise(img: Image.Image, std: float) -> Image.Image:
    """Add Gaussian noise to image."""
    if std <= 0:
        return img
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, std * 255, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def create_degraded_variants(args: argparse.Namespace) -> Tuple[int, int]:
    """Create degraded image variants and CSV mapping."""
    args.output.mkdir(parents=True, exist_ok=True)
    
    dataset = _load_dataset(args.dataset)
    blur_levels = [int(x.strip()) for x in args.blur_levels.split(",")]
    compression_levels = [int(x.strip()) for x in args.compression_levels.split(",")]
    noise_levels = [float(x.strip()) for x in args.noise_levels.split(",")]
    
    if args.dry_run:
        print(f"Would create {len(dataset)} base images × "
              f"({len(blur_levels)} blur + {len(compression_levels)} compression + {len(noise_levels)} noise) variants")
        return 0, 0
    
    # Create degraded CSV with all variants
    degraded_rows = []
    created_count = 0
    
    for row in dataset:
        img_filename = row["image"]
        img_path = args.images / img_filename
        
        if not img_path.exists():
            print(f"Warning: {img_path} not found, skipping")
            continue
        
        base_img = Image.open(img_path)
        if base_img.mode == 'RGBA':
            base_img = base_img.convert('RGB')
        
        # Blur variants
        for blur_k in blur_levels:
            blurred = _apply_blur(base_img, blur_k)
            out_name = f"{img_filename[:-4]}_blur{blur_k}.jpg"
            out_path = args.output / out_name
            blurred.save(out_path, quality=95)
            degraded_rows.append({
                "image": out_name,
                "label": row["label"],
                "distance_ft": row["distance_ft"],
                "dino_prompt": row["dino_prompt"],
                "degradation_type": "blur",
                "degradation_param": str(blur_k),
            })
            created_count += 1
        
        # Compression variants
        for quality in compression_levels:
            compressed = _apply_compression(img_path, quality)
            out_name = f"{img_filename[:-4]}_comp{quality}.jpg"
            out_path = args.output / out_name
            compressed.save(out_path, quality=quality)
            degraded_rows.append({
                "image": out_name,
                "label": row["label"],
                "distance_ft": row["distance_ft"],
                "dino_prompt": row["dino_prompt"],
                "degradation_type": "compression",
                "degradation_param": str(quality),
            })
            created_count += 1
        
        # Noise variants
        for noise_std in noise_levels:
            noisy = _apply_noise(base_img, noise_std)
            out_name = f"{img_filename[:-4]}_noise{noise_std:.3f}.jpg"
            out_path = args.output / out_name
            noisy.save(out_path, quality=95)
            degraded_rows.append({
                "image": out_name,
                "label": row["label"],
                "distance_ft": row["distance_ft"],
                "dino_prompt": row["dino_prompt"],
                "degradation_type": "noise",
                "degradation_param": f"{noise_std:.3f}",
            })
            created_count += 1
    
    # Write degraded CSV
    csv_path = _BENCHMARKS_DIR / "dataset_degraded.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "label", "distance_ft", "dino_prompt", 
            "degradation_type", "degradation_param"
        ])
        writer.writeheader()
        writer.writerows(degraded_rows)
    
    return created_count, len(degraded_rows)


if __name__ == "__main__":
    args = _parse_args()
    created, total = create_degraded_variants(args)
    print(f"Created {created} degraded images, {total} rows in {args.output / 'dataset_degraded.csv'}")
