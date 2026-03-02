"""
CLI test for ScanPipeline.

Usage:
    python -m visual_memory.tests.scripts.test_scan <image_path> [--db <database_dir>] [--focal <f_px>]

Defaults:
    --db      tests/demo_database (demo embeddings folder)
    --focal   3094.0  (iPhone 15 Plus focal length in pixels)

Example:
    python -m visual_memory.tests.scripts.test_scan src/visual_memory/tests/input_images/wallet_3ft_table.jpg
"""

import sys
import json
import argparse
from pathlib import Path

import pillow_heif
pillow_heif.register_heif_opener()
from visual_memory.utils import load_image
from visual_memory.pipelines.scan_mode.pipeline import ScanPipeline

_SCRIPTS_DIR = Path(__file__).resolve().parent   # tests/scripts/
_TESTS_DIR = _SCRIPTS_DIR.parent                 # tests/

DEMO_DATABASE_DIR = _TESTS_DIR / "demo_database"
FOCAL_LENGTH_PX_IPHONE15 = 3094.0


def parse_args():
    parser = argparse.ArgumentParser(description="Test ScanPipeline from CLI.")
    parser.add_argument("image_path", type=str, help="Path to query image.")
    parser.add_argument("--db", type=str, default=str(DEMO_DATABASE_DIR), help="Path to database folder.")
    parser.add_argument("--focal", type=float, default=FOCAL_LENGTH_PX_IPHONE15, help="Focal length in pixels.")
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = Path(args.image_path)
    database_dir = Path(args.db)

    if not image_path.exists():
        print(json.dumps({"error": f"Image not found: {image_path}"}))
        sys.exit(1)

    if not database_dir.exists():
        print(json.dumps({"error": f"Database directory not found: {database_dir}"}))
        sys.exit(1)

    image = load_image(str(image_path))

    pipeline = ScanPipeline(database_dir=database_dir, focal_length_px=args.focal)
    result = pipeline.run(image)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
