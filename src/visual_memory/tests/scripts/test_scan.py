"""
CLI test for ScanPipeline.

Usage:
    python -m visual_memory.tests.scripts.test_scan <image_path> [--db <db_path>] [--focal <f_px>]

Defaults:
    --db      data/memory.db (project-root SQLite database)
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
_PROJECT_ROOT = _TESTS_DIR.parents[2]            # project root

DEFAULT_DB_PATH = _PROJECT_ROOT / "data" / "memory.db"
FOCAL_LENGTH_PX_IPHONE15 = 3094.0


def parse_args():
    parser = argparse.ArgumentParser(description="Test ScanPipeline from CLI.")
    parser.add_argument("image_path", type=str, help="Path to query image.")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH), help="Path to SQLite database file.")
    parser.add_argument("--focal", type=float, default=FOCAL_LENGTH_PX_IPHONE15, help="Focal length in pixels.")
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = Path(args.image_path)

    if not image_path.exists():
        print(json.dumps({"error": f"Image not found: {image_path}"}))
        sys.exit(1)

    image = load_image(str(image_path))

    pipeline = ScanPipeline(focal_length_px=args.focal, db_path=args.db)
    result = pipeline.run(image)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
