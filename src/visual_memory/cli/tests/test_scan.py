"""
CLI test for ScanPipeline.

Usage:
    python -m visual_memory.cli.tests.test_scan <image_path> [--db <database_dir>] [--focal <f_px>]

Defaults:
    --db      src/visual_memory/cli/demo_database (demo embeddings folder)
    --focal   3094.0  (iPhone 15 Plus focal length in pixels)

Example:
    python -m visual_memory.cli.tests.test_scan src/visual_memory/cli/input_images/wallet_3ft_table.jpg
"""

import sys
import json
import argparse
from pathlib import Path

# Resolve project root so imports work regardless of cwd
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

import pillow_heif
pillow_heif.register_heif_opener()

from visual_memory.utils.image_utils import load_image
from visual_memory.pipelines.scan_mode.pipeline import ScanPipeline
from visual_memory.config.paths import DEMO_DATABASE_DIR, FOCAL_LENGTH_PX_IPHONE15

DEFAULT_DB = DEMO_DATABASE_DIR
DEFAULT_FOCAL = FOCAL_LENGTH_PX_IPHONE15


def parse_args():
    parser = argparse.ArgumentParser(description="Test ScanPipeline from CLI.")
    parser.add_argument("image_path", type=str, help="Path to query image.")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB), help="Path to database folder.")
    parser.add_argument("--focal", type=float, default=DEFAULT_FOCAL, help="Focal length in pixels.")
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
