"""
CLI test for RememberPipeline.

Usage:
    python -m visual_memory.tests.scripts.test_remember <image_path> <prompt>

Example:
    python -m visual_memory.tests.scripts.test_remember src/visual_memory/tests/input_images/Wallet.heic "wallet"
"""

import sys
import json
import argparse
from pathlib import Path

import pillow_heif
pillow_heif.register_heif_opener()

from visual_memory.pipelines.remember_mode.pipeline import RememberPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Test RememberPipeline from CLI.")
    parser.add_argument("image_path", type=str, help="Path to image.")
    parser.add_argument("prompt", type=str, help="Text prompt describing the object.")
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(json.dumps({"error": f"Image not found: {image_path}"}))
        sys.exit(1)

    pipeline = RememberPipeline()
    result = pipeline.run(image_path=image_path, prompt=args.prompt)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
