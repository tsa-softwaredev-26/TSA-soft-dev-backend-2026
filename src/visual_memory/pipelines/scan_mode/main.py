from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from visual_memory.config import Settings
from visual_memory.pipelines.scan_mode.pipeline import ScanPipeline
from visual_memory.utils import load_image


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run scan mode once and print JSON output.",
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--focal-length-px",
        type=float,
        default=3094.0,
        help="Camera focal length in pixels (default: 3094.0)",
    )
    parser.add_argument(
        "--scan-id",
        default=None,
        help="Optional scan id. If omitted, a UUID is generated.",
    )
    return parser


def main() -> int:
    """Run scan mode once from the CLI and return a process exit code."""
    args = _build_parser().parse_args()
    settings = Settings()
    pipeline = ScanPipeline(
        focal_length_px=args.focal_length_px,
        db_path=Path(settings.db_path),
    )
    image = load_image(args.image)
    scan_id = args.scan_id or str(uuid.uuid4())
    result = pipeline.run(image, scan_id=scan_id, focal_length_px=args.focal_length_px)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
