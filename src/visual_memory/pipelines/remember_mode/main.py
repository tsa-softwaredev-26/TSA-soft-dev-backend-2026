from __future__ import annotations

import argparse
import json
from pathlib import Path

from visual_memory.pipelines.remember_mode.pipeline import RememberPipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run remember mode once and print JSON output.",
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("prompt", help="Label/prompt to teach")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    pipeline = RememberPipeline()
    result = pipeline.run(Path(args.image), args.prompt)
    print(json.dumps(result))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
