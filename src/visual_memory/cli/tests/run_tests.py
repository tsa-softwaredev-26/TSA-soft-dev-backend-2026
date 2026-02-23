"""
Integration test runner for the visual_memory pipelines.

Run:
    python -m visual_memory.cli.run_tests

Tests:
    1. RememberPipeline: wallet_1ft_table.jpg + "small rectangular wallet"
    2. ScanPipeline: wallet_3ft_table.jpg
    3. DepthEstimator: wallet_3ft_table.jpg (ground truth = 3.0 ft)
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import pillow_heif
pillow_heif.register_heif_opener()

from visual_memory.utils import load_image


# Paths (cwd-independent)
TESTS_DIR = Path(__file__).resolve().parent
CLI_DIR = TESTS_DIR.parent
INPUT_DIR = CLI_DIR / "input_images"
DEMO_DB = CLI_DIR / "demo_database"
PROJECT_ROOT = CLI_DIR.parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

WALLET_1FT = INPUT_DIR / "wallet_1ft_table.jpg"
WALLET_3FT = INPUT_DIR / "wallet_3ft_table.jpg"

FOCAL_PX = 3094.0

_G = "\033[32m"
_R = "\033[31m"
_B = "\033[1m"
_X = "\033[0m"

_results: list[tuple[str, bool, str]] = []


def _pass(tag: str, detail: str) -> None:
    _results.append((tag, True, detail))
    print(f"  {_G}PASS{_X}  {detail}")


def _fail(tag: str, detail: str) -> None:
    _results.append((tag, False, detail))
    print(f"  {_R}FAIL{_X}  {detail}")


def _section(title: str) -> None:
    print(f"\n{_B}{title}{_X}")


# Test 1 — RememberPipeline
_section("[1] remember_mode — wallet_1ft_table.jpg")

from visual_memory.pipelines.remember_mode.pipeline import RememberPipeline

remember = RememberPipeline()

r1 = remember.run(
    image_path=WALLET_1FT,
    prompt="small rectangular wallet",
)

print("     json:", json.dumps(r1))

has_box = bool(r1.get("result") and "box" in r1["result"])

if r1["success"] and has_box:
    res = r1["result"]
    _pass(
        "remember:detect",
        f"label={res['label']}  conf={res['confidence']:.3f}"
    )
else:
    _fail("remember:detect", f"success={r1['success']}  result={r1['result']}")


# Test 2 — ScanPipeline
_section("[2] scan_mode — wallet_3ft_table.jpg")

from visual_memory.pipelines.scan_mode.pipeline import ScanPipeline

try:
    scan = ScanPipeline(database_dir=DEMO_DB, focal_length_px=FOCAL_PX)

    r2 = scan.run(load_image(str(WALLET_3FT)))
    print("     json:", json.dumps(r2))

    if r2["count"] >= 1:
        _pass("scan:match", f"count={r2['count']}")
    else:
        _fail("scan:match", "count=0")

except Exception as exc:
    _fail("scan:match", str(exc))


# Test 3 — DepthEstimator
_section("[3] estimator — wallet_3ft_table.jpg")

from visual_memory.engine.depth import DepthEstimator
from visual_memory.engine.object_detection import GroundingDinoDetector

GT_FT = 3.0

estimator = DepthEstimator(focal_length_px=FOCAL_PX)
detector = GroundingDinoDetector()

est_img = load_image(str(WALLET_3FT))

detection = detector.detect(est_img, "small rectangular wallet")

if not detection:
    _fail("estimator:depth", "No detection")
else:
    bbox = detection["box"]
    depth_map = estimator.estimate(est_img)
    dist_ft = estimator.get_depth_at_bbox(depth_map, bbox)

    err_pct = abs(dist_ft - GT_FT) / GT_FT * 100

    print(f"distance: {dist_ft:.2f} ft")
    print(f"error:    {err_pct:.1f}%")

    _pass("estimator:depth", f"dist={dist_ft:.2f} ft  error={err_pct:.1f}%")


# Summary
n_pass = sum(1 for _, ok, _ in _results if ok)
n_fail = len(_results) - n_pass

print()
print(f"{_B}Results:{_X} {n_pass} passed / {n_fail} failed / {len(_results)} total")

for tag, ok, detail in _results:
    icon = f"{_G}✓{_X}" if ok else f"{_R}✗{_X}"
    print(f"  {icon}  {tag:<20}  {detail}")

sys.exit(0 if n_fail == 0 else 1)