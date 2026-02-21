"""
DepthEstimator test suite — permanent, run when tuning thresholds or depth logic.

Tests edge cases before scan_mode integration:
  - Normal high/low/below-threshold confidence narration
  - No detection handling
  - Clock position at image extremes
  - Close range accuracy (1ft — known higher error)
  - Angled camera (floor images — straight-line > measured horizontal)
  - Small object at distance (airpods 6ft — low confidence expected)

Run from project root:
    python depth_estimation/test_estimator.py

Images in depth_estimation/test_images/ — gitignored, same set as depth_demo tests.
Focal length hardcoded to iPhone 15 Plus for testing.
# TODO: replace with focal length passed from Android camera API in production
"""

import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent.parent))

from depth_estimation import DepthEstimator
from object_detection import GroundingDinoDetector
from utils import load_image

IMAGES_DIR = Path(__file__).parent / "test_images"

# iPhone 15 Plus — winner from depth_demo calibration tests
IPHONE_15_PLUS_F_PX = 3094.0

CONFIDENCE_HIGH = 0.6
CONFIDENCE_LOW  = 0.4

# (filename, prompt, ground_truth_ft, expected_confidence_range)
# expected_confidence_range: "high", "mid", "low", "none" (no detection expected)
EDGE_CASES = [
    # Normal cases — large clear object close up
    ("mouse_1ft_table.jpg",    "computer mouse",        1.0,  "high"),
    ("mouse_3ft_table.jpg",    "computer mouse",        3.0,  "high"),

    # Small object — airpods lose confidence at distance
    ("airpods_1ft_table.jpg",  "small round earbud case", 1.0, "high"),
    ("airpods_6ft_table.jpg",  "small round earbud case", 6.0, "mid"),  # expected low conf

    # Below threshold — wallet at 6ft was weakest performer in depth_demo
    ("wallet_6ft_table.jpg",   "wallet",                6.0,  "low"),   # may drop to none

    # Angled camera — floor images, straight-line > measured horizontal distance
    # Expect higher depth error than table equivalents
    ("mouse_1ft_floor.jpg",    "computer mouse",        1.0,  "high"),
    ("airpods_1ft_floor.jpg",  "small round earbud case", 1.0, "high"),
    ("wallet_1ft_floor.jpg",   "wallet",                1.0,  "high"),

    # Clock position extremes — object should be off-center
    ("mouse_6ft_floor.jpg",    "computer mouse",        6.0,  "mid"),
]


def confidence_band(score: float) -> str:
    if score >= CONFIDENCE_HIGH:
        return "high"
    elif score >= CONFIDENCE_LOW:
        return "mid"
    else:
        return "low"


def run_tests():
    print("Loading DepthEstimator...")
    estimator = DepthEstimator(focal_length_px=IPHONE_15_PLUS_F_PX)
    detector  = GroundingDinoDetector()

    results = []

    for filename, prompt, gt_ft, expected_conf in EDGE_CASES:
        img_path = IMAGES_DIR / filename
        if not img_path.exists():
            print(f"  SKIP {filename} — not found")
            continue

        print(f"\n[{filename}]")
        image = load_image(str(img_path))
        detection = detector.detect(image, prompt)

        if not detection:
            narration = None
            print(f"  No detection  (expected: {expected_conf})")
            results.append({
                "file": filename, "detected": False,
                "expected_conf": expected_conf, "pass": expected_conf == "none"
            })
            continue

        box        = detection["box"]
        similarity = detection["score"]
        img_w, img_h = image.width, image.height

        depth_map   = estimator.estimate(image)
        distance_ft = estimator.get_depth_at_bbox(depth_map, box)
        clock       = estimator.get_clock_position(box, img_w, img_h)
        narration   = estimator.build_narration(detection["label"], clock, distance_ft, similarity)

        gt_err     = abs(distance_ft - gt_ft) / gt_ft * 100
        actual_conf = confidence_band(similarity)
        conf_pass  = actual_conf == expected_conf
        narr_pass  = (narration is not None) if expected_conf != "low" else True

        print(f"  conf={similarity:.3f} ({actual_conf})  expected={expected_conf}  {'✓' if conf_pass else '✗'}")
        print(f"  depth={distance_ft:.2f}ft  gt={gt_ft}ft  error={gt_err:.1f}%")
        print(f"  clock={clock}")
        print(f"  narration={narration}")

        results.append({
            "file": filename,
            "detected": True,
            "similarity": similarity,
            "actual_conf": actual_conf,
            "expected_conf": expected_conf,
            "conf_pass": conf_pass,
            "distance_ft": distance_ft,
            "gt_ft": gt_ft,
            "error_pct": gt_err,
            "clock": clock,
            "narration": narration,
        })

    # Summary
    detected  = [r for r in results if r["detected"]]
    passed    = [r for r in detected if r.get("conf_pass")]
    avg_err   = sum(r["error_pct"] for r in detected) / len(detected) if detected else 0

    print("\n" + "=" * 60)
    print(f"Confidence band correct : {len(passed)}/{len(detected)}")
    print(f"Average distance error  : {avg_err:.1f}%")
    print(f"No detections           : {sum(1 for r in results if not r['detected'])}")

    print("\nClock positions:")
    for r in detected:
        print(f"  {r['file']:<30} {r['clock']}")


if __name__ == "__main__":
    run_tests()
