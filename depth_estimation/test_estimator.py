"""
DepthEstimator manual test — add your own image paths and run.

Run from project root:
    python depth_estimation/test_estimator.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from depth_estimation import DepthEstimator
from object_detection import GroundingDinoDetector
from utils import load_image

# ── Configure your tests here ────────────────────────────────────────────────
# (image_path, prompt, ground_truth_ft)
# ground_truth_ft is the actual measured distance — used to compute error %
# Set to None if you just want to see the narration without error checking

FOCAL_LENGTH_PX = 3094.0  # iPhone 15 Plus — swap for your device

#Example
TESTS = [
    ("depth_estimation/test_images/mouse_1ft_table.jpg",   "computer mouse",         1.0),
    ("depth_estimation/test_images/mouse_3ft_table.jpg",   "computer mouse",         3.0),
    ("depth_estimation/test_images/airpods_1ft_table.jpg", "small round earbud case", 1.0),
    ("depth_estimation/test_images/wallet_1ft_floor.jpg",  "wallet",                 1.0),
]
# ─────────────────────────────────────────────────────────────────────────────


def main():
    estimator = DepthEstimator(focal_length_px=FOCAL_LENGTH_PX)
    detector  = GroundingDinoDetector()

    for img_path, prompt, gt_ft in TESTS:
        print(f"\n[{Path(img_path).name}]  prompt='{prompt}'")
        image = load_image(img_path)
        detection = detector.detect(image, prompt)

        if not detection:
            print("  No detection.")
            continue

        box        = detection["box"]
        similarity = detection["score"]
        depth_map  = estimator.estimate(image)
        dist_ft    = estimator.get_depth_at_bbox(depth_map, box)
        direction  = estimator.get_direction(box, image.width)
        narration  = estimator.build_narration(detection["label"], direction, dist_ft, similarity)

        err = f"{abs(dist_ft - gt_ft) / gt_ft * 100:.1f}%" if gt_ft else "n/a"
        print(f"  conf={similarity:.3f}  dist={dist_ft:.2f}ft  error={err}")
        print(f"  direction={direction}")
        print(f"  narration={narration}")


if __name__ == "__main__":
    main()
