"""
DepthEstimator manual test — add your own image paths and run.

Run from project root:
    python -m visual_memory.cli.tests.test_estimator
"""

from visual_memory.engine.depth import DepthEstimator
from visual_memory.engine.object_detection import GroundingDinoDetector
from visual_memory.utils import load_image
from visual_memory.config.paths import INPUT_IMAGES_DIR, FOCAL_LENGTH_PX_IPHONE15

# ── Configure your tests here ────────────────────────────────────────────────
# (image_path, prompt, ground_truth_ft)
# ground_truth_ft is the actual measured distance — used to compute error %
# Set to None if you just want to see the narration without error checking

FOCAL_LENGTH_PX = FOCAL_LENGTH_PX_IPHONE15

# Example test cases using images from INPUT_IMAGES_DIR
TESTS = [
    (INPUT_IMAGES_DIR / "wallet_1ft_table.jpg", "wallet", 1.0),
    (INPUT_IMAGES_DIR / "wallet_3ft_table.jpg", "wallet", 3.0),
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
