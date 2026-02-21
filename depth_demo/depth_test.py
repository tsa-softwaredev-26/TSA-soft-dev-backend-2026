"""
Depth estimation test suite — runs all 18 ground truth images.

Images go in: depth_demo/test_images/
Ground truth:  depth_demo/ground_truth.json

Run from project root:
    python depth_demo/depth_test.py

Point camera naturally — no chest-height constraint.
Depth Pro gives straight-line distance regardless of camera angle.
"""

from pathlib import Path
import sys
import json
import math
import time
import torch
import pillow_heif
import depth_pro

pillow_heif.register_heif_opener()
sys.path.append(str(Path(__file__).parent.parent))

from utils import load_image
from object_detection import GroundingDinoDetector

CURRENT_DIR  = Path(__file__).parent
IMAGES_DIR   = CURRENT_DIR / "test_images"
GROUND_TRUTH = CURRENT_DIR / "ground_truth.json"

IPHONE_15_PLUS_F_MM        = 6.24
IPHONE_15_PLUS_SENSOR_W_MM = 8.64

CONFIDENCE_HIGH = 0.6
CONFIDENCE_LOW  = 0.4


# ─── Spatial helpers ──────────────────────────────────────────────────────────

def get_clock_position(bbox, img_w, img_h) -> str:
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    nx = (cx / img_w) * 2 - 1
    ny = (cy / img_h) * 2 - 1
    angle = math.degrees(math.atan2(nx, -ny)) % 360
    hour  = round(angle / 30) % 12 or 12
    return f"{hour} o'clock"


def depth_at_bbox(depth, bbox) -> float:
    """Mean depth in inner 50% of bbox. Returns meters."""
    x1, y1, x2, y2 = [int(c) for c in bbox]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    hw, hh = (x2 - x1) // 4, (y2 - y1) // 4
    region = depth[max(cy-hh,0):min(cy+hh, depth.shape[0]),
                   max(cx-hw,0):min(cx+hw, depth.shape[1])]
    return region.mean().item()


def build_narration(label, clock, distance_ft, similarity) -> str:
    distance_str = f"{distance_ft:.1f} feet away"
    if similarity >= CONFIDENCE_HIGH:
        return f"{label.capitalize()} at {clock}, {distance_str}."
    else:
        return f"May be a {label} at {clock}, focus to verify."


# ─── Single image ─────────────────────────────────────────────────────────────

def run_single(model, transform, detector, prompt, test) -> dict:
    img_path = IMAGES_DIR / test["filename"]
    pil_image = load_image(str(img_path))
    raw_image, _, _ = depth_pro.load_rgb(str(img_path))
    image_tensor = transform(raw_image)

    # Calibrated focal length (winner from initial test)
    f_px = torch.tensor(
        (IPHONE_15_PLUS_F_MM / IPHONE_15_PLUS_SENSOR_W_MM) * pil_image.width,
        dtype=torch.float32
    )

    # TODO: DELETE hardcoded checkpoint path when moving to depth_estimation module.
    #       Replace config override with plain: depth_pro.create_model_and_transforms()
    prediction = model.infer(image_tensor, f_px=f_px)
    depth = prediction["depth"]

    detection = detector.detect(pil_image, prompt)
    if not detection:
        return {"filename": test["filename"], "error": "no detection"}

    box        = detection["box"]
    similarity = detection["score"]
    label      = test["object"]
    img_w, img_h = pil_image.width, pil_image.height

    straight_m  = depth_at_bbox(depth, box)
    straight_ft = straight_m * 3.28084
    clock       = get_clock_position(box, img_w, img_h)
    narration   = build_narration(label, clock, straight_ft, similarity)

    gt_m  = test["distance_m"]
    err   = abs(straight_m - gt_m) / gt_m * 100

    return {
        "filename":           test["filename"],
        "object":             label,
        "detection_conf":     round(similarity, 3),
        "straight_line_m":    round(straight_m, 3),
        "straight_line_ft":   round(straight_ft, 2),
        "ground_truth_m":     gt_m,
        "ground_truth_ft":    test["distance_ft"],
        "distance_error_pct": round(err, 1),
        "clock_position":     clock,
        "narration":          narration,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    with open(GROUND_TRUTH) as f:
        gt = json.load(f)

    prompts = {k: v["prompt"] for k, v in gt["objects"].items()}
    tests   = gt["tests"]

    # Load model once
    print("Loading Depth Pro...")
    t = time.time()
    config = depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT
    config.checkpoint_uri = CURRENT_DIR / "ml-depth-pro" / "checkpoints" / "depth_pro.pt"
    model, transform = depth_pro.create_model_and_transforms(config=config)
    model.eval()
    print(f"Loaded in {time.time()-t:.1f}s\n")

    detector = GroundingDinoDetector()
    results  = []
    skipped  = []

    for test in tests:
        img_path = IMAGES_DIR / test["filename"]
        if not img_path.exists():
            skipped.append(test["filename"])
            continue

        prompt = prompts[test["object"]]
        print(f"Running {test['filename']}...")
        t = time.time()
        result = run_single(model, transform, detector, prompt, test)
        results.append(result)

        if "error" in result:
            print(f"  No detection.\n")
        else:
            print(f"  {result['narration']}")
            print(f"  error={result['distance_error_pct']}%  conf={result['detection_conf']}")
        print(f"  ({time.time()-t:.1f}s)\n")

    if not results:
        print("No results — add images to depth_demo/test_images/ matching ground_truth.json filenames.")
        return

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print(f"{'FILE':<30} {'GT_FT':>5} {'PRED_FT':>7} {'ERR%':>6} {'CONF':>5}  {'CLOCK':<12}  NARRATION")
    print("-" * 85)

    valid = []
    for r in results:
        if "error" in r:
            print(f"{r['filename']:<30}  NO DETECTION")
            continue
        valid.append(r)
        print(f"{r['filename']:<30} {r['ground_truth_ft']:>5.1f} {r['straight_line_ft']:>7.2f} "
              f"{r['distance_error_pct']:>5.1f}% {r['detection_conf']:>5.2f}  {r['clock_position']:<12}  {r['narration']}")

    if valid:
        avg_err  = sum(r["distance_error_pct"] for r in valid) / len(valid)
        avg_conf = sum(r["detection_conf"] for r in valid) / len(valid)
        print("-" * 85)
        print(f"{'AVERAGE':<30} {'':>5} {'':>7} {avg_err:>5.1f}%  {avg_conf:>4.2f}")

    if skipped:
        print(f"\nSkipped (missing images): {', '.join(skipped)}")

    print(f"\nTotal: {len(valid)} ran, {len(skipped)} skipped")


if __name__ == "__main__":
    main()
