"""
Depth estimation test: Grounding DINO bbox + Depth Pro.

Run from depth_demo/:
    cd /Volumes/T7/Developer/VisualMemory/depth_demo
    python depth_test.py

Ground truth: airpod case at 42cm.
"""

from pathlib import Path
import sys
import time
import torch
import pillow_heif
import depth_pro

pillow_heif.register_heif_opener()
sys.path.append(str(Path(__file__).parent.parent))

from utils import load_image
from object_detection import GroundingDinoDetector

IMAGE_PATH = str(Path(__file__).parent / "input_images" / "depth_image.png")
PROMPT = "small round earbud case"
GROUND_TRUTH_M = 0.42

IPHONE_15_PLUS_F_MM = 6.24
IPHONE_15_PLUS_SENSOR_W_MM = 8.64


def bbox_depth(depth, bbox):
    """Mean/min depth inside inner 50% of bbox (reduces edge bleed)."""
    x1, y1, x2, y2 = [int(c) for c in bbox]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    hw, hh = (x2 - x1) // 4, (y2 - y1) // 4
    region = depth[max(cy-hh,0):min(cy+hh,depth.shape[0]),
                   max(cx-hw,0):min(cx+hw,depth.shape[1])]
    mean_m = region.mean().item()
    min_m  = region.min().item()
    err    = abs(mean_m - GROUND_TRUTH_M) / GROUND_TRUTH_M * 100
    return mean_m, min_m, err


def main():
    # 1. Detect bbox
    print(f"Detecting '{PROMPT}'...")
    detection = GroundingDinoDetector().detect(load_image(IMAGE_PATH), PROMPT)
    if not detection:
        print("No detection."); return
    box = detection["box"]
    print(f"  {detection['label']}  conf={detection['score']:.3f}  box={[round(c,1) for c in box]}\n")

    # 2. Load model
    # TODO: DELETE hardcoded checkpoint path when moving to depth_estimation module.
    #       Use symlink at project root instead: depth_pro.create_model_and_transforms()
    print("Loading Depth Pro...")
    t = time.time()
    config = depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT
    config.checkpoint_uri = Path(__file__).parent / "ml-depth-pro" / "checkpoints" / "depth_pro.pt"
    model, transform = depth_pro.create_model_and_transforms(config=config)
    model.eval()
    print(f"Loaded in {time.time()-t:.1f}s\n")

    # 3. Load image — official Depth Pro pattern
    image, _, f_px = depth_pro.load_rgb(IMAGE_PATH)
    image = transform(image)

    # 4. INFERRED — pass f_px from load_rgb (may be None if no EXIF)
    print("--- INFERRED ---")
    t = time.time()
    pred = model.infer(image, f_px=f_px)
    print(f"Inference: {time.time()-t:.1f}s")
    mean, min_d, err = bbox_depth(pred["depth"], box)
    print(f"focal_px={pred['focallength_px']:.1f}  mean={mean:.3f}m  min={min_d:.3f}m  error={err:.1f}%\n")

    # 5. CALIBRATED — iPhone 15 Plus f_px, wrapped as tensor to match load_rgb output type
    pil = load_image(IMAGE_PATH)
    f_px_iphone = torch.tensor(
        (IPHONE_15_PLUS_F_MM / IPHONE_15_PLUS_SENSOR_W_MM) * pil.width,
        dtype=torch.float32
    )
    print(f"--- CALIBRATED (iPhone 15 Plus f_px={f_px_iphone:.1f}) ---")
    t = time.time()
    pred_cal = model.infer(image, f_px=f_px_iphone)
    print(f"Inference: {time.time()-t:.1f}s")
    mean_c, min_c, err_c = bbox_depth(pred_cal["depth"], box)
    print(f"focal_px={f_px_iphone:.1f}  mean={mean_c:.3f}m  min={min_c:.3f}m  error={err_c:.1f}%\n")

    # 6. Summary
    print("=" * 40)
    print(f"Inferred   error: {err:.1f}%")
    print(f"Calibrated error: {err_c:.1f}%")
    print(f"Winner: {'INFERRED' if err <= err_c else 'CALIBRATED'}")


if __name__ == "__main__":
    main()
