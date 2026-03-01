"""
Lightweight device-check test: load each model once, log device usage, minimal inference.
Runs in ~2-3 min instead of 15+ min.

Run:
    python -m visual_memory.tests.scripts.test_devices
"""

from __future__ import annotations
import json
import os
import sys
import torch
from pathlib import Path

# Suppress noise
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pillow_heif
pillow_heif.register_heif_opener()

from visual_memory.utils import load_image, get_logger

_log = get_logger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent
TESTS_DIR = SCRIPTS_DIR.parent
INPUT_DIR = TESTS_DIR / "input_images"
PROJECT_ROOT = TESTS_DIR.parents[2]

WALLET_1FT = INPUT_DIR / "wallet_1ft_table.jpg"
WALLET_3FT = INPUT_DIR / "wallet_3ft_table.jpg"
MAGNESIUM = INPUT_DIR / "magnesium.heic"

_G = "\033[32m"
_R = "\033[31m"
_B = "\033[1m"
_Y = "\033[33m"
_X = "\033[0m"

results = []

def log_device(name: str, device_obj) -> str:
    """Extract device string from torch device or model."""
    if isinstance(device_obj, torch.device):
        dev_str = str(device_obj)
    elif isinstance(device_obj, str):
        dev_str = device_obj
    elif hasattr(device_obj, 'device'):
        dev_str = str(device_obj.device)
    else:
        dev_str = "unknown"
    print(f"  {name:<25} device: {_Y}{dev_str}{_X}")
    return dev_str

print(f"\n{_B}Device Check Test{_X}")
print(f"{_B}{'='*60}{_X}\n")

# Test 1: GroundingDINO
print(f"{_B}[1] GroundingDinoDetector{_X}")
try:
    from visual_memory.engine.object_detection import GroundingDinoDetector
    detector = GroundingDinoDetector()
    dev = log_device("Model device", detector.device)

    # Minimal inference
    img = load_image(str(WALLET_1FT))
    result = detector.detect(img, "wallet")
    has_result = result is not None

    if has_result:
        print(f"  {_G}✓ Inference OK (confidence={result['score']:.3f}){_X}\n")
        results.append(("grounding_dino", True, dev))
    else:
        print(f"  {_R}✗ No detection{_X}\n")
        results.append(("grounding_dino", False, dev))
except Exception as e:
    print(f"  {_R}✗ Error: {str(e)[:80]}{_X}\n")
    results.append(("grounding_dino", False, "error"))

# Test 2: YoloeDetector
print(f"{_B}[2] YoloeDetector{_X}")
try:
    from visual_memory.engine.object_detection import YoloeDetector
    yolo = YoloeDetector()
    # YOLOE doesn't expose device directly; check the model
    if hasattr(yolo.prompt_free_model, 'device'):
        dev = log_device("Model device", yolo.prompt_free_model.device)
    else:
        dev = "auto (ultralytics)"
        print(f"  YoloeDetector device: {_Y}{dev}{_X}")

    # Minimal inference
    boxes, scores = yolo.detect_all(str(WALLET_3FT))
    has_result = boxes is not None and len(boxes) > 0

    if has_result:
        print(f"  {_G}✓ Inference OK (detections={len(boxes)}){_X}\n")
        results.append(("yoloe", True, dev))
    else:
        print(f"  {_R}✗ No detections{_X}\n")
        results.append(("yoloe", False, dev))
except Exception as e:
    print(f"  {_R}✗ Error: {str(e)[:80]}{_X}\n")
    results.append(("yoloe", False, "error"))

# Test 3: DepthEstimator (SKIPPED — large model load, just verify config is correct)
# Uncomment to test if needed; loads ~2GB model
# print(f"{_B}[3] DepthEstimator{_X}")
# try:
#     from visual_memory.engine.depth import DepthEstimator
#     estimator = DepthEstimator(focal_length_px=3094.0)
#     dev = log_device("Model device", estimator.device)
#
#     # Minimal inference (just on WALLET_1FT, small image)
#     img = load_image(str(WALLET_1FT))
#     depth_map = estimator.estimate(img)
#
#     if depth_map is not None:
#         mean_depth = depth_map.mean().item()
#         print(f"  {_G}✓ Inference OK (mean_depth={mean_depth:.2f}m){_X}\n")
#         results.append(("depth_estimator", True, dev))
#     else:
#         print(f"  {_R}✗ No depth map{_X}\n")
#         results.append(("depth_estimator", False, dev))
# except Exception as e:
#     print(f"  {_R}✗ Error: {str(e)[:80]}{_X}\n")
#     results.append(("depth_estimator", False, "error"))

# Test 4: ImageEmbedder (DINOv3) + CLIPTextEmbedder
print(f"{_B}[4] ImageEmbedder (DINOv3) + CLIPTextEmbedder{_X}")
try:
    from visual_memory.engine.embedding import ImageEmbedder, CLIPTextEmbedder
    img_embedder = ImageEmbedder()
    dev = log_device("ImageEmbedder device", img_embedder.device)

    # Image embedding
    img = load_image(str(WALLET_1FT))
    img_emb = img_embedder.embed(img)

    text_embedder = CLIPTextEmbedder()
    text_dev = log_device("CLIPTextEmbedder device", text_embedder.device)

    # Text embedding
    text_emb = text_embedder.embed_text("wallet")

    if img_emb.shape == (1, 1024) and text_emb.shape == (1, 512):
        print(f"  {_G}✓ Inference OK (img={list(img_emb.shape)}, text={list(text_emb.shape)}){_X}\n")
        results.append(("dinov3_clip_embedder", True, dev))
    else:
        print(f"  {_R}✗ Unexpected shapes: img={img_emb.shape}, text={text_emb.shape}{_X}\n")
        results.append(("dinov3_clip_embedder", False, dev))
except Exception as e:
    print(f"  {_R}✗ Error: {str(e)[:80]}{_X}\n")
    results.append(("dinov3_clip_embedder", False, "error"))

# Test 5: TextRecognizer (PaddleOCR)
print(f"{_B}[5] TextRecognizer (PaddleOCR){_X}")
try:
    from visual_memory.engine.text_recognition import TextRecognizer
    recognizer = TextRecognizer()
    # PaddleOCR uses CPU (no GPU support)
    dev = "cpu (PaddlePaddle)"
    print(f"  TextRecognizer device: {_Y}{dev}{_X}")

    # Minimal inference
    img = load_image(str(MAGNESIUM))
    ocr_result = recognizer.recognize(img)

    if isinstance(ocr_result, dict) and "text" in ocr_result:
        text_len = len(ocr_result["text"])
        print(f"  {_G}✓ Inference OK (text_len={text_len}){_X}\n")
        results.append(("text_recognizer", True, dev))
    else:
        print(f"  {_R}✗ Unexpected result{_X}\n")
        results.append(("text_recognizer", False, dev))
except Exception as e:
    print(f"  {_R}✗ Error: {str(e)[:80]}{_X}\n")
    results.append(("text_recognizer", False, "error"))

# Summary
print(f"{_B}Results{_X}")
print(f"{_B}{'='*60}{_X}\n")

n_pass = sum(1 for _, ok, _ in results if ok)
n_total = len(results)

for name, ok, device in results:
    icon = f"{_G}✓{_X}" if ok else f"{_R}✗{_X}"
    print(f"  {icon}  {name:<25}  {device}")

print(f"\n{_B}Summary: {n_pass}/{n_total} passed (Depth Pro skipped){_X}\n")

sys.exit(0 if n_pass == n_total else 1)
