"""
Depth estimation and spatial narration for scan mode.

All spatial logic lives here.
Instantiate once per process via model_registry.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch
import depth_pro
from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
from PIL import Image

# Absolute path to the checkpoint — resolved at import time from this file's location,
# so it works regardless of the current working directory or platform.
_CHECKPOINT_PATH = Path(__file__).resolve().parents[4] / "checkpoints" / "depth_pro.pt"

CONFIDENCE_HIGH = 0.6


class DepthEstimator:

    def __init__(self):
        # Auto-detect device: CUDA > MPS > CPU.
        # depth_pro.create_model_and_transforms defaults to CPU if not passed explicitly,
        # which causes 10-30x slowdown vs GPU. Always pass device explicitly.
        from visual_memory.utils.device_utils import get_device
        self.device = torch.device(get_device())

        config = dataclasses.replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(_CHECKPOINT_PATH))
        self.model, self.transform = depth_pro.create_model_and_transforms(config=config, device=self.device)
        self.model.eval()

    def estimate(self, image: Image.Image, focal_length_px: float = None) -> torch.Tensor:
        # focal_length_px from Android: (focalLengthMm / sensorWidthMm) * imageWidthPx
        # None = Depth Pro infers (~75% error vs ~26% calibrated at close range)
        # Call once per query image — reuse depth_map for all matched objects
        # depth_pro.load_rgb expects a file path, so we use transform directly on the PIL image
        image_tensor = self.transform(image).to(self.device)
        f_px = torch.tensor(focal_length_px, dtype=torch.float32).to(self.device) if focal_length_px else None
        with torch.no_grad():
            prediction = self.model.infer(image_tensor, f_px=f_px)
        return prediction["depth"]  # (H, W) metric meters

    def get_depth_at_bbox(self, depth_map: torch.Tensor, bbox: list) -> float:
        # Inner 50% of bbox reduces background bleed at object edges
        x1, y1, x2, y2 = [int(c) for c in bbox]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        hw, hh = (x2 - x1) // 4, (y2 - y1) // 4
        region = depth_map[max(cy-hh,0):min(cy+hh, depth_map.shape[0]),
                           max(cx-hw,0):min(cx+hw, depth_map.shape[1])]
        if region.numel() == 0:
            return 0.0
        return region.mean().item() * 3.28084  # meters → feet

    # Previously used 12-hour clock position (e.g. "3 o'clock") — switched to plain
    # directions to remove mental translation step for blind users. Could revert if
    # finer granularity is needed, but 5 zones map directly to body movement.
    def get_direction(self, bbox: list, img_w: int) -> str:
        cx = (bbox[0] + bbox[2]) / 2
        nx = (cx / img_w) * 2 - 1  # -1=far left, 1=far right

        if nx < -0.5:  return "to your left"
        if nx < -0.15: return "slightly left"
        if nx < 0.15:  return "ahead"
        if nx < 0.5:   return "slightly right"
        return                "to your right"

    def build_narration(
        self,
        label: str,
        direction: str,
        distance_ft: float,
        similarity: float
    ) -> str | None:
        if similarity >= CONFIDENCE_HIGH:
            return f"{label.capitalize()} {direction}, {distance_ft:.1f} feet away."
        else:
            return f"May be a {label} {direction}, focus to verify."
