"""
Grounding DINO object detection module.

Provides a class-based interface to the Grounding DINO zero-shot object detection
model for detecting objects from images using natural language prompts.

Device selection: CUDA > MPS > CPU via get_device().
"""
from typing import Optional, Dict, Any, List
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from pathlib import Path
from PIL import Image

from visual_memory.config import Settings
from visual_memory.utils.device_utils import get_device

_defaults = Settings()


class GroundingDinoDetector:
    """
    Zero-shot object detector using Grounding DINO.

    Detects objects in images using natural language prompts and returns
    bounding box coordinates for the highest-confidence detection.
    """

    def __init__(
        self,
        model_id: str = _defaults.grounding_dino_model,
        box_threshold: float = _defaults.box_threshold,
        text_threshold: float = _defaults.text_threshold,
        device: Optional[str] = None
    ) -> None:
        self.model_id = model_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device if device else get_device()

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id
        ).to(self.device)
        self.model.eval()

    def detect(self, image: Image.Image, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Detect an object in an image using a text prompt.

        Returns dict with 'box', 'score', 'label', or None if no detection above threshold.
        """
        text_labels = [[prompt]]
        inputs = self.processor(
            images=image,
            text=text_labels,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]]
        )

        result = results[0]
        if len(result["boxes"]) == 0:
            return None

        best_idx = result["scores"].argmax()
        return {
            "box": result["boxes"][best_idx].tolist(),
            "score": result["scores"][best_idx].item(),
            "label": result["labels"][best_idx],
        }

    def detect_batch(
        self,
        images: List[Image.Image],
        prompts: List[str],
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Detect objects in a batch of images in a single forward pass.

        Returns List[Optional[dict]] aligned with the input lists.
        Each dict has 'box', 'score', 'label'; None if no detection above threshold.
        Preferred over calling detect() in a loop on CUDA - single model forward pass.
        """
        text_labels = [[p] for p in prompts]
        target_sizes = [img.size[::-1] for img in images]
        inputs = self.processor(
            images=images,
            text=text_labels,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=target_sizes,
        )

        detections = []
        for result in results:
            if len(result["boxes"]) == 0:
                detections.append(None)
            else:
                best_idx = result["scores"].argmax()
                detections.append({
                    "box": result["boxes"][best_idx].tolist(),
                    "score": result["scores"][best_idx].item(),
                    "label": result["labels"][best_idx],
                })
        return detections
