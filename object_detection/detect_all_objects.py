"""
YOLOE object detection module.

This module provides a wrapper for YOLOE object detection with both prompt-based
and prompt-free detection modes.
"""
from typing import Optional, Tuple, List
from pathlib import Path

try:
    from ultralytics import YOLOE
except ImportError:
    raise ImportError(
        "ultralytics package is required. Install with: pip install ultralytics>=8.0.0"
    )

from PIL import Image


class YoloeDetector:
    """
    YOLOE detector with fixed semantics:
    - prompt-free model: ALL detections with set confidence & iou (intersection) thresholds, 
      not very accurate class labels; breadth>accuracy of detections for similarity 
      algorithm to handle in depth since its quite fast
    
    - prompt-based model: (tried to do with "household object" prompt but unused right now) 
      BEST detection only with prompt. Grounding DINO works better for prompt-based, 
      able to handle more abstract concepts
    """

    def __init__(
        self,
        prompt_free_model_path: str = "object_detection/yoloe-26l-seg-pf.pt",
        confidence_threshold: float = 0.35, 
        intersection_threshold: float = 0.45
    ) -> None:
        """
        Args:
            prompt_free_model_path: Path to the prompt-free YOLOE model
            confidence_threshold: Minimum confidence score for detections (0-1)
            intersection_threshold: IoU (Intersection over Union) threshold (0-1)
        """
        model_path = Path(prompt_free_model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {prompt_free_model_path}\n"
                f"Please ensure the model file is in the correct location."
            )
        
        self.prompt_free_model = YOLOE(prompt_free_model_path)
        self.confidence_threshold = confidence_threshold
        self.intersection_threshold = intersection_threshold

    def detect_all(self, image_path: str) -> Tuple[Optional[List[List[float]]], Optional[List[float]]]:
        """
        Prompt-free detection: return ALL boxes.

        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (boxes, scores) where:
                - boxes: List of bounding boxes as [x1, y1, x2, y2]
                - scores: List of confidence scores
            Returns (None, None) if no detections found
            
        Raises:
            FileNotFoundError: If image file doesn't exist
        """
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            results = self.prompt_free_model.predict(
                image_path, 
                verbose=True, 
                conf=self.confidence_threshold, 
                iou=self.intersection_threshold
            )
        except Exception as e:
            raise RuntimeError(f"Error during YOLOE detection: {str(e)}") from e

        if len(results[0].boxes) == 0:
            return None, None

        boxes = results[0].boxes.xyxy.tolist()
        scores = results[0].boxes.conf.tolist()

        return boxes, scores
