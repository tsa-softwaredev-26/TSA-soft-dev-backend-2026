"""
Grounding DINO object detection module.

This module provides a class-based interface to the Grounding DINO zero-shot
object detection model for detecting objects from images using natural language prompts.

TODO: Migrate to RTX server for faster inference
    - Change device to "cuda" by default
    - Add batch processing by modifying detect() to accept List[image_path] and List[prompt]
    - Process multiple images in single forward pass: processor(images=[img1, img2, ...])
    - Return List[detection_dict] instead of single detection_dict
"""
from typing import Optional, Dict, Any
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from pathlib import Path
from PIL import Image



class GroundingDinoDetector:
    """
    Zero-shot object detector using Grounding DINO.
    
    Detects objects in images using natural language prompts and returns
    bounding box coordinates for the highest-confidence detection.
    """
    
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        box_threshold: float = 0.2, 
        text_threshold: float = 0.2, 
        device: Optional[str] = None
    ) -> None:
        """
        Initialize the Grounding DINO detector.
        
        Args:
            model_id: Hugging Face model ID for Grounding DINO
            box_threshold: Confidence threshold for bounding box detection
            text_threshold: Text-to-image similarity threshold
            device: torch device ("cuda", "mps", "cpu", or None for auto-detect)
        """
        self.model_id = model_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        
        # Auto-detect best available device if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}") # Remove debug print when called from Kotlin
        
        # Load model and processor immediately
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id
        ).to(self.device)
    
    def detect(self, image, prompt):
        """
        Detect an object in an image using a text prompt.
        
        Args:
            image: PIL Image object
            prompt: Text description of object to detect (e.g., "camo wallet")            
        Returns:
            Dictionary containing:
                - 'box': Bounding box coordinates [x1, y1, x2, y2]
                - 'score': Confidence score (0-1)
                - 'label': Detected label text
            Returns None if no detection above threshold.
        """
        # Prepare inputs - text_labels must be nested list for batch processing
        text_labels = [[prompt]]
        
        print(f"Detecting '{prompt}'...")
        inputs = self.processor(
            images=image, 
            text=text_labels, 
            return_tensors="pt"
        ).to(self.device)
        
        # Only inference (no training)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process results to get bounding boxes
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]]  # (width, height) -> (height, width)
        )
        
        result = results[0]
        
        if len(result["boxes"]) == 0:
            print(results)
            return None
        
        # Return the highest confidence detection
        best_idx = result["scores"].argmax()
        
        detection = {
            'box': result["boxes"][best_idx].tolist(),
            'score': result["scores"][best_idx].item(),
            'label': result["labels"][best_idx]
        }
        
        
        
        return detection