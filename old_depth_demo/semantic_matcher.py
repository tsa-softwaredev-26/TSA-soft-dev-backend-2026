"""
Two-stage object detection with CLIP semantic matching.

Stage 1: Detect all objects (class-agnostic)
Stage 2: Match detections to user query via CLIP embeddings
"""

import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from PIL import Image


class SemanticObjectMatcher:
    """
    Matches user text queries to detected objects using CLIP embeddings.
    More robust than text-only prompts for vague descriptions.
    """
    
    def __init__(self):
        """Initialize CLIP model for semantic matching."""
        print("Loading CLIP model for semantic matching...")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use CLIP for semantic matching
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        print("✓ CLIP loaded successfully!\n")
    
    def match_objects_to_query(self, image, detections, query_text):
        """
        Match detected objects to user query using semantic similarity.
        
        Args:
            image: PIL Image (original full image)
            detections: List of dicts with 'box', 'score', 'label'
            query_text: User's search query (e.g., "airpods")
            
        Returns:
            Best matching detection with similarity score
        """
        if not detections:
            return None
        
        # Embed the user query
        text_inputs = self.clip_processor(
            text=[query_text],
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            query_embedding = self.clip_model.get_text_features(**text_inputs)
            query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
        
        # Embed each detected object crop
        best_match = None
        best_score = -1
        
        for detection in detections:
            # Crop the object from image
            bbox = detection['box']
            x1, y1, x2, y2 = [int(x) for x in bbox]
            crop = image.crop((x1, y1, x2, y2))
            
            # Embed the crop
            image_inputs = self.clip_processor(
                images=crop,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                crop_embedding = self.clip_model.get_image_features(**image_inputs)
                crop_embedding = crop_embedding / crop_embedding.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity
            similarity = (query_embedding @ crop_embedding.T).item()
            
            # Track best match
            if similarity > best_score:
                best_score = similarity
                best_match = {
                    **detection,
                    'semantic_score': similarity,
                    'crop': crop
                }
        
        return best_match


# Usage example:
def find_object_semantic(image_path, query):
    """
    Find object using semantic matching instead of direct text prompt.
    
    Args:
        image_path: Path to image
        query: User's vague description (e.g., "airpods")
        
    Returns:
        Best matching object with location and distance
    """
    from object_detection import GroundingDinoDetector
    from depth_estimator import DepthEstimator  # Your DepthPro wrapper
    
    image = Image.open(image_path)
    
    # Stage 1: Detect ALL objects (class-agnostic)
    # Use generic prompts to get all candidates
    detector = GroundingDinoDetector()
    
    # Try generic detection prompts
    generic_prompts = ["object", "item", "thing"]
    all_detections = []
    
    for prompt in generic_prompts:
        detection = detector.detect(image, prompt)
        if detection:
            all_detections.append(detection)
    
    # Stage 2: Match to user query with CLIP
    matcher = SemanticObjectMatcher()
    best_match = matcher.match_objects_to_query(image, all_detections, query)
    
    if not best_match:
        return None
    
    # Stage 3: Get depth at matched object
    depth_estimator = DepthEstimator()
    depth_info = depth_estimator.get_depth_at_bbox(image_path, best_match['box'])
    
    return {
        'object': query,  # User's original query
        'detected_as': best_match.get('label'),
        'confidence': best_match['score'],
        'semantic_score': best_match['semantic_score'],
        'distance_feet': depth_info['distance_feet'],
        'bbox': best_match['box']
    }
