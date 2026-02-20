"""
HYBRID: Grounding DINO + CLIP re-ranking

Best of both worlds:
1. Use Grounding DINO with expanded prompts (fast, cheap)
2. Re-rank results with CLIP (accurate semantic matching)
"""

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image



class HybridObjectFinder:
    """
    Combines Grounding DINO's detection with CLIP's semantic matching.
    More robust than either alone for vague user queries.
    """
    
    def __init__(self):
        """Initialize both models."""
        from object_detection import GroundingDinoDetector
        
        self.detector = GroundingDinoDetector()
        
        # Load CLIP for re-ranking
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def expand_query(self, query):
        """
        Expand vague query into multiple specific prompts.
        No LLM needed - use rules-based expansion.
        
        Args:
            query: User's vague query (e.g., "airpods")
            
        Returns:
            List of expanded prompts
        """
        # Common object variations
        expansions = {
            "airpods": [
                "white wireless earbuds",
                "airpod case", 
                "small white case",
                "wireless headphones case"
            ],
            "phone": [
                "smartphone",
                "iphone", 
                "android phone",
                "mobile device"
            ],
            "keys": [
                "car keys",
                "house keys",
                "keychain",
                "metal keys"
            ],
            "wallet": [
                "leather wallet",
                "billfold",
                "card holder",
                "purse"
            ],
            "glasses": [
                "eyeglasses",
                "sunglasses", 
                "spectacles",
                "reading glasses"
            ]
        }
        
        # Return expanded prompts or fallback to original
        query_lower = query.lower().strip()
        
        if query_lower in expansions:
            return expansions[query_lower]
        else:
            # Fallback: just use original + generic descriptor
            return [query, f"small {query}", f"white {query}"]
    
    def find_with_reranking(self, image_path, user_query):
        """
        Find object using expanded prompts + CLIP re-ranking.
        
        Args:
            image_path: Path to image
            user_query: User's original query
            
        Returns:
            Best detection with semantic score
        """
        image = Image.open(image_path)
        
        # Step 1: Expand query
        prompts = self.expand_query(user_query)
        print(f"Expanded '{user_query}' → {prompts}")
        
        # Step 2: Detect with all prompts
        all_detections = []
        for prompt in prompts:
            detection = self.detector.detect(image, prompt)
            if detection:
                detection['prompt_used'] = prompt
                all_detections.append(detection)
        
        if not all_detections:
            print(f"No detections found for any prompt")
            return None
        
        print(f"Found {len(all_detections)} detections across {len(prompts)} prompts")
        
        # Step 3: Re-rank with CLIP
        best_match = self._rerank_with_clip(image, all_detections, user_query)
        
        return best_match
    
    def _rerank_with_clip(self, image, detections, query_text):
        """Re-rank detections using CLIP similarity to original query."""
        
        # Embed user's original query
        text_inputs = self.clip_processor(
            text=[query_text],
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            query_embedding = self.clip_model.get_text_features(**text_inputs)
            query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
        
        # Score each detection
        best_match = None
        best_score = -1
        
        for detection in detections:
            # Crop object
            bbox = detection['box']
            x1, y1, x2, y2 = [int(x) for x in bbox]
            crop = image.crop((x1, y1, x2, y2))
            
            # Embed crop
            image_inputs = self.clip_processor(
                images=crop,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                crop_embedding = self.clip_model.get_image_features(**image_inputs)
                crop_embedding = crop_embedding / crop_embedding.norm(dim=-1, keepdim=True)
            
            # Compute similarity
            similarity = (query_embedding @ crop_embedding.T).item()
            
            # Combined score: detection confidence + semantic similarity
            combined_score = (detection['score'] * 0.3) + (similarity * 0.7)
            
            print(f"  {detection['label']} (via '{detection['prompt_used']}'): "
                  f"det={detection['score']:.3f}, sem={similarity:.3f}, combined={combined_score:.3f}")
            
            if combined_score > best_score:
                best_score = combined_score
                best_match = {
                    **detection,
                    'semantic_score': similarity,
                    'combined_score': combined_score
                }
        
        print(f"\n→ Best match: {best_match['label']} (combined score: {best_match['combined_score']:.3f})")
        return best_match


# Updated usage in your test code:
if __name__ == "__main__":
    finder = HybridObjectFinder()
    
    result = finder.find_with_reranking(
        "input_images/depth_image.png",
        "airpods"  # Vague query
    )
    
    if result:
        print(f"\nFound: {result['label']}")
        print(f"Detection confidence: {result['score']:.3f}")
        print(f"Semantic match: {result['semantic_score']:.3f}")
        print(f"Bounding box: {result['box']}")
