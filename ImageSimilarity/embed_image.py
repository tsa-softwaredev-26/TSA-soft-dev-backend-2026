import torch
from torch import nn
from transformers import AutoImageProcessor, AutoModel


class ImageEmbedder:
    """
    Image embedding class using DINOv3 model.
    
    Handles model loading and image embedding generation.
    """
    
    def __init__(self, pretrained_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m"):
        """
        Initialize ImageEmbedder with DINOv3 model.
        
        Args:
            pretrained_model_name: HuggingFace model identifier for DINOv3
        """
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        # device_map="auto" handles Mac (CPU/MPS) and Linux RTX (CUDA) automatically
        self.model = AutoModel.from_pretrained(
            pretrained_model_name, 
            device_map="auto"
        )
    
    def embed(self, image):
        """
        Generate embedding for an image.
        
        Args:
            image: PIL Image or compatible image object
            
        Returns:
            Pooled embedding tensor
            
        Note:
            TODO: For future server version, implement batch processing to embed 
            multiple images at once for efficiency.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        return outputs.pooler_output
    
    @staticmethod
    def cosine_similarity(embedding1, embedding2):
        """
        Compute cosine similarity between two image embeddings.
        
        Args:
            embedding1: First image embedding tensor
            embedding2: Second image embedding tensor
            
        Returns:
            Cosine similarity score tensor
        """
        similarity = nn.CosineSimilarity(dim=1, eps=1e-8)(embedding1, embedding2)
        return similarity