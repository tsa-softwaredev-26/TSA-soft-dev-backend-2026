"""Image embedding module using DINOv3."""
from typing import Optional
import torch
from torch import nn
from transformers import AutoImageProcessor, AutoModel
from PIL import Image


class ImageEmbedder:
    
    def __init__(self, pretrained_model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m") -> None:
        """
        Initialize image embedder with DINOv3 model.
        
        Args:
            pretrained_model_name: HuggingFace model identifier (default is DINOv3 ViT-Large)
        """
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        # device_map="auto" handles Mac (CPU/MPS) and Linux RTX (CUDA) automatically
        self.model = AutoModel.from_pretrained(
            pretrained_model_name, 
            device_map="auto"
        )
    
    def embed(self, image: Image.Image) -> torch.Tensor:
        """
        Generate embedding for an image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Pooled embedding tensor of shape (1, embedding_dim)
            
        Note:
            TODO: For server version, implement batch processing to embed 
            multiple images at once for efficiency.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        return outputs.pooler_output
    
   