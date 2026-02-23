"""Image embedding module using DINOv3."""
from typing import Optional
import torch
from torch import nn
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

from visual_memory.config import Settings

_defaults = Settings()


class ImageEmbedder:
    def __init__(
        self,
        pretrained_model_name: str = _defaults.embedder_model,
        device: Optional[str] = None,
    ) -> None:
        if device:
            self.device = torch.device(device)
        else:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps") # macOS GPU support for backend dev
            elif torch.cuda.is_available():
                self.device = torch.device("cuda") # CUDA GPU for later server deployment
            else:
                self.device = torch.device("cpu")

        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)

        self.model = AutoModel.from_pretrained(pretrained_model_name)
        self.model.to(self.device)
        self.model.eval()
        
    
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
        return outputs.pooler_output.detach().cpu()
    
   