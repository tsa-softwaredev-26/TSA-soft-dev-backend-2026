"""Image embedding module using DINOv3."""
from typing import Optional, List
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

from visual_memory.config import Settings
from visual_memory.utils import get_logger
from visual_memory.utils.device_utils import get_device

_defaults = Settings()
_log = get_logger(__name__)


class ImageEmbedder:
    """
    Image embedder using DINOv3 (facebook/dinov3-vitl16-pretrain-lvd1689m).

    Returns 1024-dim L2-normalized embeddings from the pooler output.
    Self-supervised vision-only model; better object-level discrimination
    than CLIP for same-scene objects.

    Requires transformers >= 4.56.0 for DINOv3 support.
    """

    def __init__(
        self,
        pretrained_model_name: str = _defaults.image_embedder_model,
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device if device else get_device())
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        self.model = AutoModel.from_pretrained(pretrained_model_name).to(self.device).eval()

    def embed(self, image: Image.Image) -> torch.Tensor:
        """Embed a single image. Returns (1, 1024) L2-normalized tensor."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        return F.normalize(outputs.pooler_output.detach().cpu(), dim=1)

    def batch_embed(self, images: List[Image.Image]) -> torch.Tensor:
        """Embed a batch of images in a single forward pass.

        Returns (N, 1024) L2-normalized tensor. Preferred over calling
        embed() in a loop on CUDA/MPS - processes all images at once.
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        return F.normalize(outputs.pooler_output.detach().cpu(), dim=1)
