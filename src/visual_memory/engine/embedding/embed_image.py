"""Image embedding module using DINOv3."""
from typing import Optional
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

from visual_memory.config import Settings
from visual_memory.utils import get_logger

_defaults = Settings()
_log = get_logger(__name__)


class ImageEmbedder:
    """
    Image embedder using DINOv3 (facebook/dinov3-vitl16-pretrain-lvd1689m).

    Returns 1024-dim L2-normalized embeddings from the pooler output.
    Self-supervised vision-only model — better object-level discrimination
    than CLIP, which encodes scene context alongside object identity.

    Requires transformers >= 4.56.0 for DINOv3 support.

    TODO (server migration): add batch_embed(images) for GPU throughput.
    """

    def __init__(
        self,
        pretrained_model_name: str = _defaults.image_embedder_model,
        device: Optional[str] = None,
    ) -> None:
        if device:
            self.device = torch.device(device)
        else:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")  # macOS GPU for backend dev
                # TODO (server migration): remove MPS branch; server uses CUDA only.
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")  # CUDA for later server deployment
            else:
                self.device = torch.device("cpu")

        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        self.model = AutoModel.from_pretrained(pretrained_model_name).to(self.device).eval()

    def embed(self, image: Image.Image) -> torch.Tensor:
        """
        Generate L2-normalized image embedding.

        Args:
            image: PIL Image object

        Returns:
            Embedding tensor of shape (1, 1024)

        Note:
            TODO: For server version, implement batch processing to embed
            multiple images at once for efficiency.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        return F.normalize(outputs.pooler_output.detach().cpu(), dim=1)
