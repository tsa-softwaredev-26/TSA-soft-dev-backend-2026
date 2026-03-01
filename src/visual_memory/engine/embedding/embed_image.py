"""Image + text embedding module using CLIP (shared embedding space)."""
from typing import Optional
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

from visual_memory.config import Settings
from visual_memory.utils import get_logger

_defaults = Settings()
_log = get_logger(__name__)


class CLIPEmbedder:
    """
    Image and text embedder using openai/clip-vit-base-patch32.

    Both embed() and embed_text() return 512-dim L2-normalized vectors in the
    same shared CLIP embedding space, making image↔text cosine similarity
    meaningful without any concatenation hack.

    TODO (server migration): add batch_embed(images) / batch_embed_text(texts)
    using batched CLIPProcessor calls for GPU throughput.
    """

    def __init__(
        self,
        model_name: str = _defaults.embedder_model,
        device: Optional[str] = None,
    ) -> None:
        if device:
            self.device = torch.device(device)
        else:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")  # macOS GPU for backend dev
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")  # CUDA for later server deployment
            else:
                self.device = torch.device("cpu")

        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed(self, image: Image.Image) -> torch.Tensor:
        """
        Generate L2-normalized image embedding.

        Args:
            image: PIL Image object

        Returns:
            Embedding tensor of shape (1, 512)

        Note:
            TODO: For server version, implement batch processing to embed
            multiple images at once for efficiency.
        """
        # Use vision_model + visual_projection directly.
        # transformers >=5.x changed get_image_features() to return a ModelOutput
        # object instead of a plain tensor, so we call the sub-models explicitly.
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].to(self.device)
        with torch.inference_mode():
            vision_out = self.model.vision_model(pixel_values=pixel_values)
            features   = self.model.visual_projection(vision_out.pooler_output)
        return F.normalize(features, dim=-1).detach().cpu()

    def embed_text(self, text: str) -> torch.Tensor:
        """
        Generate L2-normalized text embedding.

        CLIP text input is capped at 77 tokens; longer text is silently
        truncated by the processor (truncation=True). OCR snippets from
        product labels are well within this limit.

        Args:
            text: raw string (OCR output or search query)

        Returns:
            Embedding tensor of shape (1, 512)

        TODO: Add chunking for long documents once migrated to server.
        """
        enc = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        with torch.inference_mode():
            text_out = self.model.text_model(input_ids=input_ids, attention_mask=attention_mask)
            features = self.model.text_projection(text_out.pooler_output)
        result = F.normalize(features, dim=-1).detach().cpu()

        _log.info({
            "event": "text_embedding",
            "text_length": len(text),
            "embedding_shape": list(result.shape),
        })
        return result
