"""Text embedding module using sentence-transformers."""
from typing import Optional
import torch

from visual_memory.config import Settings
from visual_memory.utils import get_logger

_settings = Settings()
_log = get_logger(__name__)


class TextEmbedder:
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        if model_name is None:
            model_name = _settings.text_embedder_model

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, text: str) -> torch.Tensor:
        """
        Generate embedding for a text string.

        Returns:
            Tensor of shape (1, 384)
        """
        embedding = self.model.encode([text], convert_to_tensor=True)
        result = embedding.detach().cpu()

        _log.info({
            "event": "text_embedding",
            "text_length": len(text),
            "embedding_shape": list(result.shape),
        })
        return result
