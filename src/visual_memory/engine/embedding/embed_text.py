"""Text embedding module using CLIP text encoder."""
from typing import Optional
import torch
import torch.nn.functional as F
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from visual_memory.config import Settings
from visual_memory.utils import get_logger

_defaults = Settings()
_log = get_logger(__name__)


class CLIPTextEmbedder:
    """
    Text embedder using the CLIP text encoder only (openai/clip-vit-base-patch32).

    Loads only the text encoder + projection head, not the full CLIP vision model.
    Returns 512-dim L2-normalized embeddings in the CLIP shared text space.

    CLIP text input is capped at 77 tokens; longer text is silently truncated.
    OCR snippets from product labels are well within this limit.

    TODO (server migration): add batch_embed_text(texts) for GPU throughput.
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
                # TODO (server migration): remove MPS branch; server uses CUDA only.
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")  # CUDA for later server deployment
            else:
                self.device = torch.device("cpu")

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModelWithProjection.from_pretrained(model_name).to(self.device).eval()

    def embed_text(self, text: str) -> torch.Tensor:
        """
        Generate L2-normalized text embedding.

        Args:
            text: raw string (OCR output or search query)

        Returns:
            Embedding tensor of shape (1, 512)

        TODO: Add chunking for long documents once migrated to server.
        """
        enc = self.tokenizer(
            [text], return_tensors="pt", padding=True, truncation=True, max_length=77
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        result = F.normalize(outputs.text_embeds, dim=-1).detach().cpu()

        _log.info({
            "event": "text_embedding",
            "text_length": len(text),
            "embedding_shape": list(result.shape),
        })
        return result
