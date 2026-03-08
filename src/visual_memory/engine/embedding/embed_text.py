"""Text embedding module using CLIP text encoder."""
from typing import Optional, List
import torch
import torch.nn.functional as F
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from visual_memory.config import Settings
from visual_memory.utils import get_logger
from visual_memory.utils.device_utils import get_device

_defaults = Settings()
_log = get_logger(__name__)


class CLIPTextEmbedder:
    """
    Text embedder using the CLIP text encoder only (openai/clip-vit-base-patch32).

    Loads only the text encoder + projection head, not the full CLIP vision model.
    Returns 512-dim L2-normalized embeddings in the CLIP shared text space.
    """

    def __init__(
        self,
        model_name: str = _defaults.embedder_model,
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device if device else get_device())
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModelWithProjection.from_pretrained(model_name).to(self.device).eval()

    def embed_text(self, text: str) -> torch.Tensor:
        """
        Generate L2-normalized text embedding.

        Args:
            text: raw string (OCR output or search query)

        Returns:
            Embedding tensor of shape (1, 512)
        """
        CHUNK = 75  # content tokens per chunk (77 - 2 special tokens)

        raw = self.tokenizer(
            [text], return_tensors="pt", padding=False,
            truncation=False, add_special_tokens=False,
        )
        token_ids = raw["input_ids"][0]

        if len(token_ids) <= CHUNK:
            enc = self.tokenizer(
                [text], return_tensors="pt", padding=True,
                truncation=True, max_length=77,
            )
            input_ids = enc["input_ids"].to(self.device)
            attn_mask = enc["attention_mask"].to(self.device)
            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, attention_mask=attn_mask)
            result = F.normalize(outputs.text_embeds, dim=-1).detach().cpu()
        else:
            bos = self.tokenizer.bos_token_id
            eos = self.tokenizer.eos_token_id
            pad = self.tokenizer.pad_token_id

            all_ids, all_masks = [], []
            for start in range(0, len(token_ids), CHUNK):
                chunk = token_ids[start : start + CHUNK].tolist()
                ids   = [bos] + chunk + [eos]
                pad_n = 77 - len(ids)
                all_ids.append(ids + [pad] * pad_n)
                all_masks.append([1] * len(ids) + [0] * pad_n)

            input_ids = torch.tensor(all_ids,   dtype=torch.long).to(self.device)
            attn_mask = torch.tensor(all_masks, dtype=torch.long).to(self.device)
            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, attention_mask=attn_mask)
            pooled = outputs.text_embeds.mean(dim=0, keepdim=True)
            result = F.normalize(pooled, dim=-1).detach().cpu()

        _log.info({
            "event": "text_embedding",
            "text_length": len(text),
            "n_chunks": max(1, (len(token_ids) + CHUNK - 1) // CHUNK),
            "embedding_shape": list(result.shape),
        })
        return result

    def batch_embed_text(self, texts: List[str]) -> torch.Tensor:
        """Embed a list of texts in a single forward pass.

        Returns (N, 512) L2-normalized tensor. Texts exceeding 77 tokens
        are truncated; use embed_text() for long documents with chunking.
        Preferred over calling embed_text() in a loop on CUDA/MPS.
        """
        enc = self.tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=77,
        )
        input_ids = enc["input_ids"].to(self.device)
        attn_mask = enc["attention_mask"].to(self.device)
        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attn_mask)
        return F.normalize(outputs.text_embeds, dim=-1).detach().cpu()
