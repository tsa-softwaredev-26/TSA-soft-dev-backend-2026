"""Combined image+text embedding for similarity matching."""
import torch
import torch.nn.functional as F

_TEXT_DIM = 512  # CLIP text encoder output dimension (paired with DINOv3 1024-dim image embeddings)


def make_combined_embedding(
    img_emb: torch.Tensor,
    text_emb: torch.Tensor | None,
    *,
    text_weight: float = 1.0,
    text_confidence: float | None = None,
    text_high_confidence_boost: float = 0.0,
    image_weight: float = 1.0,
) -> torch.Tensor:
    """
    Normalize each part then concatenate into a single embedding.

    When text_emb is None (no OCR text), the text slot is zeros, so
    cosine similarity is dominated by the image component alone.
    When both parts are present, similarity = weighted average of image
    and text sub-similarities, preventing false matches between documents
    that look visually similar but contain different text.

    Args:
        img_emb:  (1, img_dim) image embedding tensor
        text_emb: (1, 512) text embedding tensor, or None

    Returns:
        (1, img_dim + 512) weighted concatenated tensor
    """
    img_norm = F.normalize(img_emb, dim=1)
    img_norm = img_norm.to(dtype=torch.float32)
    text_scale = max(0.0, float(text_weight))
    if text_confidence is not None:
        conf = min(max(float(text_confidence), 0.0), 1.0)
        text_scale += max(0.0, float(text_high_confidence_boost)) * conf
    image_scale = max(0.0, float(image_weight))

    if text_emb is None:
        text_part = img_norm.new_zeros((img_norm.shape[0], _TEXT_DIM))
        text_scale = 0.0
    else:
        text_part = F.normalize(text_emb, dim=1).to(dtype=img_norm.dtype, device=img_norm.device)

    if image_scale == 0.0 and text_scale == 0.0:
        image_scale = 1.0

    return torch.cat([img_norm * image_scale, text_part * text_scale], dim=1)
