"""Combined image+text embedding for similarity matching."""
import torch
import torch.nn.functional as F

_TEXT_DIM = 384  # all-MiniLM-L6-v2 output dimension


def make_combined_embedding(
    img_emb: torch.Tensor,
    text_emb: torch.Tensor | None,
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
        text_emb: (1, 384) text embedding tensor, or None

    Returns:
        (1, img_dim + 384) normalized+concatenated tensor
    """
    img_norm = F.normalize(img_emb, dim=1)
    if text_emb is None:
        text_part = torch.zeros(1, _TEXT_DIM)
    else:
        text_part = F.normalize(text_emb, dim=1)
    return torch.cat([img_norm, text_part], dim=1)
