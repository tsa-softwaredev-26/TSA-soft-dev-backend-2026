"""
Embedding module: image (DINOv3), text (CLIP text encoder), and combined.
"""

from .embed_image import ImageEmbedder
from .embed_text import CLIPTextEmbedder
from .embed_combined import make_combined_embedding
