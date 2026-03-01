"""
Embedding module: image (DINOv3), text (sentence-transformers), and combined.
"""

from .embed_image import ImageEmbedder
from .embed_text import TextEmbedder
from .embed_combined import make_combined_embedding