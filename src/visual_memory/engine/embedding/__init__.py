"""
Embedding module: image + text via CLIP (shared embedding space), and combined.
"""

from .embed_image import CLIPEmbedder
from .embed_combined import make_combined_embedding
