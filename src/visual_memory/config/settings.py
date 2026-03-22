from dataclasses import dataclass, field
import os


@dataclass
class Settings:
    # GroundingDINO (remember mode)
    grounding_dino_model: str = "IDEA-Research/grounding-dino-base"
    box_threshold: float = 0.3
    text_threshold: float = 0.25

    # YOLOE (scan mode)
    yoloe_confidence: float = 0.35
    yoloe_iou: float = 0.45

    # Image embedder (both modes) - DINOv3 vision-only, better object discrimination
    image_embedder_model: str = "facebook/dinov3-vits16-pretrain-lvd1689m"

    # Text embedder (both modes) - CLIP text encoder only (no vision model loaded)
    embedder_model: str = "openai/clip-vit-base-patch32"

    # Similarity matching (scan mode)
    similarity_threshold: float = 0.3
    dedup_iou_threshold: float = 0.5

    # Narration (scan mode)
    narration_high_confidence: float = 0.6

    # OCR (text recognition)
    ocr_backend: str = "paddle"
    ocr_languages: list = field(default_factory=lambda: ["en"])
    ocr_min_confidence: float = 0.3

    # Text similarity (CLIP text embeddings)
    text_similarity_threshold: float = 0.4

    # Pipeline feature toggles - overridable via env vars before pipeline import
    enable_depth: bool = field(default_factory=lambda: os.environ.get("ENABLE_DEPTH", "1") != "0")
    enable_ocr:   bool = field(default_factory=lambda: os.environ.get("ENABLE_OCR",   "1") != "0")
    enable_dedup: bool = field(default_factory=lambda: os.environ.get("ENABLE_DEDUP", "1") != "0")

    # Projection head (personalization)
    projection_head_path: str = "models/projection_head.pt"
    projection_head_dim: int = 1536

    # Database
    db_path: str = "data/memory.db"

    # API server
    api_host: str = "127.0.0.1"
    api_port: int = 5000
