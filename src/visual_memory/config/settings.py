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

    # Learning / online feedback training
    # enable_learning: apply projection head during scan (can be toggled via PATCH /settings)
    enable_learning: bool = field(default_factory=lambda: os.environ.get("ENABLE_LEARNING", "1") != "0")
    # min triplets (pos+neg pairs) required before /retrain will proceed
    min_feedback_for_training: int = 10
    # max blend weight between raw embedding (0.0) and projected embedding (1.0)
    # acts as a ceiling; actual weight ramps up automatically with triplet count
    projection_head_weight: float = 1.0
    # triplet count at which projection_head_weight is fully reached (linear ramp from 0)
    projection_head_ramp_at: int = 50
    # training epochs for each /retrain call
    projection_head_epochs: int = 20

    # Database
    db_path: str = "data/memory.db"

    # API server
    api_host: str = "127.0.0.1"
    api_port: int = 5000
