from dataclasses import dataclass, field


@dataclass
class Settings:
    # GroundingDINO (remember mode)
    grounding_dino_model: str = "IDEA-Research/grounding-dino-base"
    box_threshold: float = 0.5
    text_threshold: float = 0.3

    # YOLOE (scan mode)
    yoloe_confidence: float = 0.5
    yoloe_iou: float = 0.45

    # Embedder (both modes) — CLIP shared image+text space
    embedder_model: str = "openai/clip-vit-base-patch32"

    # Similarity matching (scan mode)
    similarity_threshold: float = 0.2
    dedup_iou_threshold: float = 0.5

    # Narration (scan mode)
    narration_high_confidence: float = 0.6

    # OCR (text recognition)
    ocr_backend: str = "paddle"
    ocr_languages: list = field(default_factory=lambda: ["en"])
    ocr_min_confidence: float = 0.3

    # Text similarity (CLIP text embeddings, same space as image)
    text_similarity_threshold: float = 0.3
