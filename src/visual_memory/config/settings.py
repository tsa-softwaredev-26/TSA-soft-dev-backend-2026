from dataclasses import dataclass


@dataclass
class Settings:
    # GroundingDINO (remember mode)
    grounding_dino_model: str = "IDEA-Research/grounding-dino-base"
    box_threshold: float = 0.5
    text_threshold: float = 0.3

    # YOLOE (scan mode)
    yoloe_confidence: float = 0.5
    yoloe_iou: float = 0.45

    # Embedder (both modes)
    embedder_model: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"

    # Similarity matching (scan mode)
    similarity_threshold: float = 0.3
    dedup_iou_threshold: float = 0.5

    # Narration (scan mode) 
    narration_high_confidence: float = 0.6
