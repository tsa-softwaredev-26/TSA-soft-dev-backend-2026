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

    # Image embedder (both modes) — DINOv3 vision-only, better object discrimination
    image_embedder_model: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"

    # Text embedder (both modes) — CLIP text encoder only (no vision model loaded)
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

    # Pipeline feature toggles — set False for lightweight / faster runs
    enable_depth: bool = True    # load + run DepthEstimator (Depth Pro, ~2GB); disable to skip distance/direction
    enable_ocr: bool = True      # run TextRecognizer on each crop; disable for speed (image-only matching)
    enable_dedup: bool = True    # deduplicate overlapping match boxes after Pass 1
