from dataclasses import dataclass, field
import os
from typing import Optional


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
    image_embedder_model: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"

    # Text embedder (both modes) - CLIP text encoder only (no vision model loaded)
    embedder_model: str = "openai/clip-vit-base-patch32"

    # Similarity matching (scan mode)
    similarity_threshold: float = 0.79
    similarity_threshold_baseline: Optional[float] = None
    similarity_threshold_personalized: Optional[float] = None
    similarity_threshold_document: Optional[float] = None
    dedup_iou_threshold: float = 0.5

    # Narration (scan mode)
    narration_high_confidence: float = 0.6

    # Detection quality tiers (remember mode, GroundingDINO score)
    # When no label history: absolute thresholds apply.
    # When label history exists: score is normalized by avg confidence, then tiered.
    # score < detection_quality_low_max  -> "low"
    # score < detection_quality_high_min -> "medium"
    # score >= detection_quality_high_min -> "high"
    detection_quality_low_max: float = 0.40
    detection_quality_high_min: float = 0.65

    # Image sharpness via Laplacian variance (remember mode).
    # Values below this threshold are flagged as blurry.
    # Typical range: 50 (very blurry) to 500+ (sharp). 100 is a good general threshold.
    blur_sharpness_threshold: float = 100.0

    # Second-pass detection (remember mode): retry with reformulated prompts when
    # the first detection attempt returns nothing. Disable to save latency.
    detection_second_pass_enabled: bool = True

    # Darkness detection (both modes).
    # Mean grayscale luminance (0-255) below this value is considered too dark
    # for reliable detection. 30 is conservative - only genuine dark rooms trigger
    # this, not dim lighting. Typical dark room with no lights: mean 5-25.
    darkness_threshold: float = 30.0

    # OCR text pre-check: normalized edge density threshold below which OCR is
    # skipped entirely on a crop. 0.0 = always run OCR, 1.0 = never run.
    # Range 0.0-1.0. Tuned for demo goal: trigger OCR reliably for receipts in
    # good lighting at 1-3ft while preserving low OCR load on textless objects.
    ocr_text_likelihood_threshold: float = 0.30
    # Rescue path threshold for bright, sharp crops that are receipt-like but
    # score below the primary threshold.
    ocr_text_likelihood_rescue_threshold: float = 0.10
    # Rescue path minimum brightness and sharpness to avoid broad OCR fallback.
    ocr_text_likelihood_rescue_min_luminance: float = 40.0
    ocr_text_likelihood_rescue_min_blur_score: float = 130.0

    # OCR (text recognition)
    ocr_backend: str = "http"
    ocr_languages: list = field(default_factory=lambda: ["en"])
    ocr_min_confidence: float = 0.3
    ocr_service_url: str = field(default_factory=lambda: os.environ.get("OCR_SERVICE_URL", "http://127.0.0.1:8001/ocr"))
    ocr_health_url: str = field(default_factory=lambda: os.environ.get("OCR_HEALTH_URL", ""))
    ocr_timeout_seconds: float = field(default_factory=lambda: float(os.environ.get("OCR_TIMEOUT_SECONDS", "3.5")))
    # Upper bound for OCR gating band. OCR runs only when likelihood is within:
    # [ocr_text_likelihood_threshold, ocr_text_likelihood_upper_threshold].
    # Keeps OCR from running on extreme-noise crops while preserving text-like crops.
    ocr_text_likelihood_upper_threshold: float = 0.85

    # Text similarity (CLIP text embeddings)
    text_similarity_threshold: float = 0.4
    # Combined embedding weighting. Text can be boosted relative to image to
    # improve retrieval for text-heavy objects (receipts, labels).
    combined_text_weight: float = 1.10
    combined_text_weight_high_confidence_boost: float = 0.10

    # Pipeline feature toggles - overridable via env vars before pipeline import
    enable_depth: bool = field(default_factory=lambda: os.environ.get("ENABLE_DEPTH", "1") != "0")
    enable_ocr:   bool = field(default_factory=lambda: os.environ.get("ENABLE_OCR",   "1") != "0")
    enable_dedup: bool = field(default_factory=lambda: os.environ.get("ENABLE_DEDUP", "1") != "0")

    # Projection head (personalization)
    projection_head_path: str = "models/projection_head.pt"
    projection_head_dim: int = 1536

    # VRAM management; offload exclusive-pipeline models to CPU RAM between calls.
    # Enable on GPUs with < 8 GB VRAM (e.g. GTX 1060 6 GB).
    # Remember mode keeps GDino + DINOv3 + CLIP on GPU; offloads YOLOE + Depth.
    # Scan mode keeps YOLOE + Depth + DINOv3 + CLIP on GPU; offloads GDino.
    # Transfer cost: ~1-2 s for GDino, ~3-5 s for Depth Pro (PCIe, not disk reload).
    save_vram: bool = field(default_factory=lambda: os.environ.get("SAVE_VRAM", "0") != "0")

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
    # ramp curve power for projection head blend progression:
    # alpha = projection_head_weight * (progress ** projection_head_ramp_power),
    # where progress = clamp(triplet_count / projection_head_ramp_at, 0..1).
    # 1.0 = linear, >1.0 = slower early ramp, <1.0 = faster early ramp.
    projection_head_ramp_power: float = 1.0
    # training epochs for each /retrain call
    projection_head_epochs: int = 20
    # Triplet loss parameters for projection head training.
    # Defaults preserve current behavior.
    triplet_margin: float = 0.2
    triplet_positive_weight: float = 1.0
    triplet_negative_weight: float = 1.0
    # Optional emphasis on hard negatives (negatives too close to anchor).
    # 0.0 disables emphasis.
    triplet_hard_negative_boost: float = 0.0

    # Ollama LLM (used by /ask and /item/ask for query parsing, and remember mode
    # enhanced second-pass detection prompt generation)
    # llm_query_fallback_enabled: gate LLM query parsing fallback in /ask and /item/ask.
    # User settings map this by performance mode (fast=False, balanced/accurate=True).
    llm_query_fallback_enabled: bool = True
    # ollama_max_retries: number of times to retry a failed Ollama call before
    # falling back to embedding-only search or keyword matching.
    ollama_max_retries: int = 2
    # ollama_detection_retries: number of LLM-suggested prompts to try in remember
    # mode when all _SECOND_PASS_TEMPLATES also fail (Ollama-enhanced third pass).
    ollama_detection_retries: int = 2
    # ollama_timeout_seconds: seconds to wait for one Ollama chat response before
    # giving up. Lower values keep /ask latency predictable when Ollama is slow.
    ollama_timeout_seconds: float = field(default_factory=lambda: float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "5.0")))
    # Per-mode Ollama models. Fast mode disables LLM parsing.
    ollama_model_balanced: str = field(default_factory=lambda: os.environ.get("OLLAMA_MODEL_BALANCED", "llama3.2:1b"))
    ollama_model_accurate: str = field(default_factory=lambda: os.environ.get("OLLAMA_MODEL_ACCURATE", "phi3:mini"))

    def get_ollama_model(self, mode: str = "balanced") -> str | None:
        if mode == "fast":
            return None
        if mode == "accurate":
            return self.ollama_model_accurate
        return self.ollama_model_balanced

    def get_similarity_threshold_baseline(self) -> float:
        if self.similarity_threshold_baseline is not None:
            return float(self.similarity_threshold_baseline)
        return float(self.similarity_threshold)

    def get_similarity_threshold_personalized(self) -> float:
        if self.similarity_threshold_personalized is not None:
            return float(self.similarity_threshold_personalized)
        return self.get_similarity_threshold_baseline()

    def get_similarity_threshold_document(self) -> float:
        if self.similarity_threshold_document is not None:
            return float(self.similarity_threshold_document)
        return self.get_similarity_threshold_baseline()

    # Whisper speech recognition (voice input)
    whisper_model: str = "openai/whisper-large-v3-turbo"
    whisper_sample_rate: int = 16000
    whisper_language: str = "en"
    whisper_context_enabled: bool = True
    whisper_keep_warm: bool = field(default_factory=lambda: os.environ.get("WHISPER_KEEP_WARM", "0") != "0")
    whisper_context_max_labels: int = 64
    whisper_context_max_rooms: int = 24
    whisper_context_max_chars: int = 512

    # Database
    db_path: str = "data/memory.db"
    # Per-scan cache TTL for feedback/crop lookup.
    scan_cache_ttl_seconds: int = field(default_factory=lambda: int(os.environ.get("SCAN_CACHE_TTL_SECONDS", "600")))

    # API server
    api_host: str = field(default_factory=lambda: os.environ.get("API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: int(os.environ.get("API_PORT", "5000")))
