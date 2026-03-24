from __future__ import annotations
import time
from pathlib import Path
from typing import Optional
import numpy as np
import pillow_heif
from PIL import Image

from visual_memory.config import Settings
from visual_memory.utils import load_image, crop_object, get_logger, mean_luminance
from visual_memory.engine.embedding import make_combined_embedding
from visual_memory.engine.model_registry import registry
from visual_memory.database import DatabaseStore

pillow_heif.register_heif_opener()

_settings = Settings()
_log = get_logger(__name__)

# Reformulated prompts tried in order when initial detection returns nothing.
# {prompt} is substituted with the user's original label.
_SECOND_PASS_TEMPLATES = [
    "a {prompt}",
    "{prompt} object",
    "close up of a {prompt}",
]


# ---- image quality helpers ----

def _blur_score(image: Image.Image) -> float:
    """
    Laplacian variance of the image. Higher = sharper.
    Uses 4-neighbor discrete Laplacian via numpy; no extra dependencies.
    """
    gray = np.array(image.convert("L"), dtype=np.float32)
    lap = (
        np.roll(gray, 1, 0) + np.roll(gray, -1, 0) +
        np.roll(gray, 1, 1) + np.roll(gray, -1, 1) - 4.0 * gray
    )
    return float(lap.var())


# ---- detection quality helpers ----

def _score_quality(score: float, avg_conf: Optional[float]) -> str:
    """
    Classify detection score as "low", "medium", or "high".

    When the label has historical average confidence (avg_conf), the score is
    normalized by that average so labels that are inherently hard to detect are
    not penalized relative to their own baseline.

    Without history, absolute thresholds from settings apply.
    """
    if avg_conf and avg_conf > 0:
        relative = score / avg_conf
        if relative < 0.75:
            return "low"
        if relative < 1.15:
            return "medium"
        return "high"

    if score < _settings.detection_quality_low_max:
        return "low"
    if score < _settings.detection_quality_high_min:
        return "medium"
    return "high"


def _quality_hint(quality: str, is_blurry: bool) -> str:
    if quality == "high" and not is_blurry:
        return "Detection confidence is high."
    if quality == "high":
        return (
            "Detection confidence is high, but image appears blurry. "
            "Consider a steadier shot for better future scans."
        )
    if quality == "medium" and is_blurry:
        return (
            "Moderate confidence and blurry image. "
            "A steadier shot or better lighting may improve results."
        )
    if quality == "medium":
        return "Moderate detection confidence. A different angle or closer shot may help."
    if is_blurry:
        return (
            "Low confidence and blurry image. "
            "Steady the camera, move closer, or improve lighting."
        )
    return (
        "Low confidence. Try a more descriptive label, zoom in, or change the angle."
    )


# ---- pipeline ----

class RememberPipeline:
    def __init__(self):
        self.detector        = registry.gdino_detector
        self.img_embedder    = registry.img_embedder
        self.text_embedder   = registry.text_embedder   if _settings.enable_ocr else None
        self.text_recognizer = registry.text_recognizer if _settings.enable_ocr else None

        self.db = DatabaseStore(Path(_settings.db_path))

    def add_to_database(self, embedding, metadata):
        self.db.add_item(
            label=metadata["label"],
            combined_embedding=embedding,
            ocr_text=metadata.get("ocr_text", ""),
            image_path=metadata.get("image_path", ""),
            confidence=metadata.get("confidence", 0.0),
            timestamp=time.time(),
            label_embedding=metadata.get("label_embedding"),
        )

    def _detect_with_fallback(self, image: Image.Image, prompt: str):
        """
        Run detection on image. If nothing is found, retry with second-pass
        reformulated prompts (when detection_second_pass_enabled is True).

        Returns (detection_or_None, second_pass_prompt_or_None).
        """
        detection = self.detector.detect(image, prompt)
        if detection is not None:
            return detection, None

        if not _settings.detection_second_pass_enabled:
            return None, None

        for template in _SECOND_PASS_TEMPLATES:
            alt_prompt = template.format(prompt=prompt)
            detection = self.detector.detect(image, alt_prompt)
            if detection is not None:
                _log.info({
                    "event": "remember_second_pass",
                    "original_prompt": prompt,
                    "successful_prompt": alt_prompt,
                })
                return detection, alt_prompt

        return None, None

    def detect_score(self, image_path: str | Path, prompt: str) -> dict:
        """
        Detection-only pass - no embedding or DB write. Used by the multi-image
        endpoint to rank candidates before committing to a full run.

        Returns:
            detected (bool), score (float), blur_score (float),
            is_dark (bool), darkness_level (float),
            second_pass_prompt (str | None)
        """
        image = load_image(str(Path(image_path)))
        lum = mean_luminance(image)
        is_dark = lum < _settings.darkness_threshold
        blur = _blur_score(image)
        if is_dark:
            return {
                "detected": False,
                "score": 0.0,
                "blur_score": blur,
                "is_dark": True,
                "darkness_level": round(lum, 2),
                "second_pass_prompt": None,
            }
        detection, second_pass_prompt = self._detect_with_fallback(image, prompt)
        return {
            "detected": detection is not None,
            "score": detection["score"] if detection else 0.0,
            "blur_score": blur,
            "is_dark": False,
            "darkness_level": round(lum, 2),
            "second_pass_prompt": second_pass_prompt,
        }

    def run(self, image_path: str | Path, prompt: str):
        """
        image_path: path to image file
        prompt: text label from the user
        returns structured result dict
        """
        image_path = Path(image_path)
        image = load_image(str(image_path))

        # Darkness check - must come before any detection attempt
        lum = mean_luminance(image)
        if lum < _settings.darkness_threshold:
            return {
                "success": False,
                "message": "Image is too dark for detection. Enable the flashlight and retry.",
                "is_dark": True,
                "darkness_level": round(lum, 2),
                "blur_score": round(_blur_score(image), 2),
                "is_blurry": False,
                "result": None,
            }

        # Sharpness check on the full image (before detection / crop)
        blur = _blur_score(image)
        is_blurry = blur < _settings.blur_sharpness_threshold

        # Detection with optional second pass
        detection, second_pass_prompt = self._detect_with_fallback(image, prompt)

        if not detection:
            return {
                "success": False,
                "message": "No object detected.",
                "is_dark": False,
                "darkness_level": round(lum, 2),
                "blur_score": round(blur, 2),
                "is_blurry": is_blurry,
                "result": None,
            }

        box = detection["box"]
        label = detection["label"]
        score = detection["score"]

        # Crop detected region
        cropped = crop_object(image, box)

        # Create image embedding
        embedding = self.img_embedder.embed(cropped)

        # OCR: extract text from the crop (skipped if enable_ocr=False)
        ocr_text = ""
        ocr_confidence = 0.0
        text_embedding = None
        if self.text_recognizer is not None:
            ocr_result = self.text_recognizer.recognize(cropped)
            ocr_text = ocr_result["text"]
            ocr_confidence = ocr_result["confidence"]
            if ocr_text:
                text_embedding = self.text_embedder.embed_text(ocr_text)

        _log.info({
            "event": "remember_ocr",
            "label": label,
            "ocr_text_length": len(ocr_text),
            "ocr_confidence": round(ocr_confidence, 4),
            "has_text_embedding": text_embedding is not None,
        })

        # Build combined embedding (image + text) for consistent matching
        combined = make_combined_embedding(embedding, text_embedding)

        label_embedding = None
        if self.text_embedder is not None:
            label_embedding = self.text_embedder.embed_text(label)

        # Query historical average BEFORE writing the new item
        avg_conf = self.db.get_label_avg_confidence(prompt)

        self.add_to_database(
            combined,
            metadata={
                "label": label,
                "location": None,
                "confidence": float(score),
                "timestamp": None,
                "image_path": str(image_path),
                "ocr_text": ocr_text,
                "label_embedding": label_embedding,
            }
        )

        quality = _score_quality(score, avg_conf)

        result = {
            "label": label,
            "confidence": round(float(score), 4),
            "detection_quality": quality,
            "detection_hint": _quality_hint(quality, is_blurry),
            "blur_score": round(blur, 2),
            "is_blurry": is_blurry,
            "is_dark": False,
            "darkness_level": round(lum, 2),
            "second_pass": second_pass_prompt is not None,
            "second_pass_prompt": second_pass_prompt,
            "box": box,
            "ocr_text": ocr_text,
            "ocr_confidence": round(ocr_confidence, 4),
        }

        return {
            "success": True,
            "message": "Object detected and embedded successfully.",
            "result": result,
        }
