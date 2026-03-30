from __future__ import annotations
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
import pillow_heif
from PIL import Image

from visual_memory.config import Settings
from visual_memory.utils import load_image, crop_object, get_logger, mean_luminance, estimate_text_likelihood, collect_system_metrics
from visual_memory.utils.logger import LogTag
from visual_memory.engine.embedding import make_combined_embedding
from visual_memory.engine.model_registry import registry
from visual_memory.engine.text_recognition import TextRecognizer
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


# image quality helpers

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


# detection quality helpers

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


# pipeline

class RememberPipeline:
    def __init__(self):
        self.detector        = registry.gdino_detector
        self.img_embedder    = registry.img_embedder
        self.text_embedder   = registry.text_embedder   if _settings.enable_ocr else None
        self.ocr_client      = TextRecognizer() if _settings.enable_ocr else None

        self.db = DatabaseStore(Path(_settings.db_path))

    def add_to_database(self, embedding, metadata):
        ocr_text = metadata.get("ocr_text", "")
        ocr_embedding = None
        if ocr_text and self.text_embedder is not None:
            ocr_embedding = self.text_embedder.embed_text(ocr_text)

        self.db.add_item(
            label=metadata["label"],
            combined_embedding=embedding,
            ocr_text=ocr_text,
            image_path=metadata.get("image_path", ""),
            confidence=metadata.get("confidence", 0.0),
            timestamp=time.time(),
            label_embedding=metadata.get("label_embedding"),
            ocr_embedding=ocr_embedding,
        )

    def _detect_with_ollama_fallback(
        self,
        image: Image.Image,
        prompt: str,
    ) -> Tuple[Optional[dict], Optional[str]]:
        """Try Ollama-suggested prompt variants and return first successful detect."""
        from visual_memory.utils.ollama_utils import _chat
        n = max(1, _settings.ollama_detection_retries)
        ollama_prompt = (
            f"A vision model failed to detect '{prompt}' in an image. "
            f"Give {n} alternative phrasings that a vision model might recognize better. "
            f"Reply with only the phrases, one per line, no numbering, no explanation."
        )
        suggestions_raw = _chat(ollama_prompt)
        if suggestions_raw:
            suggestions = [s.strip() for s in suggestions_raw.splitlines() if s.strip()][:n]
            for suggestion in suggestions:
                detection = self.detector.detect(image, suggestion)
                if detection is not None:
                    _log.info({
                        "event": "remember_third_pass_ollama",
                        "original_prompt": prompt,
                        "successful_prompt": suggestion,
                    })
                    return detection, suggestion
        return None, None

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

        return self._detect_with_ollama_fallback(image, prompt)

    def detect_score_batch(
        self,
        image_paths: Sequence[Union[str, Path]],
        prompt: str,
    ) -> List[dict]:
        """
        Batched detection-only pass for multi-image remember ranking.

        Uses Grounding DINO detect_batch() for first and second pass prompts.
        """
        registry.prepare_for_remember()
        paths = [Path(p) for p in image_paths]
        images = [load_image(str(p)) for p in paths]

        luminances = [mean_luminance(img) for img in images]
        blurs = [_blur_score(img) for img in images]
        is_dark = [lum < _settings.darkness_threshold for lum in luminances]

        results: List[dict] = []
        detections: List[Optional[dict]] = [None] * len(images)
        second_pass_prompt: List[Optional[str]] = [None] * len(images)

        active_indices = [i for i, dark in enumerate(is_dark) if not dark]
        if active_indices:
            first_pass = self.detector.detect_batch(
                [images[i] for i in active_indices],
                [prompt] * len(active_indices),
            )
            for idx, detection in zip(active_indices, first_pass):
                detections[idx] = detection

            if _settings.detection_second_pass_enabled:
                pending = [i for i in active_indices if detections[i] is None]
                for template in _SECOND_PASS_TEMPLATES:
                    if not pending:
                        break
                    alt_prompt = template.format(prompt=prompt)
                    second_pass = self.detector.detect_batch(
                        [images[i] for i in pending],
                        [alt_prompt] * len(pending),
                    )
                    unresolved = []
                    for idx, detection in zip(pending, second_pass):
                        if detection is None:
                            unresolved.append(idx)
                            continue
                        detections[idx] = detection
                        second_pass_prompt[idx] = alt_prompt
                        _log.info({
                            "event": "remember_second_pass",
                            "original_prompt": prompt,
                            "successful_prompt": alt_prompt,
                        })
                    pending = unresolved

                for idx in pending:
                    detection, third_pass_prompt = self._detect_with_ollama_fallback(images[idx], prompt)
                    detections[idx] = detection
                    second_pass_prompt[idx] = third_pass_prompt

        for i in range(len(images)):
            if is_dark[i]:
                results.append({
                    "detected": False,
                    "score": 0.0,
                    "blur_score": blurs[i],
                    "is_dark": True,
                    "darkness_level": round(luminances[i], 2),
                    "second_pass_prompt": None,
                })
                continue
            detection = detections[i]
            results.append({
                "detected": detection is not None,
                "score": detection["score"] if detection else 0.0,
                "blur_score": blurs[i],
                "is_dark": False,
                "darkness_level": round(luminances[i], 2),
                "second_pass_prompt": second_pass_prompt[i],
            })
        return results

    def detect_score(self, image_path: Union[str, Path], prompt: str) -> dict:
        """
        Detection-only pass - no embedding or DB write. Used by the multi-image
        endpoint to rank candidates before committing to a full run.

        Returns:
            detected (bool), score (float), blur_score (float),
            is_dark (bool), darkness_level (float),
            second_pass_prompt (Optional[str])
        """
        registry.prepare_for_remember()
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

    def run(self, image_path: Union[str, Path], prompt: str):
        """
        image_path: path to image file
        prompt: text label from the user
        returns structured result dict
        """
        t0 = time.monotonic()
        stage_start = t0
        timing = {
            "prepare_ms": 0,
            "detect_ms": 0,
            "embed_ms": 0,
            "ocr_ms": 0,
            "db_ms": 0,
        }
        registry.prepare_for_remember()
        timing["prepare_ms"] = round((time.monotonic() - stage_start) * 1000)
        image_path = Path(image_path)
        image = load_image(str(image_path))

        # Darkness check - must come before any detection attempt
        lum = mean_luminance(image)
        if lum < _settings.darkness_threshold:
            _log.info({
                "event": "remember_complete",
                "tag": LogTag.PERF,
                "success": False,
                "reason": "dark",
                "label": prompt,
                "duration_ms": round((time.monotonic() - t0) * 1000),
                **timing,
                **collect_system_metrics(),
            })
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
        stage_start = time.monotonic()
        detection, second_pass_prompt = self._detect_with_fallback(image, prompt)
        timing["detect_ms"] = round((time.monotonic() - stage_start) * 1000)

        if not detection:
            _log.info({
                "event": "remember_complete",
                "tag": LogTag.PERF,
                "success": False,
                "reason": "no_detection",
                "label": prompt,
                "duration_ms": round((time.monotonic() - t0) * 1000),
                **timing,
                **collect_system_metrics(),
            })
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
        # Always use the user's original prompt as the canonical label so that
        # second-pass reformulations ("a wallet") don't pollute the stored name.
        label = prompt
        score = detection["score"]

        # Crop detected region
        cropped = crop_object(image, box)

        # Create image embedding
        stage_start = time.monotonic()
        embedding = self.img_embedder.embed(cropped)
        timing["embed_ms"] = round((time.monotonic() - stage_start) * 1000)

        # OCR: extract text from the crop (skipped if enable_ocr=False or below
        # text likelihood threshold - saves 3-18s per image on plain objects)
        text_likelihood = estimate_text_likelihood(cropped)
        ocr_text = ""
        ocr_confidence = 0.0
        text_embedding = None
        ocr_t0 = time.monotonic()
        if (self.ocr_client is not None
                and text_likelihood >= _settings.ocr_text_likelihood_threshold):
            ocr_result = self.ocr_client.recognize(cropped)
            ocr_text = ocr_result["text"]
            ocr_confidence = ocr_result["confidence"]
            if ocr_text:
                text_embedding = self.text_embedder.embed_text(ocr_text)
        timing["ocr_ms"] = round((time.monotonic() - ocr_t0) * 1000)

        _log.info({
            "event": "remember_ocr",
            "label": label,
            "text_likelihood": round(text_likelihood, 3),
            "ocr_ran": self.ocr_client is not None and text_likelihood >= _settings.ocr_text_likelihood_threshold,
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
        avg_conf = self.db.get_label_avg_confidence(label)

        # Auto-replace: remove any previous teaches for this label before storing
        # the new one. Queried avg_conf already captured the history above.
        db_t0 = time.monotonic()
        replaced_count = self.db.delete_items_by_label(label)

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
        timing["db_ms"] = round((time.monotonic() - db_t0) * 1000)

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
            "replaced_previous": replaced_count > 0,
            "second_pass": second_pass_prompt is not None,
            "second_pass_prompt": second_pass_prompt,
            "box": box,
            "text_likelihood": round(text_likelihood, 3),
            "ocr_text": ocr_text,
            "ocr_confidence": round(ocr_confidence, 4),
        }

        _log.info({
            "event": "remember_complete",
            "tag": LogTag.PERF,
            "success": True,
            "label": label,
            "detection_quality": quality,
            "second_pass": second_pass_prompt is not None,
            "duration_ms": round((time.monotonic() - t0) * 1000),
            **timing,
            **collect_system_metrics(),
        })
        return {
            "success": True,
            "message": "Object detected and embedded successfully.",
            "result": result,
        }
