from __future__ import annotations
import time
from pathlib import Path
import pillow_heif

from visual_memory.config import Settings
from visual_memory.utils import load_image, crop_object, get_logger
from visual_memory.engine.embedding import make_combined_embedding
from visual_memory.engine.model_registry import registry
from visual_memory.database import DatabaseStore

pillow_heif.register_heif_opener()

_settings = Settings()
_log = get_logger(__name__)


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

    def run(self, image_path: str | Path, prompt: str):
        """
        image_path: path to image file
        prompt: text prompt
        returns structured result dict
        """

        # Ensure Path object
        image_path = Path(image_path)

        # Load image from disk
        image = load_image(str(image_path))

        # Run detection
        detection = self.detector.detect(image, prompt)

        if not detection:
            return {
                "success": False,
                "message": "No object detected.",
                "result": None
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

        # Hook for database storage
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

        result = {
            "label": label,
            "confidence": float(score),
            "box": box,
            "ocr_text": ocr_text,
            "ocr_confidence": round(ocr_confidence, 4),
        }

        return {
            "success": True,
            "message": "Object detected and embedded successfully.",
            "result": result,
        }