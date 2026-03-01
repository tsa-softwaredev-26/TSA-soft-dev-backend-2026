from __future__ import annotations
from pathlib import Path
import pillow_heif

from visual_memory.config import Settings
from visual_memory.utils import load_image, crop_object, get_logger
from visual_memory.engine.object_detection import GroundingDinoDetector
from visual_memory.engine.embedding import ImageEmbedder, CLIPTextEmbedder, make_combined_embedding
from visual_memory.engine.text_recognition import TextRecognizer

pillow_heif.register_heif_opener()

_settings = Settings()
_log = get_logger(__name__)


class RememberPipeline:
    def __init__(self):
        self.detector = GroundingDinoDetector()
        self.img_embedder = ImageEmbedder()
        self.text_embedder = CLIPTextEmbedder() if _settings.enable_ocr else None
        self.text_recognizer = TextRecognizer() if _settings.enable_ocr else None

        # TODO: replace with real database later
        self.database = None

    def add_to_database(self, embedding, metadata):
        """
        Placeholder for future database integration.
        For now this does nothing.
        """
        # Later this could:
        # - insert into SQLite
        # - save to vector DB
        # - write to file system
        pass

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