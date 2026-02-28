from __future__ import annotations
from pathlib import Path
import pillow_heif

from visual_memory.utils import load_image, crop_object, get_logger
from visual_memory.engine.object_detection import GroundingDinoDetector
from visual_memory.engine.embedding import ImageEmbedder, TextEmbedder
from visual_memory.engine.text_recognition import TextRecognizer

pillow_heif.register_heif_opener()

_log = get_logger(__name__)


class RememberPipeline:
    def __init__(self):
        self.detector = GroundingDinoDetector()
        self.embedder = ImageEmbedder()
        self.text_recognizer = TextRecognizer()
        self.text_embedder = TextEmbedder()

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
        embedding = self.embedder.embed(cropped)

        # OCR: extract text from the crop
        ocr_result = self.text_recognizer.recognize(cropped)
        text_embedding = None
        if ocr_result["text"]:
            text_embedding = self.text_embedder.embed(ocr_result["text"])

        _log.info({
            "event": "remember_ocr",
            "label": label,
            "ocr_text_length": len(ocr_result["text"]),
            "ocr_confidence": round(ocr_result["confidence"], 4),
            "has_text_embedding": text_embedding is not None,
        })

        # Hook for database storage
        self.add_to_database(
            embedding,
            metadata={
                "label": label,
                "location": None,
                "confidence": float(score),
                "timestamp": None,
                "image_path": str(image_path),
                "ocr_text": ocr_result["text"],
                "text_embedding": text_embedding,
            }
        )

        result = {
            "label": label,
            "confidence": float(score),
            "box": box,
            "ocr_text": ocr_result["text"],
            "ocr_confidence": round(ocr_result["confidence"], 4),
        }

        return {
            "success": True,
            "message": "Object detected and embedded successfully.",
            "result": result,
        }