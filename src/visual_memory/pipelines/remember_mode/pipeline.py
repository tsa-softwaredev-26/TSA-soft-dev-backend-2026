from __future__ import annotations
from pathlib import Path
import pillow_heif

from visual_memory.utils import load_image, crop_object
from visual_memory.engine.object_detection import GroundingDinoDetector
from visual_memory.engine.embedding import ImageEmbedder

pillow_heif.register_heif_opener()


class RememberPipeline:
    def __init__(self):
        self.detector = GroundingDinoDetector()
        self.embedder = ImageEmbedder()

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

        # Create embedding
        embedding = self.embedder.embed(cropped)

        # Hook for database storage
        self.add_to_database(
            embedding,
            metadata={
                "label": label,
                "location": None,
                "confidence": float(score),
                "timestamp": None,
                "image_path": str(image_path)
            }
        )

        return {
            "success": True,
            "message": "Object detected and embedded successfully.",
            "result": {
                "label": label,
                "confidence": float(score),
                "box": box
            }
        }