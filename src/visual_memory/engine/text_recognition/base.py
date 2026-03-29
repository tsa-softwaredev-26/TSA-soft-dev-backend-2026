"""Abstract base class for text recognition engines."""
from abc import ABC, abstractmethod
from PIL import Image


class BaseTextRecognizer(ABC):
    @abstractmethod
    def recognize(self, image: Image.Image) -> dict:
        """
        Run OCR on a PIL Image.

        Returns:
            dict with keys:
                "text": str - all recognized text joined with spaces
                "confidence": float - average confidence across accepted segments
                "segments": list of (text, confidence) tuples
        """

    def recognize_batch(self, images: list[Image.Image]) -> list[dict]:
        """Run OCR over a batch of PIL Images.

        Default behavior calls recognize() per image to preserve compatibility
        for recognizers that do not implement a true batch backend.
        """
        return [self.recognize(img) for img in images]
