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
