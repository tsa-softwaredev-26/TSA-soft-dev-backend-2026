"""Text recognition engine — factory returns the active PaddleOCR backend."""

from .base import BaseTextRecognizer
from .paddle_recognizer import PaddleOCRRecognizer


def make_recognizer() -> BaseTextRecognizer:
    return PaddleOCRRecognizer()


# Pipelines do: self.text_recognizer = TextRecognizer()
TextRecognizer = make_recognizer
