"""Text recognition engine factory."""

from .base import BaseTextRecognizer
from .http_recognizer import HTTPOCRRecognizer


def make_recognizer() -> BaseTextRecognizer:
    return HTTPOCRRecognizer()


TextRecognizer = make_recognizer
