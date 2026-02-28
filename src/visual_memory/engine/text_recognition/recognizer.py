"""Text recognition engine using EasyOCR."""
import numpy as np
from PIL import Image

from visual_memory.config import Settings
from visual_memory.utils import get_logger

_settings = Settings()
_log = get_logger(__name__)


class TextRecognizer:
    def __init__(self, languages=None, gpu=None) -> None:
        import easyocr
        if languages is None:
            languages = _settings.ocr_languages
        if gpu is None:
            import torch
            # EasyOCR does not support MPS; fall back to CPU on Apple Silicon
            gpu = torch.cuda.is_available()
        self.reader = easyocr.Reader(languages, gpu=gpu)
        self._min_confidence = _settings.ocr_min_confidence

    def recognize(self, image: Image.Image) -> dict:
        """
        Run OCR on a PIL Image.

        Returns:
            dict with keys:
                "text": str — all recognized text joined with spaces
                "confidence": float — average confidence across accepted segments
                "segments": list of (text, confidence) tuples
        """
        img_np = np.array(image.convert("RGB"))
        raw = self.reader.readtext(img_np)

        segments = [
            (text, float(conf))
            for (_, text, conf) in raw
            if float(conf) >= self._min_confidence
        ]

        if not segments:
            result = {"text": "", "confidence": 0.0, "segments": []}
        else:
            texts = [t for t, _ in segments]
            confs = [c for _, c in segments]
            result = {
                "text": " ".join(texts),
                "confidence": float(sum(confs) / len(confs)),
                "segments": segments,
            }

        _log.info({
            "event": "text_recognition",
            "text_length": len(result["text"]),
            "segment_count": len(segments),
            "avg_confidence": round(result["confidence"], 4),
        })
        return result
