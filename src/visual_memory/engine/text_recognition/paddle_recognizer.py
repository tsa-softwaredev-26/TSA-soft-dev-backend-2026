"""Text recognition engine using PaddleOCR-VL-1.5."""
import os
import json
import tempfile
from PIL import Image

from visual_memory.utils import get_logger
from .base import BaseTextRecognizer

_log = get_logger(__name__)

# Skip network check for cached models
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


class PaddleOCRRecognizer(BaseTextRecognizer):
    """OCR via PaddleOCR-VL-1.5 + PP-DocLayoutV3 layout detection.

    Saves PIL image to a temp PNG, passes the file path to predict() (required
    by PaddleOCRVL), then extracts text from the JSON result's
    parsing_res_list[].block_content fields.
    """

    def __init__(self) -> None:
        from paddleocr import PaddleOCRVL
        self.pipeline = PaddleOCRVL()
        _log.info({"event": "paddle_init", "status": "ready"})

    def recognize(self, image: Image.Image) -> dict:
        """Run OCR on a PIL Image. Returns {"text", "confidence", "segments"}."""
        temp_path = None
        try:
            rgb = image.convert("RGB")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                temp_path = f.name
            rgb.save(temp_path, format="PNG")

            results = self.pipeline.predict(temp_path)

            segments = []
            for result in (results or []):
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as jf:
                    json_path = jf.name
                result.save_to_json(json_path)
                with open(json_path, encoding="utf-8") as jf:
                    json_data = json.load(jf)
                os.unlink(json_path)

                text = self._extract_text_from_json(json_data)
                if text:
                    segments.append((text, 1.0))

            if not segments:
                out = {"text": "", "confidence": 0.0, "segments": []}
            else:
                out = {
                    "text": " ".join(t for t, _ in segments),
                    "confidence": 1.0,
                    "segments": segments,
                }

            _log.info({
                "event": "text_recognition",
                "engine": "paddle",
                "status": "success" if segments else "no_segments",
                "segment_count": len(segments),
                "text_length": len(out["text"]),
            })
            return out

        except Exception as exc:
            _log.error({"event": "text_recognition_error", "engine": "paddle", "error": str(exc)})
            return {"text": "", "confidence": 0.0, "segments": []}

        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    @staticmethod
    def _extract_text_from_json(json_data: dict) -> str:
        """Extract text from PaddleOCRVL JSON.

        Primary:  parsing_res_list[].block_content  (PaddleOCRVL standard)
        Fallback: pages[].paragraphs[].text
        """
        if not isinstance(json_data, dict):
            return ""

        parts = [
            block["block_content"].strip()
            for block in json_data.get("parsing_res_list", [])
            if isinstance(block, dict) and block.get("block_content", "").strip()
        ]

        if not parts:
            for page in json_data.get("pages", []):
                for para in page.get("paragraphs", []) if isinstance(page, dict) else []:
                    if isinstance(para, dict) and para.get("text", "").strip():
                        parts.append(para["text"].strip())

        return " ".join(parts)
