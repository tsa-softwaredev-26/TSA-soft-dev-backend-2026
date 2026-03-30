"""Text recognition client using an external HTTP OCR service."""
from __future__ import annotations

import json
import os
import uuid
from io import BytesIO
from urllib import error as urlerror
from urllib import request as urlrequest

from PIL import Image

from visual_memory.config import Settings
from visual_memory.utils import get_logger
from .base import BaseTextRecognizer

_log = get_logger(__name__)
_settings = Settings()


class HTTPOCRRecognizer(BaseTextRecognizer):
    """OCR client that sends image crops to a local HTTP OCR microservice."""

    def __init__(self) -> None:
        self.url = _settings.ocr_service_url
        self.timeout_s = float(_settings.ocr_timeout_seconds)
        _log.info({
            "event": "ocr_service_init",
            "url": self.url,
            "timeout_s": self.timeout_s,
            "note": "OCR batch path removed after measured regressions; using per-image OCR only",
        })

    def recognize(self, image: Image.Image) -> dict:
        """Run OCR on a PIL image through HTTP. Returns {'text', 'confidence', 'segments'}."""
        try:
            payload, content_type = self._build_multipart_image_payload(image)
            req = urlrequest.Request(
                self.url,
                data=payload,
                method="POST",
                headers=self._request_headers(content_type),
            )
            with urlrequest.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw) if raw.strip() else {}
            text = str(data.get("text", "") or "")
            confidence = float(data.get("confidence", 0.0) or 0.0)
            ocr_time_ms = float(data.get("_ocr_time_ms", 0.0) or 0.0)
            out = {
                "text": text,
                "confidence": confidence,
                "segments": [(text, confidence)] if text else [],
            }
            _log.info({
                "event": "text_recognition",
                "engine": "http",
                "status": "success" if text else "no_text",
                "text_length": len(text),
                "confidence": round(confidence, 4),
                "ocr_time_ms": round(ocr_time_ms, 1),
            })
            return out
        except (urlerror.URLError, TimeoutError, ValueError, OSError, json.JSONDecodeError) as exc:
            _log.error({"event": "text_recognition_error", "engine": "http", "error": str(exc)})
            return {"text": "", "confidence": 0.0, "segments": []}

    def recognize_batch(self, images: list[Image.Image]) -> list[dict]:
        # Batch endpoint was removed after repeated p95 regressions in server A/B runs.
        # Keep batch API shape at client level for compatibility, but execute sequentially.
        return [self.recognize(img) for img in images]

    @staticmethod
    def _build_multipart_image_payload(image: Image.Image) -> tuple[bytes, str]:
        boundary = f"----visualmemory-ocr-{uuid.uuid4().hex}"
        img_buf = BytesIO()
        image.convert("RGB").save(img_buf, format="PNG")
        img_bytes = img_buf.getvalue()

        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="image"; filename="crop.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode("utf-8") + img_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
        return body, f"multipart/form-data; boundary={boundary}"

    @staticmethod
    def _request_headers(content_type: str) -> dict[str, str]:
        headers = {"Content-Type": content_type, "Accept": "application/json"}
        api_key = os.environ.get("API_KEY", "").strip()
        if api_key:
            headers["X-API-Key"] = api_key
        return headers
