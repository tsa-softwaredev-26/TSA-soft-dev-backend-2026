"""Text recognition client using an external HTTP OCR service."""
from __future__ import annotations

import json
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
        self.batch_url = self.url.rstrip("/") + "/batch"
        self.timeout_s = float(_settings.ocr_timeout_seconds)
        _log.info({
            "event": "ocr_service_init",
            "url": self.url,
            "batch_url": self.batch_url,
            "timeout_s": self.timeout_s,
        })

    def recognize(self, image: Image.Image) -> dict:
        """Run OCR on a PIL image through HTTP. Returns {'text', 'confidence', 'segments'}."""
        try:
            payload, content_type = self._build_multipart_image_payload(image)
            req = urlrequest.Request(
                self.url,
                data=payload,
                method="POST",
                headers={"Content-Type": content_type, "Accept": "application/json"},
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
        if not images:
            return []
        try:
            payload, content_type = self._build_multipart_batch_payload(images)
            req = urlrequest.Request(
                self.batch_url,
                data=payload,
                method="POST",
                headers={"Content-Type": content_type, "Accept": "application/json"},
            )
            with urlrequest.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw) if raw.strip() else {}
            raw_results = data.get("results") if isinstance(data, dict) else None
            if not isinstance(raw_results, list):
                raise ValueError("invalid batch OCR response payload")
            out: list[dict] = []
            for item in raw_results:
                if not isinstance(item, dict):
                    out.append({"text": "", "confidence": 0.0, "segments": []})
                    continue
                text = str(item.get("text", "") or "")
                confidence = float(item.get("confidence", 0.0) or 0.0)
                segments = item.get("segments")
                if not isinstance(segments, list):
                    segments = [(text, confidence)] if text else []
                out.append({
                    "text": text,
                    "confidence": confidence,
                    "segments": segments,
                })
            if len(out) < len(images):
                out.extend([{"text": "", "confidence": 0.0, "segments": []}] * (len(images) - len(out)))

            batch_total_ms = float(data.get("_batch_total_ms", 0.0) or 0.0)
            per_item_avg_ms = float(data.get("_per_item_avg_ms", 0.0) or 0.0)
            batch_size = int(data.get("_batch_size", len(images)) or len(images))

            _log.info({
                "event": "text_recognition_batch",
                "engine": "http",
                "count": len(images),
                "non_empty": sum(1 for r in out if r.get("text")),
                "batch_total_ms": round(batch_total_ms, 1),
                "per_item_avg_ms": round(per_item_avg_ms, 1),
            })
            return out[: len(images)]
        except (urlerror.URLError, TimeoutError, ValueError, OSError, json.JSONDecodeError) as exc:
            _log.error({
                "event": "text_recognition_batch_error",
                "engine": "http",
                "count": len(images),
                "error": str(exc),
            })
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
    def _build_multipart_batch_payload(images: list[Image.Image]) -> tuple[bytes, str]:
        boundary = f"----visualmemory-ocr-batch-{uuid.uuid4().hex}"
        chunks: list[bytes] = []
        for i, image in enumerate(images):
            img_buf = BytesIO()
            image.convert("RGB").save(img_buf, format="PNG")
            img_bytes = img_buf.getvalue()
            header = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="images"; filename="crop_{i}.png"\r\n'
                "Content-Type: image/png\r\n\r\n"
            ).encode("utf-8")
            chunks.append(header + img_bytes + b"\r\n")
        chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
        return b"".join(chunks), f"multipart/form-data; boundary={boundary}"
