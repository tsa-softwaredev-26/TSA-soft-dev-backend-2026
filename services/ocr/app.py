"""HTTP OCR microservice backed by PaddleOCR."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
import io
import logging
import os
import time
from functools import lru_cache

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image

_API_KEY = os.environ.get("API_KEY", "").strip()
_OCR_MAX_CONCURRENCY = max(1, int(os.environ.get("OCR_MAX_CONCURRENCY", "1")))
_OCR_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("OCR_REQUEST_TIMEOUT_SECONDS", "40"))
_OCR_THROTTLE_RETRY_AFTER_SECONDS = max(1, int(os.environ.get("OCR_THROTTLE_RETRY_AFTER_SECONDS", "2")))
_OCR_RATE_LIMIT_PER_MIN = max(0, int(os.environ.get("OCR_RATE_LIMIT_PER_MIN", "120")))
_OCR_SEMAPHORE = asyncio.Semaphore(_OCR_MAX_CONCURRENCY)
_OCR_RATE_LOCK = asyncio.Lock()
_OCR_RATE_WINDOW_SECONDS = 60.0
_OCR_RATE_BUCKETS: dict[str, deque[float]] = defaultdict(deque)
_LOG = logging.getLogger("spaitra.ocr")


def _bool_env(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _decode_image(data: bytes) -> Image.Image | None:
    if not data:
        return None
    try:
        return Image.open(io.BytesIO(data))
    except OSError:
        return None


def _resize_for_ocr(image: Image.Image) -> Image.Image:
    max_side = int(os.environ.get("OCR_MAX_SIDE", "2000"))
    width, height = image.size
    largest_side = max(width, height)
    if largest_side <= max_side:
        return image
    scale = max_side / float(largest_side)
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(new_size, Image.Resampling.LANCZOS)


@lru_cache(maxsize=1)
def _get_ocr_engine():
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR dependencies are not installed. "
            "Install the OCR environment with `pip install -e .[ocr]`."
        ) from exc

    lang = os.environ.get("OCR_LANG", "en")
    use_angle_cls = _bool_env("OCR_USE_ANGLE_CLS")
    enable_mkldnn = _bool_env("OCR_ENABLE_MKLDNN", "false")
    return PaddleOCR(
        lang=lang,
        use_doc_orientation_classify=use_angle_cls,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=enable_mkldnn,
    )


def _extract_text(image: Image.Image) -> tuple[dict, float]:
    """Extract text from image. Returns (result_dict, elapsed_ms)."""
    start_time = time.time()
    engine = _get_ocr_engine()
    rgb = np.array(image.convert("RGB"))
    try:
        result = engine.ocr(rgb, cls=False)
    except TypeError:
        # PaddleOCR v3 removed the cls kwarg on predict/ocr path.
        result = engine.ocr(rgb)
    lines = result[0] if result else []
    min_conf = float(os.environ.get("OCR_MIN_CONFIDENCE", "0.3"))

    segments: list[dict[str, float | str]] = []
    texts: list[str] = []
    confidences: list[float] = []

    for line in lines or []:
        text: str | None = None
        confidence: float | None = None

        if isinstance(line, dict):
            text = line.get("text")
            confidence = line.get("score", line.get("confidence"))
        elif isinstance(line, (list, tuple)):
            if len(line) >= 2 and isinstance(line[1], (list, tuple)):
                pair = line[1]
                if len(pair) >= 2:
                    text = pair[0]
                    confidence = pair[1]
            elif len(line) >= 1:
                text = line[0]
                if len(line) >= 2:
                    confidence = line[1]
        else:
            text = str(line)

        if text is None:
            continue
        if confidence is None:
            confidence = 1.0
        confidence = float(confidence)
        if confidence < min_conf:
            continue
        text = str(text).strip()
        if not text:
            continue
        segments.append({"text": text, "confidence": confidence})
        texts.append(text)
        confidences.append(confidence)

    elapsed_ms = (time.time() - start_time) * 1000
    result_dict = {
        "text": " ".join(texts).strip(),
        "confidence": (sum(confidences) / len(confidences)) if confidences else 0.0,
        "segments": segments,
    }
    return result_dict, elapsed_ms


def _run_ocr_sync(image: Image.Image) -> dict:
    resized_image = _resize_for_ocr(image)
    result, ocr_time_ms = _extract_text(resized_image)
    result["_ocr_time_ms"] = ocr_time_ms
    return result


app = FastAPI(title="Spaitra OCR Service", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ocr")
async def run_ocr(
    request: Request,
    image: UploadFile = File(...),
    api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict:
    if _API_KEY and api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")

    if not image.filename:
        raise HTTPException(status_code=400, detail="missing image")

    client_ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    if not client_ip and request.client is not None:
        client_ip = request.client.host or "unknown"
    if not client_ip:
        client_ip = "unknown"

    if _OCR_RATE_LIMIT_PER_MIN > 0:
        now = time.monotonic()
        async with _OCR_RATE_LOCK:
            bucket = _OCR_RATE_BUCKETS[client_ip]
            cutoff = now - _OCR_RATE_WINDOW_SECONDS
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= _OCR_RATE_LIMIT_PER_MIN:
                retry_after = max(1, int(bucket[0] + _OCR_RATE_WINDOW_SECONDS - now))
                _LOG.warning(
                    "ocr_rate_limited client_ip=%s limit_per_min=%s retry_after_s=%s",
                    client_ip,
                    _OCR_RATE_LIMIT_PER_MIN,
                    retry_after,
                )
                return JSONResponse(
                    status_code=429,
                    headers={"Retry-After": str(retry_after)},
                    content={
                        "error": "rate_limited",
                        "message": "OCR request rate is limited. Retry later.",
                        "retry_after_seconds": retry_after,
                        "limit_per_minute": _OCR_RATE_LIMIT_PER_MIN,
                    },
                )
            bucket.append(now)

    acquired = False
    try:
        try:
            await asyncio.wait_for(_OCR_SEMAPHORE.acquire(), timeout=0.001)
            acquired = True
        except TimeoutError:
            _LOG.warning(
                "ocr_throttled capacity=%s retry_after_s=%s",
                _OCR_MAX_CONCURRENCY,
                _OCR_THROTTLE_RETRY_AFTER_SECONDS,
            )
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(_OCR_THROTTLE_RETRY_AFTER_SECONDS)},
                content={
                    "error": "server_busy",
                    "message": "OCR capacity is full. Retry shortly.",
                    "retry_after_seconds": _OCR_THROTTLE_RETRY_AFTER_SECONDS,
                    "capacity": _OCR_MAX_CONCURRENCY,
                },
            )

        data = await image.read()
        if not data:
            raise HTTPException(status_code=400, detail="empty image")

        decode_t0 = time.time()
        try:
            pil_image = _decode_image(data)
        except OSError as exc:
            raise HTTPException(status_code=400, detail="invalid image") from exc
        if pil_image is None:
            raise HTTPException(status_code=400, detail="invalid image")
        decode_ms = (time.time() - decode_t0) * 1000

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(_run_ocr_sync, pil_image),
                timeout=_OCR_REQUEST_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            _LOG.warning(
                "ocr_timeout timeout_s=%s capacity=%s",
                _OCR_REQUEST_TIMEOUT_SECONDS,
                _OCR_MAX_CONCURRENCY,
            )
            return JSONResponse(
                status_code=504,
                content={
                    "error": "ocr_timeout",
                    "message": "OCR request timed out under current load.",
                    "timeout_seconds": _OCR_REQUEST_TIMEOUT_SECONDS,
                },
            )
        total_ms = decode_ms + float(result.get("_ocr_time_ms", 0.0))
        result["_decode_ms"] = decode_ms
        result["_total_ms"] = total_ms
        return result
    except (RuntimeError, NotImplementedError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if acquired:
            _OCR_SEMAPHORE.release()
