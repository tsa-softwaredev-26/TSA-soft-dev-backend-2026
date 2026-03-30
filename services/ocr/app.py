"""HTTP OCR microservice backed by PaddleOCR."""

from __future__ import annotations

import io
import os
import time
from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile
import numpy as np
from PIL import Image


def _bool_env(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _decode_image(data: bytes) -> Image.Image | None:
    if not data:
        return None
    try:
        return Image.open(io.BytesIO(data))
    except OSError:
        return None

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
    return PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)


def _extract_text(image: Image.Image) -> tuple[dict, float]:
    """Extract text from image. Returns (result_dict, elapsed_ms)."""
    start_time = time.time()
    engine = _get_ocr_engine()
    result = engine.ocr(np.array(image.convert("RGB")), cls=False)
    lines = result[0] if result else []
    min_conf = float(os.environ.get("OCR_MIN_CONFIDENCE", "0.3"))

    segments: list[dict[str, float | str]] = []
    texts: list[str] = []
    confidences: list[float] = []

    for line in lines or []:
        if not line or len(line) < 2:
            continue
        text, confidence = line[1]
        if not text:
            continue
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


app = FastAPI(title="Spaitra OCR Service", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ocr")
async def run_ocr(image: UploadFile = File(...)) -> dict:
    if not image.filename:
        raise HTTPException(status_code=400, detail="missing image")

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
        result, ocr_time_ms = _extract_text(pil_image)
        total_ms = decode_ms + ocr_time_ms
        result["_ocr_time_ms"] = ocr_time_ms
        result["_decode_ms"] = decode_ms
        result["_total_ms"] = total_ms
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
