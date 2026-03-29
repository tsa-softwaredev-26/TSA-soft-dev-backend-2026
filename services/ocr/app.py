"""HTTP OCR microservice backed by PaddleOCR."""

from __future__ import annotations

import io
import os
from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile
import numpy as np
from PIL import Image


def _bool_env(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


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


def _extract_text(image: Image.Image) -> dict:
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

    return {
        "text": " ".join(texts).strip(),
        "confidence": (sum(confidences) / len(confidences)) if confidences else 0.0,
        "segments": segments,
    }


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

    try:
        pil_image = Image.open(io.BytesIO(data))
    except OSError as exc:
        raise HTTPException(status_code=400, detail="invalid image") from exc

    try:
        return _extract_text(pil_image)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ocr/batch")
async def run_ocr_batch(images: list[UploadFile] = File(...)) -> dict:
    if not images:
        raise HTTPException(status_code=400, detail="missing images")

    out = []
    for image in images:
        data = await image.read()
        if not data:
            out.append({"text": "", "confidence": 0.0, "segments": []})
            continue
        try:
            pil_image = Image.open(io.BytesIO(data))
        except OSError:
            out.append({"text": "", "confidence": 0.0, "segments": []})
            continue
        try:
            out.append(_extract_text(pil_image))
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"results": out}
