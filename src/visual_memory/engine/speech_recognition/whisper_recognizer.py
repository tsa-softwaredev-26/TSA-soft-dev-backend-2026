"""Whisper-based speech recognition using HuggingFace transformers."""
from __future__ import annotations

import time

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from visual_memory.config import Settings
from visual_memory.utils import get_device, get_logger
from visual_memory.utils.logger import LogTag

from .base import BaseSpeechRecognizer

_settings = Settings()
_log = get_logger(__name__)


class WhisperRecognizer(BaseSpeechRecognizer):
    def __init__(self, model_id: str | None = None, device: str | None = None):
        self.model_id = model_id or _settings.whisper_model
        self.device = device or get_device()
        self.sample_rate = _settings.whisper_sample_rate

        _log.info({
            "event": "whisper_init_start",
            "tag": LogTag.API,
            "model": self.model_id,
            "device": self.device,
        })
        t0 = time.monotonic()
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model.generation_config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=_settings.whisper_language,
            task="transcribe",
        )
        _log.info({
            "event": "whisper_init_complete",
            "tag": LogTag.API,
            "model": self.model_id,
            "device": self.device,
            "load_time_ms": round((time.monotonic() - t0) * 1000),
        })

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        context: str | None = None,
    ) -> dict:
        t0 = time.monotonic()
        if sample_rate != self.sample_rate:
            raise ValueError(f"expected sample_rate={self.sample_rate}, got {sample_rate}")
        if audio.ndim != 1:
            raise ValueError("audio must be mono")

        inputs = self.processor(audio, sampling_rate=self.sample_rate, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)

        generation_kwargs: dict = {}
        if context and _settings.whisper_context_enabled:
            prompt = context.strip()[:Settings().whisper_context_max_chars]
            if prompt:
                generation_kwargs["prompt_ids"] = self.processor.get_prompt_ids(prompt)

        with torch.no_grad():
            predicted_ids = self.model.generate(input_features, **generation_kwargs)

        text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        duration_ms = round((time.monotonic() - t0) * 1000)
        _log.info({
            "event": "whisper_transcribe",
            "tag": LogTag.API,
            "text_length": len(text),
            "audio_duration_s": round(len(audio) / self.sample_rate, 2),
            "transcribe_ms": duration_ms,
            "context_used": bool(context and _settings.whisper_context_enabled),
        })
        return {
            "text": text,
            "language": _settings.whisper_language,
            "confidence": 1.0,
        }

    def build_context_prompt(self, db_store) -> str:
        cfg = Settings()
        labels: list[str] = []
        rooms: list[str] = []
        try:
            labels = db_store.get_known_item_labels(limit=cfg.whisper_context_max_labels)
            rooms = db_store.get_recent_room_names(limit=cfg.whisper_context_max_rooms)
        except Exception as exc:
            _log.warning({
                "event": "whisper_context_db_error",
                "tag": LogTag.API,
                "error": str(exc),
            })

        common_query_terms = [
            "where", "find", "left", "right", "ahead", "behind",
            "put", "leave", "last", "seen", "my",
        ]

        def _dedupe(seq: list[str]) -> list[str]:
            out = []
            seen = set()
            for entry in seq:
                norm = entry.lower()
                if norm in seen or not norm:
                    continue
                seen.add(norm)
                out.append(entry)
            return out

        context_terms = _dedupe(labels) + _dedupe(rooms) + common_query_terms
        prompt = " ".join(context_terms)[:cfg.whisper_context_max_chars]
        _log.info({
            "event": "whisper_context_built",
            "tag": LogTag.API,
            "label_count": len(_dedupe(labels)),
            "room_count": len(_dedupe(rooms)),
            "term_count": len(context_terms),
            "prompt_chars": len(prompt),
        })
        return prompt

    def to_cpu(self) -> None:
        if self.device == "cpu":
            return
        self.model.to("cpu")
        self.device = "cpu"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def to_gpu(self) -> None:
        target = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        if self.device == target:
            return
        self.model.to(target)
        self.device = target
