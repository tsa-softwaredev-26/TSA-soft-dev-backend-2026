"""Audio processing utilities for speech recognition."""
from __future__ import annotations

import io

import numpy as np


_WEBM_MAGIC = b"\x1a\x45\xdf\xa3"
_OGG_MAGIC = b"OggS"
_WAV_MAGIC = b"RIFF"


def validate_audio_format(audio_bytes: bytes) -> tuple[bool, str]:
    if not audio_bytes or len(audio_bytes) < 4:
        return False, "empty"
    if audio_bytes[:4] == _WEBM_MAGIC:
        return True, "webm"
    if audio_bytes[:4] == _OGG_MAGIC:
        return True, "ogg"
    if len(audio_bytes) >= 12 and audio_bytes[4:8] == b"ftyp":
        return True, "mp4"
    if audio_bytes[:4] == _WAV_MAGIC:
        return False, "wav"
    return False, "unknown"


def load_audio_bytes(audio_bytes: bytes, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    is_valid, fmt = validate_audio_format(audio_bytes)
    if not is_valid:
        raise ValueError(f"unsupported audio format: {fmt}. use webm, ogg, or m4a/mp4 audio")

    try:
        import torch
        import torchaudio
    except Exception as exc:
        raise RuntimeError("audio decoding dependencies missing (torchaudio)") from exc

    try:
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes), format=fmt)
    except Exception as exc:
        raise ValueError(f"failed to decode {fmt} audio: {exc}") from exc

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)
        sample_rate = target_sr

    audio_array = waveform.squeeze().numpy().astype(np.float32)
    return audio_array, int(sample_rate)
