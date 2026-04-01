"""Audio processing utilities for speech recognition."""
from __future__ import annotations

import io
import time

import numpy as np


_WEBM_MAGIC = b"\x1a\x45\xdf\xa3"
_OGG_MAGIC = b"OggS"
_WAV_MAGIC = b"RIFF"

_MIN_AUDIO_SECONDS = 0.20
_SILENCE_RMS_THRESHOLD = 0.0025


class AudioError(Exception):
    code = "audio_error"
    user_message = "I could not process that audio. Please try again."

    def __init__(self, detail: str = ""):
        super().__init__(detail)
        self.detail = detail


class InvalidAudioFormatError(AudioError):
    code = "stt_invalid_format"
    user_message = "That audio format is not supported. Please use the app microphone."


class AudioTooShortError(AudioError):
    code = "stt_too_short"
    user_message = "I did not catch that. Hold the button a bit longer and try again."


class NearSilentAudioError(AudioError):
    code = "stt_near_silent"
    user_message = "I could not hear speech clearly. Please speak louder and try again."


class AudioDecodeError(AudioError):
    code = "stt_decode_failed"
    user_message = "I could not decode that audio. Please try again."


class RecognizerFailureError(AudioError):
    code = "stt_recognizer_failed"
    user_message = "I could not transcribe that right now. Please try again."


class RecognizerTimeoutError(AudioError):
    code = "stt_timeout"
    user_message = "Transcription timed out. Please try a shorter request."


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
        raise InvalidAudioFormatError(f"unsupported audio format: {fmt}. use webm, ogg, or m4a/mp4 audio")

    try:
        import torch
        import torchaudio
    except Exception as exc:
        raise RuntimeError("audio decoding dependencies missing (torchaudio)") from exc

    try:
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes), format=fmt)
    except Exception as exc:
        raise AudioDecodeError(f"failed to decode {fmt} audio: {exc}") from exc

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)
        sample_rate = target_sr

    audio_array = waveform.squeeze().numpy().astype(np.float32)
    duration_s = len(audio_array) / float(sample_rate) if sample_rate else 0.0
    if duration_s < _MIN_AUDIO_SECONDS:
        raise AudioTooShortError(f"audio too short: {duration_s:.3f}s")
    rms = float(np.sqrt(np.mean(np.square(audio_array)))) if len(audio_array) else 0.0
    if rms < _SILENCE_RMS_THRESHOLD:
        raise NearSilentAudioError(f"audio near silent: rms={rms:.6f}")
    return audio_array, int(sample_rate)
