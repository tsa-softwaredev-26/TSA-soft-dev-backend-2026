"""Abstract base class for speech recognition engines."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseSpeechRecognizer(ABC):
    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        context: str | None = None,
    ) -> dict:
        """Transcribe mono float32 audio in [-1, 1] to text."""

    def to_cpu(self) -> None:
        """Move model to CPU RAM."""
        return None

    def to_gpu(self) -> None:
        """Move model to GPU VRAM."""
        return None

