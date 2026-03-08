"""Device selection utility for cross-platform GPU acceleration."""
import torch


def get_device() -> str:
    """Return best available compute device string: CUDA > MPS > CPU.

    Priority rationale: CUDA is preferred for server/GPU deployments.
    MPS is the fallback for Apple Silicon (macOS local dev).
    CPU is the final fallback for any platform.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
