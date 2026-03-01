"""Text recognition engine using DeepSeek-OCR (vision-language model)."""
from PIL import Image

from visual_memory.utils import get_logger
from .base import BaseTextRecognizer

_log = get_logger(__name__)


class DeepSeekRecognizer(BaseTextRecognizer):
    def __init__(self) -> None:
        try:
            import torch
            import transformers
            # DeepSeek-OCR requires transformers 4.46.x — removed API in 4.47+
            ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
            if ver >= (4, 47):
                raise ImportError(
                    f"DeepSeek-OCR requires transformers==4.46.3, "
                    f"but {transformers.__version__} is installed. "
                    f"Use the pinned venv: pip install -r requirements-deepseek.txt"
                )
        except ImportError as e:
            raise ImportError(
                "DeepSeek-OCR requires 'transformers==4.46.3' and 'torch'. "
                "Install via: pip install -r requirements-deepseek.txt"
            ) from e

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        dtype = torch.float16 if self.device in ("mps", "cuda") else torch.float32

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-OCR",
            torch_dtype=dtype,
            device_map={"": self.device},
            trust_remote_code=True,
        )
        self.processor = transformers.AutoProcessor.from_pretrained(
            "deepseek-ai/DeepSeek-OCR",
            trust_remote_code=True,
        )

    def recognize(self, image: Image.Image) -> dict:
        import torch

        prompt = "<image>\nFree OCR."
        inputs = self.processor(
            images=image.convert("RGB"),
            text=prompt,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512)

        text = self.processor.decode(outputs[0], skip_special_tokens=True).strip()

        result = {
            "text": text,
            "confidence": 1.0,
            "segments": [(text, 1.0)] if text else [],
        }

        _log.info({
            "event": "text_recognition",
            "engine": "deepseek",
            "text_length": len(text),
            "segment_count": len(result["segments"]),
            "avg_confidence": 1.0,
        })
        return result
