from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Optional

from PIL import Image

from visual_memory.utils import get_logger
from visual_memory.utils.device_utils import get_device

_log = get_logger(__name__)


class VLMPipeline:
    """Moondream wrapper with bounded inference latency."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _get_model(self):
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_id = "vikhyatk/moondream2"
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
            )
            self._model = self._model.to(get_device())
        return self._model, self._tokenizer

    def _infer(self, image_path: Path) -> str:
        model, tokenizer = self._get_model()
        with Image.open(image_path) as image_obj:
            image = image_obj.convert("RGB")

        if hasattr(model, "query"):
            response = model.query(image, "Describe this object in one short sentence.")
            return str((response or {}).get("answer", "")).strip()

        encoded = model.encode_image(image)
        answer = model.answer_question(
            encoded,
            "Describe this object in one short sentence.",
            tokenizer=tokenizer,
        )
        return str(answer or "").strip()

    def describe(self, image_path: str | Path, timeout: float) -> str:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"image not found: {path}")

        timeout = max(float(timeout), 0.1)
        future = self._executor.submit(self._infer, path)
        try:
            result = future.result(timeout=timeout)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise TimeoutError("vlm inference timed out") from exc
        except Exception:
            _log.exception({"event": "vlm_describe_failed", "image_path": str(path)})
            raise

        result = (result or "").strip()
        if not result:
            raise RuntimeError("empty VLM description")
        return result


_vlm_pipeline: Optional[VLMPipeline] = None


def get_vlm_pipeline() -> VLMPipeline:
    global _vlm_pipeline
    if _vlm_pipeline is None:
        _vlm_pipeline = VLMPipeline()
    return _vlm_pipeline
