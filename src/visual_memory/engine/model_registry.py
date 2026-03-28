from __future__ import annotations
import time
from visual_memory.config import Settings
from visual_memory.utils.logger import get_logger, LogTag
from visual_memory.utils.metrics import collect_system_metrics

_settings = Settings()
_log = get_logger(__name__)


class ModelRegistry:
    # Lazy-loading store for all shared ML models.
    # One module-level instance (registry) is shared across all pipelines in the process.
    # Flask startup: call registry.preload() to avoid first-request latency.

    def __init__(self):
        self._img_embedder    = None
        self._text_embedder   = None
        self._depth_estimator = None
        self._gdino_detector  = None
        self._yoloe_detector  = None

    @property
    def img_embedder(self):
        if self._img_embedder is None:
            from visual_memory.engine.embedding import ImageEmbedder
            self._img_embedder = ImageEmbedder()
        return self._img_embedder

    @property
    def text_embedder(self):
        if self._text_embedder is None:
            from visual_memory.engine.embedding import CLIPTextEmbedder
            self._text_embedder = CLIPTextEmbedder()
        return self._text_embedder

    @property
    def depth_estimator(self):
        if self._depth_estimator is None:
            from visual_memory.engine.depth import DepthEstimator
            self._depth_estimator = DepthEstimator()
        return self._depth_estimator

    @property
    def gdino_detector(self):
        if self._gdino_detector is None:
            from visual_memory.engine.object_detection import GroundingDinoDetector
            self._gdino_detector = GroundingDinoDetector()
        return self._gdino_detector

    @property
    def yoloe_detector(self):
        if self._yoloe_detector is None:
            from visual_memory.engine.object_detection import YoloeDetector
            self._yoloe_detector = YoloeDetector()
        return self._yoloe_detector

    def prepare_for_remember(self) -> None:
        """Offload scan-only models to CPU; ensure remember models are on GPU.

        No-op when SAVE_VRAM=0 or CUDA is unavailable.
        Called at the start of every RememberPipeline.run() / detect_score().
        """
        import torch
        if not _settings.save_vram or not torch.cuda.is_available():
            return
        t0 = time.monotonic()
        offloaded = []
        if self._yoloe_detector is not None:
            self._yoloe_detector.to_cpu()
            offloaded.append("yoloe")
        if self._depth_estimator is not None:
            self._depth_estimator.to_cpu()
            offloaded.append("depth")
        torch.cuda.empty_cache()
        if self._gdino_detector is not None:
            self._gdino_detector.to_gpu()
        _log.info({
            "event": "vram_layout",
            "tag": LogTag.VRAM,
            "mode": "remember",
            "offloaded": offloaded,
            "duration_ms": round((time.monotonic() - t0) * 1000),
            **collect_system_metrics(),
        })

    def prepare_for_scan(self) -> None:
        """Offload remember-only models to CPU; ensure scan models are on GPU.

        No-op when SAVE_VRAM=0 or CUDA is unavailable.
        Called at the start of every ScanPipeline.run() and after warm_all().
        """
        import torch
        if not _settings.save_vram or not torch.cuda.is_available():
            return
        t0 = time.monotonic()
        offloaded = []
        if self._gdino_detector is not None:
            self._gdino_detector.to_cpu()
            offloaded.append("gdino")
        torch.cuda.empty_cache()
        if self._yoloe_detector is not None:
            self._yoloe_detector.to_gpu()
        if self._depth_estimator is not None and _settings.enable_depth:
            self._depth_estimator.to_gpu()
        _log.info({
            "event": "vram_layout",
            "tag": LogTag.VRAM,
            "mode": "scan",
            "offloaded": offloaded,
            "duration_ms": round((time.monotonic() - t0) * 1000),
            **collect_system_metrics(),
        })

    def preload(self, depth: bool = False) -> None:
        # Eagerly load all models. Call at Flask startup to avoid first-request latency.
        _ = self.img_embedder
        if _settings.enable_ocr:
            _ = self.text_embedder
        _ = self.gdino_detector
        _ = self.yoloe_detector
        if depth:
            _ = self.depth_estimator


registry = ModelRegistry()
