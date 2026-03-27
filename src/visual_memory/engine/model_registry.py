from __future__ import annotations
from visual_memory.config import Settings

_settings = Settings()


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
