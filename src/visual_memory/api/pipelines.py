"""Lazy singleton accessors for pipeline instances.

Models are multi-GB; loaded once at process start and reused across requests.
"""
from pathlib import Path

from visual_memory.config import Settings

_settings = Settings()

_remember_pipeline = None
_scan_pipeline = None
_feedback_store = None


def get_remember_pipeline():
    global _remember_pipeline
    if _remember_pipeline is None:
        from visual_memory.pipelines.remember_mode.pipeline import RememberPipeline
        _remember_pipeline = RememberPipeline()
    return _remember_pipeline


def get_scan_pipeline():
    global _scan_pipeline
    if _scan_pipeline is None:
        from visual_memory.pipelines.scan_mode.pipeline import ScanPipeline
        _scan_pipeline = ScanPipeline(focal_length_px=0)
    return _scan_pipeline


def get_feedback_store():
    global _feedback_store
    if _feedback_store is None:
        from visual_memory.learning.feedback_store import FeedbackStore
        _feedback_store = FeedbackStore(Path("feedback"))
    return _feedback_store


def get_settings():
    """Return the module-level Settings instance for read/write by API routes."""
    return _settings


def warm_all():
    """Load all pipeline singletons at startup to avoid first-request latency."""
    get_remember_pipeline()
    get_scan_pipeline()
    get_feedback_store()
