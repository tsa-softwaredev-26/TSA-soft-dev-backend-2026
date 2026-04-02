"""Lazy singleton accessors for pipeline instances.

Models are multi-GB; loaded once and reused across requests.
"""
import os
from visual_memory.config import Settings
from visual_memory.engine.model_registry import registry
from visual_memory.utils.logger import get_logger

_settings = Settings()
_log = get_logger(__name__)

_database = None
_remember_pipeline = None
_scan_pipeline = None
_feedback_store = None
_user_settings = None


def get_database():
    """Shared DatabaseStore instance - single connection for API-layer reads/writes."""
    global _database
    if _database is None:
        from visual_memory.database.store import DatabaseStore
        _database = DatabaseStore(_settings.db_path)
    return _database


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


def get_loaded_scan_pipeline():
    """Return the scan pipeline only if it is already initialized."""
    return _scan_pipeline


def reload_scan_database_if_loaded() -> bool:
    """Reload scan DB cache when pipeline is warm; never forces model load."""
    pipeline = get_loaded_scan_pipeline()
    if pipeline is None:
        return False
    pipeline.reload_database()
    return True


def reload_scan_head_if_loaded() -> bool:
    """Reload projection head when scan pipeline is warm; never forces model load."""
    pipeline = get_loaded_scan_pipeline()
    if pipeline is None:
        return False
    pipeline.reload_head()
    return True


def apply_scan_learning_if_loaded(enabled: bool) -> bool:
    """Apply learning toggle to a warm scan pipeline without forcing initialization."""
    pipeline = get_loaded_scan_pipeline()
    if pipeline is None:
        return False
    pipeline.set_enable_learning(enabled)
    return True


def apply_scan_head_weight_if_loaded(weight: float, ramp_at: int, ramp_power: float) -> bool:
    """Apply head blend params to warm scan pipeline without forcing initialization."""
    pipeline = get_loaded_scan_pipeline()
    if pipeline is None:
        return False
    pipeline.set_head_weight(weight, ramp_at, ramp_power)
    return True


def get_feedback_store():
    global _feedback_store
    if _feedback_store is None:
        from visual_memory.learning.feedback_store import FeedbackStore
        _feedback_store = FeedbackStore(get_database())
    return _feedback_store


def get_settings():
    """Return the module-level Settings instance for read/write by API routes."""
    return _settings


def get_user_settings():
    """Return the UserSettings singleton, loading from DB on first access."""
    global _user_settings
    if _user_settings is None:
        from visual_memory.config.user_settings import UserSettings
        _user_settings = UserSettings.load(get_database())
    return _user_settings


def warm_all():
    """Load all pipeline singletons at startup to avoid first-request latency."""
    get_remember_pipeline()
    get_scan_pipeline()
    get_feedback_store()
    _apply_persisted_ml_settings(load_scan_pipeline=True)
    # When SAVE_VRAM is on, settle into scan-mode VRAM layout after loading everything.
    # This offloads GDino (~900 MB) immediately, leaving scan models warm on GPU.
    if _settings.save_vram:
        registry.prepare_for_scan()


def warm_voice_only():
    """Warm startup for low-VRAM hosts: keep only voice path warm at boot."""
    get_database()
    get_feedback_store()
    _apply_persisted_ml_settings(load_scan_pipeline=False)
    registry.get_whisper_recognizer()
    registry.prepare_for_voice()


def warm_startup():
    """Startup warm strategy controlled by STARTUP_WARM_MODE.

    Supported values:
    - full: eager remember+scan warmup (previous behavior)
    - voice: warm only DB/settings/Whisper (default)
    - none: no model warmup at startup
    """
    mode = os.environ.get("STARTUP_WARM_MODE", "voice").strip().lower()
    if mode == "none":
        get_database()
        get_feedback_store()
        _apply_persisted_ml_settings(load_scan_pipeline=False)
        _log.info({"event": "startup_warm_mode", "mode": "none"})
        return
    if mode == "full":
        _log.info({"event": "startup_warm_mode", "mode": "full"})
        warm_all()
        return

    _log.info({"event": "startup_warm_mode", "mode": "voice"})
    warm_voice_only()


def _apply_persisted_ml_settings(*, load_scan_pipeline: bool):
    """Restore ML settings saved via PATCH /settings from previous runs."""
    saved = get_database().load_ml_settings()
    if not saved:
        return
    s = get_settings()
    pipeline = get_scan_pipeline() if load_scan_pipeline else None
    _PATCHABLE_TYPES = {
        "enable_learning": bool,
        "min_feedback_for_training": int,
        "projection_head_weight": float,
        "projection_head_ramp_at": int,
        "projection_head_ramp_power": float,
        "projection_head_epochs": int,
        "triplet_margin": float,
        "triplet_positive_weight": float,
        "triplet_negative_weight": float,
        "triplet_hard_negative_boost": float,
        "similarity_threshold": float,
        "similarity_threshold_baseline": float,
        "similarity_threshold_personalized": float,
        "similarity_threshold_document": float,
        "scan_similarity_margin": float,
        "scan_similarity_margin_document": float,
        "remember_max_prototypes_per_label": int,
    }
    for key, cast in _PATCHABLE_TYPES.items():
        if key in saved:
            try:
                setattr(s, key, cast(saved[key]))
            except (TypeError, ValueError):
                pass
    if pipeline is not None:
        pipeline.set_enable_learning(s.enable_learning)
        pipeline.set_head_weight(
            s.projection_head_weight,
            s.projection_head_ramp_at,
            s.projection_head_ramp_power,
        )
