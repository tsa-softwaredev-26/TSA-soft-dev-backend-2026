"""Lazy singleton accessors for pipeline instances.

Models are multi-GB; loaded once at process start and reused across requests.
"""
from visual_memory.config import Settings

_settings = Settings()

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
    _apply_persisted_ml_settings()
    # When SAVE_VRAM is on, settle into scan-mode VRAM layout after loading everything.
    # This offloads GDino (~900 MB) immediately, leaving scan models warm on GPU.
    if _settings.save_vram:
        from visual_memory.engine.model_registry import registry
        registry.prepare_for_scan()


def _apply_persisted_ml_settings():
    """Restore ML settings saved via PATCH /settings from previous runs."""
    saved = get_database().load_ml_settings()
    if not saved:
        return
    s = get_settings()
    pipeline = get_scan_pipeline()
    _PATCHABLE_TYPES = {
        "enable_learning": bool,
        "min_feedback_for_training": int,
        "projection_head_weight": float,
        "projection_head_ramp_at": int,
        "projection_head_epochs": int,
    }
    for key, cast in _PATCHABLE_TYPES.items():
        if key in saved:
            try:
                setattr(s, key, cast(saved[key]))
            except (TypeError, ValueError):
                pass
    pipeline.set_enable_learning(s.enable_learning)
    pipeline.set_head_weight(s.projection_head_weight, s.projection_head_ramp_at)
