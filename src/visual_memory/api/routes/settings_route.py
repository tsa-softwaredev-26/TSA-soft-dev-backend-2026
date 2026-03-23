from flask import Blueprint, request, jsonify

from visual_memory.api.pipelines import get_feedback_store, get_scan_pipeline, get_settings

settings_bp = Blueprint("settings", __name__)

# Fields the frontend is allowed to update at runtime via PATCH /settings.
# Persisted in-memory only; see ARCHITECTURE.md TODO for DB persistence.
_PATCHABLE = {
    "enable_learning": bool,
    "min_feedback_for_training": int,
    "projection_head_weight": float,
    "projection_head_ramp_at": int,
    "projection_head_epochs": int,
}


@settings_bp.get("/settings")
def get_settings_route():
    """Return current runtime-tunable settings and live pipeline state.

    Response:
        {
            "enable_learning": bool,
            "min_feedback_for_training": int,
            "projection_head_weight": float,
            "projection_head_ramp_at": int,
            "projection_head_epochs": int,
            "head_trained": bool,
            "triplet_count": int,
            "feedback_counts": {"positives": int, "negatives": int, "triplets": int}
        }
    """
    s = get_settings()
    pipeline = get_scan_pipeline()
    counts = get_feedback_store().count()

    return jsonify({
        "enable_learning": pipeline._enable_learning,
        "min_feedback_for_training": s.min_feedback_for_training,
        "projection_head_weight": pipeline._head_weight,
        "projection_head_ramp_at": pipeline._head_ramp_at,
        "projection_head_epochs": s.projection_head_epochs,
        "head_trained": pipeline._head_trained,
        "triplet_count": pipeline._triplet_count,
        "feedback_counts": counts,
    })


@settings_bp.patch("/settings")
def patch_settings():
    """Update one or more runtime-tunable settings.

    Accepted fields (all optional):
        enable_learning         bool   - toggle projection head application during scan
        min_feedback_for_training int  - minimum triplets before /retrain proceeds
        projection_head_weight  float  - max blend fraction (0.0-1.0)
        projection_head_ramp_at int    - triplet count at which max weight is reached
        projection_head_epochs  int    - training epochs per /retrain call

    Unrecognised fields are ignored. Type errors return 400.

    Response: full settings state (same schema as GET /settings).
    """
    data = request.get_json(silent=True) or {}
    errors = {}
    applied = {}

    for key, expected_type in _PATCHABLE.items():
        if key not in data:
            continue
        raw = data[key]
        try:
            value = expected_type(raw)
        except (TypeError, ValueError):
            errors[key] = f"expected {expected_type.__name__}, got {type(raw).__name__}"
            continue

        if key == "projection_head_weight" and not (0.0 <= value <= 1.0):
            errors[key] = "must be between 0.0 and 1.0"
            continue
        if key in ("min_feedback_for_training", "projection_head_ramp_at", "projection_head_epochs") and value < 1:
            errors[key] = "must be >= 1"
            continue

        applied[key] = value

    if errors:
        return jsonify({"errors": errors}), 400

    # write valid values to Settings singleton, then propagate to pipeline in one pass
    s = get_settings()
    for key, value in applied.items():
        setattr(s, key, value)

    pipeline = get_scan_pipeline()
    if "enable_learning" in applied:
        pipeline.set_enable_learning(s.enable_learning)
    if "projection_head_weight" in applied or "projection_head_ramp_at" in applied:
        pipeline.set_head_weight(s.projection_head_weight, s.projection_head_ramp_at)

    # return full state after applying changes
    return get_settings_route()
