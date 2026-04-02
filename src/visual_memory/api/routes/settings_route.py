from flask import Blueprint, request, jsonify

from visual_memory.api.pipelines import (
    apply_scan_head_weight_if_loaded,
    apply_scan_learning_if_loaded,
    get_database,
    get_feedback_store,
    get_scan_pipeline,
    get_settings,
)
from ._json_utils import coerce_json_value, read_json_dict

settings_bp = Blueprint("settings", __name__)

# Fields the frontend is allowed to update at runtime via PATCH /settings.
# Persisted to DB (user_state.ml_settings) and restored on startup via warm_all().
_PATCHABLE = {
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


@settings_bp.get("/settings")
def get_settings_route():
    """Return current runtime-tunable settings and live pipeline state.

    Response:
        {
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
        "projection_head_ramp_power": pipeline._head_ramp_power,
        "projection_head_epochs": s.projection_head_epochs,
        "triplet_margin": s.triplet_margin,
        "triplet_positive_weight": s.triplet_positive_weight,
        "triplet_negative_weight": s.triplet_negative_weight,
        "triplet_hard_negative_boost": s.triplet_hard_negative_boost,
        "similarity_threshold": s.similarity_threshold,
        "similarity_threshold_baseline": s.similarity_threshold_baseline,
        "similarity_threshold_personalized": s.similarity_threshold_personalized,
        "similarity_threshold_document": s.similarity_threshold_document,
        "scan_similarity_margin": s.scan_similarity_margin,
        "scan_similarity_margin_document": s.scan_similarity_margin_document,
        "remember_max_prototypes_per_label": s.remember_max_prototypes_per_label,
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
        projection_head_ramp_power float - ramp curve power (>0)
        projection_head_epochs  int    - training epochs per /retrain call
        triplet_margin          float  - margin term for triplet loss (>=0)
        triplet_positive_weight float  - weight on positive distance (>0)
        triplet_negative_weight float  - weight on negative distance (>0)
        triplet_hard_negative_boost float - extra penalty for hard negatives (>=0)
        similarity_threshold    float  - legacy scan threshold fallback (0.0-1.0)
        similarity_threshold_baseline float - scan threshold when personalization is inactive (0.0-1.0)
        similarity_threshold_personalized float - scan threshold when personalization is active (0.0-1.0)
        similarity_threshold_document float - override threshold for document-like labels (0.0-1.0)
        scan_similarity_margin float - min top1-top2 similarity gap for non-document labels (>=0)
        scan_similarity_margin_document float - min top1-top2 similarity gap for document labels (>=0)
        remember_max_prototypes_per_label int - keep this many newest teaches per label (>=1)

    Unrecognised fields are ignored. Type errors return 400.

    Response: full settings state (same schema as GET /settings).
    """
    data, err = read_json_dict(request)
    if err is not None:
        body, status = err
        return jsonify(body), status
    errors = {}
    applied = {}

    for key, expected_type in _PATCHABLE.items():
        if key not in data:
            continue
        raw = data[key]
        value, coerce_error = coerce_json_value(raw, expected_type)
        if coerce_error is not None:
            errors[key] = f"{coerce_error}, got {type(raw).__name__}"
            continue

        if key == "projection_head_weight" and not (0.0 <= value <= 1.0):
            errors[key] = "must be between 0.0 and 1.0"
            continue
        if key == "projection_head_ramp_power" and value <= 0.0:
            errors[key] = "must be > 0.0"
            continue
        if key == "triplet_margin" and value < 0.0:
            errors[key] = "must be >= 0.0"
            continue
        if key in ("triplet_positive_weight", "triplet_negative_weight") and value <= 0.0:
            errors[key] = "must be > 0.0"
            continue
        if key == "triplet_hard_negative_boost" and value < 0.0:
            errors[key] = "must be >= 0.0"
            continue
        if key.startswith("similarity_threshold") and not (0.0 <= value <= 1.0):
            errors[key] = "must be between 0.0 and 1.0"
            continue
        if key.startswith("scan_similarity_margin") and value < 0.0:
            errors[key] = "must be >= 0.0"
            continue
        if key in ("min_feedback_for_training", "projection_head_ramp_at", "projection_head_epochs") and value < 1:
            errors[key] = "must be >= 1"
            continue
        if key == "remember_max_prototypes_per_label" and value < 1:
            errors[key] = "must be >= 1"
            continue

        applied[key] = value

    if errors:
        return jsonify({"errors": errors}), 400

    # write valid values to Settings singleton, then propagate to pipeline in one pass
    s = get_settings()
    for key, value in applied.items():
        setattr(s, key, value)

    # Never force heavy scan-model initialization from /settings PATCH.
    if "enable_learning" in applied:
        apply_scan_learning_if_loaded(s.enable_learning)
    if (
        "projection_head_weight" in applied
        or "projection_head_ramp_at" in applied
        or "projection_head_ramp_power" in applied
    ):
        apply_scan_head_weight_if_loaded(
            s.projection_head_weight,
            s.projection_head_ramp_at,
            s.projection_head_ramp_power,
        )

    get_database().save_ml_settings({
        "enable_learning": s.enable_learning,
        "min_feedback_for_training": s.min_feedback_for_training,
        "projection_head_weight": s.projection_head_weight,
        "projection_head_ramp_at": s.projection_head_ramp_at,
        "projection_head_ramp_power": s.projection_head_ramp_power,
        "projection_head_epochs": s.projection_head_epochs,
        "triplet_margin": s.triplet_margin,
        "triplet_positive_weight": s.triplet_positive_weight,
        "triplet_negative_weight": s.triplet_negative_weight,
        "triplet_hard_negative_boost": s.triplet_hard_negative_boost,
        "similarity_threshold": s.similarity_threshold,
        "similarity_threshold_baseline": s.similarity_threshold_baseline,
        "similarity_threshold_personalized": s.similarity_threshold_personalized,
        "similarity_threshold_document": s.similarity_threshold_document,
        "scan_similarity_margin": s.scan_similarity_margin,
        "scan_similarity_margin_document": s.scan_similarity_margin_document,
        "remember_max_prototypes_per_label": s.remember_max_prototypes_per_label,
    })

    # return full state after applying changes
    return get_settings_route()
