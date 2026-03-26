from flask import Blueprint, request, jsonify

from visual_memory.api.pipelines import get_database, get_scan_pipeline, get_user_settings
from visual_memory.config.user_settings import PerformanceMode, _BUTTON_LAYOUTS

user_settings_bp = Blueprint("user_settings", __name__)

# Fields the mobile app may update via PATCH /user-settings.
_PATCHABLE = {
    "performance_mode": str,
    "voice_speed": float,
    "scan_update_location": bool,
    "learning_enabled": bool,
    "button_layout": str,
}


@user_settings_bp.get("/user-settings")
def get_user_settings_route():
    """Return current user preferences.

    Response: {
        "performance_mode": "fast" | "balanced" | "accurate",
        "voice_speed": float,
        "scan_update_location": bool,
        "learning_enabled": bool,
        "button_layout": "default" | "swapped",
        "performance_config": {"depth_enabled": bool, "target_latency": float}
    }
    """
    return jsonify(get_user_settings().to_dict())


@user_settings_bp.patch("/user-settings")
def patch_user_settings():
    """Update one or more user preferences and persist to disk.

    Accepted fields (all optional):
        performance_mode      str   - "fast" | "balanced" | "accurate"
        voice_speed           float - 0.5 to 2.0
        scan_update_location  bool  - prompt to update location after found
        learning_enabled      bool  - collect feedback and adapt with use
        button_layout         str   - "default" | "swapped" (stub)

    Unrecognised fields are ignored. Type and range errors return 400.
    Response: full user settings state (same schema as GET /user-settings).
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

        if key == "performance_mode":
            try:
                value = PerformanceMode(value)
            except ValueError:
                errors[key] = f"must be one of: {[m.value for m in PerformanceMode]}"
                continue

        if key == "voice_speed" and not (0.5 <= value <= 2.0):
            errors[key] = "must be between 0.5 and 2.0"
            continue

        if key == "button_layout" and value not in _BUTTON_LAYOUTS:
            errors[key] = f"must be one of: {sorted(_BUTTON_LAYOUTS)}"
            continue

        applied[key] = value

    if errors:
        return jsonify({"errors": errors}), 400

    us = get_user_settings()
    for key, value in applied.items():
        setattr(us, key, value)

    # propagate learning toggle to scan pipeline immediately
    if "learning_enabled" in applied:
        get_scan_pipeline().set_enable_learning(us.learning_enabled)

    us.save(get_database())

    return jsonify(us.to_dict())
