from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.get("/health")
def health():
    """Return a lightweight liveness response for probes and quick checks."""
    return jsonify({"status": "ok"})
