import tempfile
from pathlib import Path

from flask import Blueprint, request, jsonify

from visual_memory.api.pipelines import get_remember_pipeline, get_scan_pipeline

remember_bp = Blueprint("remember", __name__)


@remember_bp.post("/remember")
def remember():
    if "image" not in request.files:
        return jsonify({"error": "missing field: image"}), 400
    prompt = request.form.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "missing field: prompt"}), 400

    image_file = request.files["image"]
    suffix = Path(image_file.filename).suffix if image_file.filename else ".jpg"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        image_file.save(tmp_path)

    try:
        result = get_remember_pipeline().run(tmp_path, prompt)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        tmp_path.unlink(missing_ok=True)

    if result.get("success"):
        get_scan_pipeline().reload_database()

    return jsonify(result)
