from uuid import uuid4

from flask import Blueprint, request, jsonify
from PIL import Image

from visual_memory.api.pipelines import get_scan_pipeline

scan_bp = Blueprint("scan", __name__)


@scan_bp.post("/scan")
def scan():
    if "image" not in request.files:
        return jsonify({"error": "missing field: image"}), 400

    focal_length_px = 0.0
    raw = request.form.get("focal_length_px", "")
    if raw:
        try:
            focal_length_px = float(raw)
        except ValueError:
            return jsonify({"error": "focal_length_px must be a float"}), 400

    pipeline = get_scan_pipeline()

    image_file = request.files["image"]
    try:
        image = Image.open(image_file.stream).convert("RGB")
    except Exception as exc:
        return jsonify({"error": f"could not open image: {exc}"}), 400

    scan_id = uuid4().hex

    try:
        result = pipeline.run(image, scan_id=scan_id, focal_length_px=focal_length_px)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    result["scan_id"] = scan_id
    return jsonify(result)
