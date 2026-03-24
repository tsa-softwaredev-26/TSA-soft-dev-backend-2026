import io

from flask import Blueprint, request, Response, jsonify

from visual_memory.api.pipelines import get_scan_pipeline

crop_bp = Blueprint("crop", __name__)


@crop_bp.get("/crop")
def crop():
    scan_id = request.args.get("scan_id", "").strip()
    index_raw = request.args.get("index", "").strip()

    if not scan_id:
        return jsonify({"error": "missing param: scan_id"}), 400
    try:
        index = int(index_raw)
    except (ValueError, TypeError):
        return jsonify({"error": "index must be an integer"}), 400

    img = get_scan_pipeline().get_cached_crop(scan_id, index)
    if img is None:
        return jsonify({"error": "not found"}), 404

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return Response(buf.getvalue(), mimetype="image/jpeg")
