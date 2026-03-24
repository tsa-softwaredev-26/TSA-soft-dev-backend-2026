from flask import Blueprint, request, jsonify

from visual_memory.api.pipelines import get_database, get_scan_pipeline

items_bp = Blueprint("items", __name__)


@items_bp.delete("/items/<label>")
def delete_item(label):
    db = get_database()
    count = db.delete_items_by_label(label)
    if count == 0:
        return jsonify({"error": "not found"}), 404
    get_scan_pipeline().reload_database()
    return jsonify({"deleted": True, "label": label, "count": count})
