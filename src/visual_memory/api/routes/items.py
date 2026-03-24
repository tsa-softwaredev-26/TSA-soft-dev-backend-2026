from flask import Blueprint, request, jsonify

from visual_memory.api.pipelines import get_database, get_scan_pipeline

items_bp = Blueprint("items", __name__)


@items_bp.get("/items")
def list_items():
    label = request.args.get("label", "").strip() or None
    db = get_database()
    items = db.get_items_metadata(label=label)
    return jsonify({"items": items, "count": len(items)})


@items_bp.delete("/items/<label>")
def delete_item(label):
    db = get_database()
    count = db.delete_items_by_label(label)
    if count == 0:
        return jsonify({"error": "not found"}), 404
    get_scan_pipeline().reload_database()
    return jsonify({"deleted": True, "label": label, "count": count})
