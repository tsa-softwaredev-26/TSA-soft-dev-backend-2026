from flask import Blueprint, request, jsonify

from visual_memory.api.pipelines import get_database, reload_scan_database_if_loaded
from visual_memory.utils import get_logger
from ._json_utils import read_json_dict

items_bp = Blueprint("items", __name__)
_log = get_logger(__name__)


@items_bp.get("/items")
def list_items():
    label = request.args.get("label", "").strip() or None
    db = get_database()
    items = db.get_items_metadata(label=label)
    return jsonify({"items": items, "count": len(items)})


@items_bp.delete("/items/<label>")
def delete_item(label):
    label = (label or "").strip()
    if not label:
        return jsonify({"error": "label cannot be empty"}), 400
    db = get_database()
    count = db.delete_items_by_label(label)
    if count == 0:
        return jsonify({"error": "not found"}), 404

    # Deletion already succeeded in DB; only refresh in-memory scan cache if pipeline is warm.
    try:
        reload_scan_database_if_loaded()
    except Exception as exc:
        _log.warning({
            "event": "delete_item_cache_reload_failed",
            "label": label,
            "deleted_count": count,
            "error": str(exc),
        })
    return jsonify({"deleted": True, "label": label, "count": count})


@items_bp.post("/items/<label>/rename")
def rename_item(label):
    """
    Rename all items stored under <label> to a new label.

    Body (JSON):
        new_label (str, required)
        force     (bool, optional, default false)

    If new_label already exists and force=false, returns 409 with
    {"conflict": true, ...} so the client can ask the user to confirm,
    then resend with force=true to overwrite.
    """
    data, err = read_json_dict(request)
    if err is not None:
        body, status = err
        return jsonify(body), status
    new_label = (data.get("new_label") or "").strip()
    if not new_label:
        return jsonify({"error": "missing field: new_label"}), 400
    if new_label == label:
        return jsonify({"error": "new_label is the same as the current label"}), 400

    force = bool(data.get("force", False))
    db = get_database()

    if not db.get_items_metadata(label=label):
        return jsonify({"error": "not found"}), 404

    conflict_items = db.get_items_metadata(label=new_label)
    if conflict_items and not force:
        return jsonify({
            "conflict": True,
            "message": (
                f'A memory named "{new_label}" already exists '
                f"({len(conflict_items)} item(s)). "
                "Send with force=true to overwrite it."
            ),
            "existing_count": len(conflict_items),
        }), 409

    result = db.rename_label(label, new_label)
    try:
        reload_scan_database_if_loaded()
    except Exception as exc:
        _log.warning({
            "event": "rename_item_cache_reload_failed",
            "old_label": label,
            "new_label": new_label,
            "error": str(exc),
        })
    return jsonify({
        "renamed": True,
        "old_label": label,
        "new_label": new_label,
        "count": result["renamed"],
        "replaced": result["replaced"],
    })
