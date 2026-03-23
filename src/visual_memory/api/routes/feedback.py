from flask import Blueprint, request, jsonify

from visual_memory.api.pipelines import get_scan_pipeline, get_feedback_store, get_settings

feedback_bp = Blueprint("feedback", __name__)

_VALID_FEEDBACK = {"correct", "wrong"}


@feedback_bp.post("/feedback")
def feedback():
    data = request.get_json(silent=True) or {}

    scan_id = data.get("scan_id", "").strip()
    label = data.get("label", "").strip()
    fb = data.get("feedback", "").strip()

    if not scan_id:
        return jsonify({"error": "missing field: scan_id"}), 400
    if not label:
        return jsonify({"error": "missing field: label"}), 400
    if fb not in _VALID_FEEDBACK:
        return jsonify({"error": 'feedback must be "correct" or "wrong"'}), 400

    cached = get_scan_pipeline().get_cached_embeddings(scan_id, label)
    if cached is None:
        return jsonify({"error": "scan_id not found or expired"}), 404

    anchor_emb, query_emb = cached
    store = get_feedback_store()

    if fb == "correct":
        store.record_positive(anchor_emb, query_emb, label)
    else:
        store.record_negative(anchor_emb, query_emb, label)

    counts = store.count()
    return jsonify({
        "recorded": True,
        "label": label,
        "feedback": fb,
        "triplets": counts["triplets"],
        "min_for_training": get_settings().min_feedback_for_training,
    })
