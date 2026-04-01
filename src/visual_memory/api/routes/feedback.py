from flask import Blueprint, request, jsonify

from visual_memory.api.pipelines import get_scan_pipeline, get_feedback_store, get_settings
from ._json_utils import read_json_dict

feedback_bp = Blueprint("feedback", __name__)

_VALID_FEEDBACK = {"correct", "wrong"}


@feedback_bp.post("/feedback")
def feedback():
    data, err = read_json_dict(request)
    if err is not None:
        body, status = err
        return jsonify(body), status

    scan_id = str(data.get("scan_id", "") or "").strip()
    label = str(data.get("label", "") or "").strip()
    fb = str(data.get("feedback", "") or "").strip()

    if not scan_id:
        return jsonify({"error": "missing field: scan_id"}), 400
    if not label:
        return jsonify({"error": "missing field: label"}), 400
    if fb not in _VALID_FEEDBACK:
        return jsonify({"error": 'feedback must be "correct" or "wrong"'}), 400

    cached = get_scan_pipeline().get_cached_embeddings(scan_id, label)
    if cached is None:
        scan_meta = get_scan_pipeline().get_scan_cache_meta(scan_id)
        payload = {"error": "scan_id not found or expired", "scan_id": scan_id}
        if scan_meta:
            payload["cache"] = scan_meta
        return jsonify(payload), 410

    match_meta = get_scan_pipeline().get_cached_match_meta(scan_id, label)

    if not isinstance(cached, (tuple, list)) or len(cached) != 2:
        return jsonify({"error": "invalid cached embedding payload", "scan_id": scan_id}), 500
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
        "match_meta": match_meta or {},
    })
