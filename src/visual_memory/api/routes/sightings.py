import time
from flask import Blueprint, request, jsonify

from visual_memory.api.pipelines import get_database

sightings_bp = Blueprint("sightings", __name__)


@sightings_bp.post("/sightings")
def add_sightings():
    """
    Record user-confirmed locations for one or more labels after a scan.

    The frontend already holds the scan result (label, direction, distance_ft,
    similarity). When the user confirms a location name ("kitchen"), the frontend
    sends all of it in a single call. room_name is optional - omit it if the
    user just confirmed the detection without naming the room.

    Body (JSON):
        room_name  (str, optional)  - user-provided room or location name
        sightings  (list, required) - one entry per confirmed label:
            label        (str, required)
            direction    (str, optional)   - from scan response
            distance_ft  (float, optional) - from scan response
            similarity   (float, optional) - from scan response

    Response:
        {"saved": int, "labels": [str]}
    """
    data = request.get_json(silent=True) or {}
    room_name = (data.get("room_name") or "").strip() or None
    entries = data.get("sightings")

    if not entries or not isinstance(entries, list):
        return jsonify({"error": "missing field: sightings (must be a non-empty list)"}), 400

    db = get_database()
    now = time.time()
    saved_labels = []

    for entry in entries:
        label = (entry.get("label") or "").strip()
        if not label:
            continue
        db.add_sighting(
            label=label,
            direction=entry.get("direction"),
            distance_ft=entry.get("distance_ft"),
            similarity=entry.get("similarity"),
            room_name=room_name,
            timestamp=now,
        )
        saved_labels.append(label)

    if not saved_labels:
        return jsonify({"error": "no valid sighting entries (each must have a non-empty label)"}), 400

    return jsonify({"saved": len(saved_labels), "labels": saved_labels})
