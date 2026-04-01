import re
import time
from flask import Blueprint, request, jsonify

from visual_memory.api.pipelines import get_database
from ._json_utils import read_json_dict

sightings_bp = Blueprint("sightings", __name__)

# Prefixes stripped before storing room names so that natural speech variants
# ("in the kitchen", "the kitchen", "my bedroom") all normalize to the same key.
_ROOM_PREFIX = re.compile(
    r"^(in (the|my) |the |my )",
    re.IGNORECASE,
)


def _normalize_room(name: str) -> str:
    """
    Lowercase and strip filler prefixes from a room name.

    "In the Kitchen" -> "kitchen"
    "The Bedroom"    -> "bedroom"
    "my living room" -> "living room"
    "kitchen"        -> "kitchen"
    """
    name = _ROOM_PREFIX.sub("", name.strip()).strip().lower()
    return name or ""


@sightings_bp.post("/sightings")
def add_sightings():
    """
    Record a confirmed location for one or more labels.

    Dual-use: called from both remember mode and scan mode.

    Remember mode (after POST /remember):
        { "room_name": "kitchen", "sightings": [{ "label": "wallet" }] }
        No direction/distance needed - user just placed the object there.

    Scan mode (after POST /scan, user confirms location):
        {
          "room_name": "kitchen",
          "sightings": [
            { "label": "wallet", "direction": "to your left",
              "distance_ft": 3.2, "similarity": 0.61 }
          ]
        }

    room_name is optional in both cases - omit it if the user confirmed the
    detection/placement without naming a room.

    room_name is normalized before storage: "In The Kitchen" -> "kitchen",
    "The Bedroom" -> "bedroom", "my living room" -> "living room".

    Body (JSON):
        room_name  (str, optional)  - user-provided room or location name
        sightings  (list, required) - one entry per label:
            label        (str, required)
            direction    (str, optional)
            distance_ft  (float, optional)
            similarity   (float, optional)

    Response:
        {"saved": int, "labels": [str], "room_name": str | null}
    """
    data, err = read_json_dict(request)
    if err is not None:
        body, status = err
        return jsonify(body), status
    raw_room = (data.get("room_name") or "").strip()
    room_name = _normalize_room(raw_room) or None

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

    return jsonify({"saved": len(saved_labels), "labels": saved_labels, "room_name": room_name})
