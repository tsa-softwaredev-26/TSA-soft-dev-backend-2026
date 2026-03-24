import time

from flask import Blueprint, request, jsonify

from visual_memory.api.pipelines import get_database

find_bp = Blueprint("find", __name__)


def _format_sighting(row: dict) -> dict:
    """Add a human-readable `last_seen` string to a sighting dict."""
    out = dict(row)
    ts = row.get("timestamp")
    if ts is not None:
        age = time.time() - ts
        if age < 60:
            out["last_seen"] = "just now"
        elif age < 3600:
            mins = int(age // 60)
            out["last_seen"] = f"{mins} minute{'s' if mins != 1 else ''} ago"
        elif age < 86400:
            hours = int(age // 3600)
            out["last_seen"] = f"{hours} hour{'s' if hours != 1 else ''} ago"
        else:
            days = int(age // 86400)
            out["last_seen"] = f"{days} day{'s' if days != 1 else ''} ago"
    return out


@find_bp.get("/find")
def find():
    label = request.args.get("label", "").strip()
    limit_raw = request.args.get("limit", "1").strip()

    try:
        limit = int(limit_raw)
        if limit < 1 or limit > 100:
            raise ValueError
    except ValueError:
        return jsonify({"error": "limit must be an integer between 1 and 100"}), 400

    db = get_database()

    if not label:
        # No label given - return the most recent sighting for each known label.
        labels = db.get_known_labels()
        results = []
        for lbl in labels:
            row = db.get_last_sighting(lbl)
            if row:
                results.append(_format_sighting(row))
        return jsonify({"results": results, "count": len(results)})

    rows = db.get_sightings(label=label, limit=limit)
    if not rows:
        return jsonify({"label": label, "found": False, "sightings": []})

    sightings = [_format_sighting(r) for r in rows]
    return jsonify({
        "label": label,
        "found": True,
        "last_sighting": sightings[0],
        "sightings": sightings,
    })
