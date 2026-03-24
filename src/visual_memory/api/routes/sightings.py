from flask import Blueprint, jsonify

from visual_memory.api.pipelines import get_database

sightings_bp = Blueprint("sightings", __name__)


@sightings_bp.delete("/sightings/<int:sighting_id>")
def delete_sighting(sighting_id):
    db = get_database()
    deleted = db.delete_sighting(sighting_id)
    if not deleted:
        return jsonify({"error": "not found"}), 404
    return jsonify({"deleted": True, "id": sighting_id})
