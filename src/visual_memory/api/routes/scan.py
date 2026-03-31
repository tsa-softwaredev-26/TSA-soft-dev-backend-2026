from pathlib import Path
from uuid import uuid4

from flask import Blueprint
from PIL import Image

from visual_memory.api.pipelines import get_database, get_scan_pipeline

scan_bp = Blueprint("scan", __name__)
_SCANS_DIR = Path(__file__).resolve().parents[4] / "scans"


<<<<<<< HEAD
def process_scan_request(image_file, focal_length_raw: str) -> tuple[dict, int]:
=======
def process_scan_request(image_file, focal_length_raw: str = "") -> tuple[dict, int]:
>>>>>>> api/whisper-update
    if image_file is None:
        return {"error": "missing field: image"}, 400

    focal_length_px = 0.0
<<<<<<< HEAD
    raw = (focal_length_raw or "").strip()
    if raw:
=======
    if focal_length_raw:
>>>>>>> api/whisper-update
        try:
            focal_length_px = float(focal_length_raw)
        except ValueError:
            return {"error": "focal_length_px must be a float"}, 400

    pipeline = get_scan_pipeline()
    try:
        image = Image.open(image_file.stream).convert("RGB")
    except Exception as exc:
        return {"error": f"could not open image: {exc}"}, 400

    scan_id = uuid4().hex
    try:
        result = pipeline.run(image, scan_id=scan_id, focal_length_px=focal_length_px)
    except Exception as exc:
        return {"error": str(exc)}, 500

    db = get_database()
    for i, match in enumerate(result.get("matches", [])):
        crop_path = None
        crop = pipeline.get_cached_crop(scan_id, i)
        if crop is not None:
            _SCANS_DIR.mkdir(exist_ok=True)
            crop_path = str(_SCANS_DIR / f"{scan_id}_{i}.jpg")
            crop.save(crop_path, format="JPEG", quality=85)
        db.add_sighting(
            label=match["label"],
            direction=match.get("direction"),
            distance_ft=match.get("distance_ft"),
            similarity=match.get("similarity"),
            crop_path=crop_path,
        )

    result["scan_id"] = scan_id
    return result, 200
<<<<<<< HEAD


@scan_bp.post("/scan")
def scan():
    result, status = process_scan_request(
        image_file=request.files.get("image"),
        focal_length_raw=request.form.get("focal_length_px", ""),
    )
    return jsonify(result), status
=======
>>>>>>> api/whisper-update
