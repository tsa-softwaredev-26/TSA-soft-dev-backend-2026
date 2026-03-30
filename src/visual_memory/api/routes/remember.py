import tempfile
from pathlib import Path

from flask import Blueprint, request, jsonify

from visual_memory.api.pipelines import get_remember_pipeline, get_scan_pipeline

remember_bp = Blueprint("remember", __name__)


@remember_bp.post("/remember")
def remember():
    """
    Teach the system a new object.

    Single image:
        multipart/form-data fields:
            image  (file, required)
            prompt (str,  required)

    Multi-image (frontend sends several frames; backend picks the highest-
    confidence detection and saves only that one):
        multipart/form-data fields:
            images[]  (file[], required - one or more files)
            prompt    (str,    required)

    Response (success):
        {
            "success": true,
            "message": "...",
            "images_tried": int,            # present when multi-image
            "images_with_detection": int,   # present when multi-image
            "result": {
                "label": str,
                "confidence": float,
                "detection_quality": "low" | "medium" | "high",
                "detection_hint": str,
                "blur_score": float,
                "is_blurry": bool,
                "second_pass": bool,
                "second_pass_prompt": str | null,
                "box": [x1, y1, x2, y2],
                "ocr_text": str,
                "ocr_confidence": float
            }
        }
    """
    result, status = process_remember_request(
        prompt=request.form.get("prompt", ""),
        image_file=request.files.get("image"),
        image_files=request.files.getlist("images[]"),
    )
    return jsonify(result), status


# helpers

def _save_temp(image_file) -> Path:
    suffix = Path(image_file.filename).suffix if image_file.filename else ".jpg"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    image_file.save(tmp.name)
    tmp.close()
    return Path(tmp.name)


def _remember_single(image_file, prompt: str) -> tuple[dict, int]:
    tmp_path = _save_temp(image_file)
    try:
        result = get_remember_pipeline().run(tmp_path, prompt)
    except Exception as exc:
        return {"error": str(exc)}, 500
    finally:
        tmp_path.unlink(missing_ok=True)

    if result.get("success"):
        get_scan_pipeline().reload_database()

    return result, 200


def _remember_multi(image_files, prompt: str) -> tuple[dict, int]:
    """
    Save all images to temp files, run detect_score on each to find the best
    candidate, then run the full pipeline only on the winner.
    """
    pipeline = get_remember_pipeline()
    tmp_paths = []
    try:
        for f in image_files:
            tmp_paths.append(_save_temp(f))

        # Rank by detection score (lightweight, no embedding/DB write)
        try:
            scores = pipeline.detect_score_batch(tmp_paths, prompt)
        except Exception:
            scores = []
            for p in tmp_paths:
                try:
                    s = pipeline.detect_score(p, prompt)
                except Exception:
                    s = {
                        "detected": False,
                        "score": 0.0,
                        "blur_score": 0.0,
                        "is_dark": False,
                        "darkness_level": 0.0,
                        "second_pass_prompt": None,
                    }
                scores.append(s)

        detected_count = sum(1 for s in scores if s["detected"])

        # Pick the image with the highest detection score
        best_idx = max(range(len(scores)), key=lambda i: scores[i]["score"])

        if not scores[best_idx]["detected"]:
            return {
                "success": False,
                "message": "No object detected in any of the provided images.",
                "images_tried": len(tmp_paths),
                "images_with_detection": 0,
                "result": None,
            }, 200

        try:
            result = pipeline.run(tmp_paths[best_idx], prompt)
        except Exception as exc:
            return {"error": str(exc)}, 500

    finally:
        for p in tmp_paths:
            p.unlink(missing_ok=True)

    if result.get("success"):
        get_scan_pipeline().reload_database()
        result["images_tried"] = len(tmp_paths)
        result["images_with_detection"] = detected_count

    return result, 200


def process_remember_request(prompt: str, image_file, image_files) -> tuple[dict, int]:
    prompt = (prompt or "").strip()
    if not prompt:
        return {"error": "missing field: prompt"}, 400

    if image_files:
        return _remember_multi(image_files, prompt)

    if not image_file:
        return {"error": "missing field: image or images[]"}, 400
    return _remember_single(image_file, prompt)
