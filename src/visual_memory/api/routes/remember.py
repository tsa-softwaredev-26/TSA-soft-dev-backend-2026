import time
from pathlib import Path
from uuid import uuid4

from flask import Blueprint, jsonify, request

from visual_memory.api.pipelines import get_remember_pipeline, get_scan_pipeline

remember_bp = Blueprint("remember", __name__)
_UPLOADS_DIR = Path(__file__).resolve().parents[4] / "uploads"


def _save_upload(image_file) -> Path:
    suffix = Path(image_file.filename).suffix if image_file.filename else ".jpg"
    _UPLOADS_DIR.mkdir(exist_ok=True)
    path = _UPLOADS_DIR / f"remember_{int(time.time() * 1000)}_{uuid4().hex}{suffix}"
    image_file.save(str(path))
    return path


def _remember_single(image_file, prompt: str) -> tuple[dict, int]:
    tmp_path = _save_upload(image_file)
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
    pipeline = get_remember_pipeline()
    tmp_paths = []
    try:
        for f in image_files:
            tmp_paths.append(_save_upload(f))

        try:
            scores = pipeline.detect_score_batch(tmp_paths, prompt)
        except Exception:
            scores = []
            for p in tmp_paths:
                try:
                    score = pipeline.detect_score(p, prompt)
                except Exception:
                    score = {
                        "detected": False,
                        "score": 0.0,
                        "blur_score": 0.0,
                        "is_dark": False,
                        "darkness_level": 0.0,
                        "second_pass_prompt": None,
                    }
                scores.append(score)

        detected_count = sum(1 for s in scores if s["detected"])
        best_idx = max(range(len(scores)), key=lambda i: scores[i]["score"]) if scores else -1

        if best_idx < 0 or not scores[best_idx]["detected"]:
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


def process_remember_request(prompt: str, image_file=None, image_files=None) -> tuple[dict, int]:
    normalized_prompt = (prompt or "").strip()
    if not normalized_prompt:
        return {"error": "missing field: prompt"}, 400
    if image_files:
        return _remember_multi(image_files, normalized_prompt)
    if image_file is None:
        return {"error": "missing field: image or images[]"}, 400
    return _remember_single(image_file, normalized_prompt)


@remember_bp.post("/remember")
def remember():
    result, status = process_remember_request(
        prompt=request.form.get("prompt", ""),
        image_file=request.files.get("image"),
        image_files=request.files.getlist("images[]"),
    )
    return jsonify(result), status
