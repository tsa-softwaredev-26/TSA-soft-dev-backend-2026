import threading
from pathlib import Path

from flask import Blueprint, jsonify

from visual_memory.api.pipelines import get_database, get_feedback_store, get_scan_pipeline, get_settings
from visual_memory.learning import ProjectionTrainer
from visual_memory.utils import get_logger

_log = get_logger(__name__)

retrain_bp = Blueprint("retrain", __name__)

_status: dict = {"running": False, "last_result": None, "error": None}
_lock = threading.Lock()


def _run_training(settings, store, pipeline) -> None:
    """Background thread: train, save, reload. Updates _status when done."""
    global _status
    try:
        triplets = store.load_triplets()
        trainer = ProjectionTrainer(pipeline._head, lr=1e-4)
        final_loss = trainer.train(triplets, epochs=settings.projection_head_epochs)

        head_path = Path(settings.projection_head_path)
        head_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save(head_path)
        get_database().save_projection_head(pipeline._head.state_dict())

        pipeline.reload_head()
        triplet_count = store.count()["triplets"]
        pipeline._triplet_count = triplet_count

        effective_weight = min(
            settings.projection_head_weight,
            settings.projection_head_weight * triplet_count / max(settings.projection_head_ramp_at, 1),
        )
        result = {
            "trained": True,
            "triplets": triplet_count,
            "final_loss": round(final_loss, 6),
            "head_weight": round(effective_weight, 4),
        }
        _log.info({"event": "retrain_complete", **result})
        with _lock:
            _status["running"] = False
            _status["last_result"] = result
            _status["error"] = None
    except Exception as exc:
        _log.error({"event": "retrain_error", "error": str(exc)})
        with _lock:
            _status["running"] = False
            _status["error"] = str(exc)


@retrain_bp.post("/retrain")
def retrain():
    """Start projection head training in a background thread.

    Returns immediately with {"started": true}. Poll GET /retrain/status
    to check progress and retrieve the final result.

    Response (started):
        {"started": true, "triplets": N}

    Response (already running):
        {"started": false, "reason": "already_running"}

    Response (insufficient data):
        {"started": false, "reason": "insufficient_data", "triplets": N, "min_required": M}
    """
    with _lock:
        if _status["running"]:
            return jsonify({"started": False, "reason": "already_running"})

        settings = get_settings()
        store = get_feedback_store()
        counts = store.count()
        triplet_count = counts["triplets"]

        if triplet_count < settings.min_feedback_for_training:
            return jsonify({
                "started": False,
                "reason": "insufficient_data",
                "triplets": triplet_count,
                "min_required": settings.min_feedback_for_training,
            })

        pipeline = get_scan_pipeline()
        _status["running"] = True
        _status["error"] = None

    thread = threading.Thread(
        target=_run_training,
        args=(settings, store, pipeline),
        daemon=True,
    )
    thread.start()
    return jsonify({"started": True, "triplets": triplet_count})


@retrain_bp.get("/retrain/status")
def retrain_status():
    """Poll training progress.

    Response:
        {
          "running": bool,
          "last_result": {"trained": true, "triplets": N, "final_loss": 0.012, "head_weight": 0.5} | null,
          "error": "..." | null
        }
    """
    with _lock:
        return jsonify({
            "running": _status["running"],
            "last_result": _status["last_result"],
            "error": _status["error"],
        })
