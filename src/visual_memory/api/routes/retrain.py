from pathlib import Path

from flask import Blueprint, jsonify

from visual_memory.api.pipelines import get_database, get_feedback_store, get_scan_pipeline, get_settings
from visual_memory.learning import ProjectionTrainer
from visual_memory.utils import get_logger

_log = get_logger(__name__)

retrain_bp = Blueprint("retrain", __name__)


@retrain_bp.post("/retrain")
def retrain():
    """Train the projection head from collected feedback triplets.

    Checks that enough triplets have been collected (min_feedback_for_training),
    runs ProjectionTrainer, saves weights, and reloads the head in ScanPipeline
    so the updated model is used immediately on the next /scan request.

    Response (insufficient data):
        {"trained": false, "reason": "insufficient_data", "triplets": N, "min_required": M}

    Response (success):
        {"trained": true, "triplets": N, "final_loss": 0.0123, "head_weight": 0.5}
    """
    settings = get_settings()
    store = get_feedback_store()
    counts = store.count()
    triplet_count = counts["triplets"]

    if triplet_count < settings.min_feedback_for_training:
        return jsonify({
            "trained": False,
            "reason": "insufficient_data",
            "triplets": triplet_count,
            "min_required": settings.min_feedback_for_training,
        })

    triplets = store.load_triplets()

    pipeline = get_scan_pipeline()
    trainer = ProjectionTrainer(pipeline._head, lr=1e-4)
    final_loss = trainer.train(triplets, epochs=settings.projection_head_epochs)

    head_path = Path(settings.projection_head_path)
    trainer.save(head_path)
    get_database().save_projection_head(pipeline._head.state_dict())

    pipeline.reload_head()
    pipeline._triplet_count = triplet_count

    # effective blend weight at current triplet count, for caller info
    effective_weight = min(
        settings.projection_head_weight,
        settings.projection_head_weight * triplet_count / max(settings.projection_head_ramp_at, 1),
    )

    _log.info({
        "event": "retrain_complete",
        "triplets": triplet_count,
        "final_loss": round(final_loss, 6),
        "effective_head_weight": round(effective_weight, 4),
    })

    return jsonify({
        "trained": True,
        "triplets": triplet_count,
        "final_loss": round(final_loss, 6),
        "head_weight": round(effective_weight, 4),
    })
