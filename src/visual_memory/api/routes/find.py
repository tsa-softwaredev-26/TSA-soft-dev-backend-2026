import time
from typing import Optional

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


def _fuzzy_label_match(query: str, threshold: float) -> list[str]:
    """Return known labels whose text embedding is within `threshold` of the query.

    Uses stored label embeddings from DB when available; falls back to
    embedding on the fly for labels that predate the label_embedding column.
    """
    from visual_memory.api.pipelines import get_scan_pipeline
    from visual_memory.utils.similarity_utils import cosine_similarity

    db = get_database()
    pipeline = get_scan_pipeline()
    if pipeline.text_embedder is None:
        return []

    query_emb = pipeline.text_embedder.embed_text(query)

    # Build label -> embedding map; stored embeddings take priority
    stored = {row["label"]: row["embedding"] for row in db.get_label_embeddings()}
    all_labels = db.get_known_labels()
    label_embs: dict[str, object] = {}
    for lbl in all_labels:
        if lbl in stored:
            label_embs[lbl] = stored[lbl]
        else:
            label_embs[lbl] = pipeline.text_embedder.embed_text(lbl)

    if not label_embs:
        return []

    matched: list[tuple[str, float]] = []
    for lbl, emb in label_embs.items():
        sim = cosine_similarity(query_emb, emb).item()
        if sim >= threshold:
            matched.append((lbl, sim))
    matched.sort(key=lambda x: x[1], reverse=True)
    return [lbl for lbl, _ in matched]


def _parse_float_param(value: str, name: str):
    """Parse a float query param; return (float, None) or (None, error response)."""
    try:
        return float(value), None
    except ValueError:
        from flask import jsonify
        return None, (jsonify({"error": f"{name} must be a Unix timestamp"}), 400)


@find_bp.get("/find")
def find():
    label = request.args.get("label", "").strip()
    limit_raw = request.args.get("limit", "1").strip()
    since_raw = request.args.get("since", "").strip()
    before_raw = request.args.get("before", "").strip()

    try:
        limit = int(limit_raw)
        if limit < 1 or limit > 100:
            raise ValueError
    except ValueError:
        return jsonify({"error": "limit must be an integer between 1 and 100"}), 400

    since: Optional[float] = None
    if since_raw:
        since, err = _parse_float_param(since_raw, "since")
        if err:
            return err

    before: Optional[float] = None
    if before_raw:
        before, err = _parse_float_param(before_raw, "before")
        if err:
            return err

    db = get_database()

    if not label:
        labels = db.get_known_labels()
        results = []
        for lbl in labels:
            row = db.get_last_sighting(lbl)
            if row:
                results.append(_format_sighting(row))
        return jsonify({"results": results, "count": len(results)})

    rows = db.get_sightings(label=label, limit=limit, since=since, before=before)

    fuzzy_matched: Optional[str] = None
    if not rows:
        from visual_memory.api.pipelines import get_settings
        settings = get_settings()
        candidates = _fuzzy_label_match(label, settings.text_similarity_threshold)
        if candidates:
            fuzzy_matched = candidates[0]
            rows = db.get_sightings(
                label=fuzzy_matched, limit=limit, since=since, before=before
            )

    if not rows:
        return jsonify({"label": label, "found": False, "sightings": []})

    sightings = [_format_sighting(r) for r in rows]
    out = {
        "label": label,
        "found": True,
        "last_sighting": sightings[0],
        "sightings": sightings,
    }
    if fuzzy_matched is not None:
        out["matched_label"] = fuzzy_matched
    return jsonify(out)
