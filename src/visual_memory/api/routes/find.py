import re
import time
from typing import Optional

from flask import Blueprint, request, jsonify

from visual_memory.api.pipelines import get_database

find_bp = Blueprint("find", __name__)

_ROOM_PREFIX = re.compile(r"^(in (the|my) |the |my )", re.IGNORECASE)


def _normalize_room(name: str) -> str:
    return _ROOM_PREFIX.sub("", name.strip()).strip().lower() or ""


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


def build_narration(label: str, sighting: dict) -> str:
    """Build a spoken narration string from a sighting dict.

    Examples:
        "Your wallet is in the kitchen, to your left. Last seen 5 minutes ago."
        "Your wallet is to your left, 2.3 feet away. Last seen just now."
        "Your wallet was last seen 3 days ago."
    """
    parts = []
    room = sighting.get("room_name")
    direction = sighting.get("direction")
    distance = sighting.get("distance_ft")
    last_seen = sighting.get("last_seen", "some time ago")

    location_parts = []
    if room:
        location_parts.append(f"in the {room}")
    if direction:
        if distance:
            location_parts.append(f"{direction}, {distance:.1f} feet away")
        else:
            location_parts.append(direction)

    if location_parts:
        parts.append(f"Your {label} is {', '.join(location_parts)}.")
    else:
        parts.append(f"Your {label} was last seen {last_seen}.")
        return " ".join(parts)

    parts.append(f"Last seen {last_seen}.")
    return " ".join(parts)


def _fuzzy_label_match(query: str, threshold: float) -> list[str]:
    """Return known labels whose text embedding is within `threshold` of the query."""
    from visual_memory.api.pipelines import get_scan_pipeline
    from visual_memory.utils.similarity_utils import cosine_similarity

    db = get_database()
    pipeline = get_scan_pipeline()
    if pipeline.text_embedder is None:
        return []

    query_emb = pipeline.text_embedder.embed_text(query)

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


def _ocr_content_match(query: str, threshold: float) -> Optional[str]:
    """Search OCR text of all taught items for semantic match.

    Uses the pre-embedded OCR vector stored at teach time when available.
    Falls back to re-embedding on the fly for older items that lack a stored
    embedding, so the function is backward-compatible with existing databases.

    Returns the best-matching label above threshold, or None.
    """
    from visual_memory.api.pipelines import get_scan_pipeline
    from visual_memory.utils.similarity_utils import cosine_similarity

    db = get_database()
    pipeline = get_scan_pipeline()
    if pipeline.text_embedder is None:
        return None

    items = db.get_items_with_ocr()
    if not items:
        return None

    query_emb = pipeline.text_embedder.embed_text(query)

    best_label: Optional[str] = None
    best_sim = -1.0
    for item in items:
        ocr_emb = item.get("ocr_embedding")
        if ocr_emb is None:
            # Backward compat: item was taught before OCR pre-embedding was added
            ocr_emb = pipeline.text_embedder.embed_text(item["ocr_text"])
        sim = cosine_similarity(query_emb, ocr_emb).item()
        if sim >= threshold and sim > best_sim:
            best_sim = sim
            best_label = item["label"]

    return best_label


def _parse_float_param(value: str, name: str):
    try:
        return float(value), None
    except ValueError:
        from flask import jsonify
        return None, (jsonify({"error": f"{name} must be a Unix timestamp"}), 400)


@find_bp.get("/find")
def find():
    label = request.args.get("label", "").strip()
    room_raw = request.args.get("room", "").strip()
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

    # ---- room query: list all items last seen in a room ----
    if room_raw and not label:
        room = _normalize_room(room_raw)
        rows = db.get_labels_last_seen_in_room(room)
        results = [_format_sighting(r) for r in rows]
        for s in results:
            s["narration"] = build_narration(s["label"], s)
        return jsonify({"room": room, "results": results, "count": len(results)})

    # ---- list all known items (no label, no room) ----
    if not label:
        labels = db.get_known_labels()
        results = []
        for lbl in labels:
            row = db.get_last_sighting(lbl)
            if row:
                s = _format_sighting(row)
                s["narration"] = build_narration(lbl, s)
                results.append(s)
        return jsonify({"results": results, "count": len(results)})

    # ---- label lookup ----
    rows = db.get_sightings(label=label, limit=limit, since=since, before=before)

    fuzzy_matched: Optional[str] = None
    ocr_matched: Optional[str] = None

    if not rows:
        from visual_memory.api.pipelines import get_settings
        settings = get_settings()
        candidates = _fuzzy_label_match(label, settings.text_similarity_threshold)
        if candidates:
            fuzzy_matched = candidates[0]
            rows = db.get_sightings(label=fuzzy_matched, limit=limit, since=since, before=before)

    # ---- OCR content fallback ----
    if not rows:
        from visual_memory.api.pipelines import get_settings
        settings = get_settings()
        ocr_matched = _ocr_content_match(label, settings.text_similarity_threshold)
        if ocr_matched:
            rows = db.get_sightings(label=ocr_matched, limit=limit, since=since, before=before)

    if not rows:
        return jsonify({"label": label, "found": False, "sightings": []})

    resolved_label = ocr_matched or fuzzy_matched or label
    sightings = [_format_sighting(r) for r in rows]
    out = {
        "label": label,
        "found": True,
        "last_sighting": sightings[0],
        "narration": build_narration(resolved_label, sightings[0]),
        "sightings": sightings,
    }
    if fuzzy_matched is not None:
        out["matched_label"] = fuzzy_matched
        out["matched_by"] = "fuzzy_label"
    if ocr_matched is not None:
        out["matched_label"] = ocr_matched
        out["matched_by"] = "ocr"
    return jsonify(out)
