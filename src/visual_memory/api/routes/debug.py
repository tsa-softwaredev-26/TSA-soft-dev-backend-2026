"""Debug endpoints for frontend integration testing.

All routes require the normal API_KEY auth when it is set.
Not intended for production traffic - exposes internal state,
includes a destructive wipe, and allows direct config mutation.
"""
import dataclasses
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from flask import Blueprint, jsonify, request
from PIL import Image

import visual_memory.api.pipelines as _pm
from visual_memory.api.pipelines import (
    get_database,
    get_scan_pipeline,
    get_settings,
)
from visual_memory.config.settings import Settings as _Settings
from visual_memory.utils.device_utils import get_device
from visual_memory.utils.metrics import collect_system_metrics
from visual_memory.utils.quality_utils import mean_luminance
from ._json_utils import read_json_dict

debug_bp = Blueprint("debug", __name__)

_TEST_IMAGE = (
    Path(__file__).resolve().parents[2] / "tests" / "input_images" / "wallet_1ft_table.jpg"
)
_TEST_PROMPT = "wallet"
_LOG_DIR = Path(__file__).resolve().parents[4] / "logs"
_APP_LOG = _LOG_DIR / "app.log"
_IMPORTANT_LOG = _LOG_DIR / "important.log"
_CRASH_LOG = _LOG_DIR / "crash.log"

# Build type map for PATCH /debug/config from Settings defaults.
# Only primitive fields (bool, int, float, str) are settable at runtime.
def _build_settable_types() -> dict:
    defaults = _Settings()
    types = {}
    for f in dataclasses.fields(defaults):
        val = getattr(defaults, f.name)
        if isinstance(val, bool):
            types[f.name] = bool
        elif isinstance(val, int):
            types[f.name] = int
        elif isinstance(val, float):
            types[f.name] = float
        elif isinstance(val, str):
            types[f.name] = str
    return types

_SETTABLE_TYPES = _build_settable_types()

_ML_PATCHABLE = [
    "enable_learning", "min_feedback_for_training",
    "projection_head_weight", "projection_head_ramp_at",
    "projection_head_epochs",
]


def _ocr_health(s: _Settings) -> dict:
    url = s.ocr_health_url
    if not url:
        parsed = urllib.parse.urlparse(s.ocr_service_url)
        url = f"{parsed.scheme}://{parsed.netloc}/health"
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            reachable = resp.status == 200
    except (urllib.error.URLError, OSError):
        reachable = False
    latency_ms = round((time.monotonic() - t0) * 1000, 1)
    return {"reachable": reachable, "latency_ms": latency_ms if reachable else None, "url": url}


def _blur_score(image: Image.Image) -> float:
    gray = np.array(image.convert("L"), dtype=np.float32)
    lap = (
        np.roll(gray, 1, 0) + np.roll(gray, -1, 0) +
        np.roll(gray, 1, 1) + np.roll(gray, -1, 1) - 4.0 * gray
    )
    return float(lap.var())


def _tail_json_log(
    path: Path,
    n: int,
    *,
    since_line: int = 0,
    event: str | None = None,
    level: str | None = None,
    tag: str | None = None,
    contains: str | None = None,
) -> list[dict]:
    if not path.exists():
        return []
    contains_l = contains.lower() if contains else None
    level_u = level.upper() if level else None
    all_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    lines = all_lines[max(since_line, 0):]
    out = []
    for line in reversed(lines):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event is not None and rec.get("event") != event:
            continue
        if level_u is not None and str(rec.get("level", "")).upper() != level_u:
            continue
        if tag is not None and rec.get("tag") != tag:
            continue
        if contains_l is not None and contains_l not in json.dumps(rec).lower():
            continue
        out.append(rec)
        if len(out) >= n:
            break
    out.reverse()
    return out


def _tail_crash_blocks(path: Path, n: int, contains: str | None = None) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return []
    blocks = [b.strip() for b in text.split("--- CRASH ") if b.strip()]
    if contains:
        term = contains.lower()
        blocks = [b for b in blocks if term in b.lower()]
    selected = blocks[-n:]
    return [f"--- CRASH {b}" for b in selected]


def _summarize_perf(entries: list[dict]) -> dict:
    durations: dict[str, list[float]] = {}
    ram = []
    swap = []
    vram_alloc = []
    vram_res = []
    cpu_temp = []
    vram_layout_count = 0
    vram_swap_count = 0
    stage_keys = (
        "prepare_ms",
        "detect_ms",
        "embed_ms",
        "ocr_ms",
        "match_ms",
        "dedup_ms",
        "depth_ms",
        "db_ms",
    )
    stage_values: dict[str, list[float]] = {k: [] for k in stage_keys}

    for entry in entries:
        event = entry.get("event")
        duration = entry.get("duration_ms")
        if event and duration is not None:
            try:
                durations.setdefault(event, []).append(float(duration))
            except (TypeError, ValueError):
                pass
        if event == "vram_layout":
            vram_layout_count += 1
            swap_count_raw = entry.get("swap_count")
            if swap_count_raw is None:
                swap_count = len(entry.get("offloaded") or [])
            else:
                try:
                    swap_count = int(swap_count_raw)
                except (TypeError, ValueError):
                    swap_count = len(entry.get("offloaded") or [])
            if swap_count > 0:
                vram_swap_count += 1

        for key in stage_keys:
            value = entry.get(key)
            if value is None:
                continue
            try:
                stage_values[key].append(float(value))
            except (TypeError, ValueError):
                pass

        for key, bucket in (
            ("ram_used_mb", ram),
            ("swap_used_mb", swap),
            ("vram_allocated_mb", vram_alloc),
            ("vram_reserved_mb", vram_res),
            ("cpu_temp_c", cpu_temp),
        ):
            value = entry.get(key)
            if value is None:
                continue
            try:
                bucket.append(float(value))
            except (TypeError, ValueError):
                pass

    duration_summary = {}
    for event, values in durations.items():
        duration_summary[event] = {
            "count": len(values),
            "avg_ms": round(sum(values) / len(values), 1),
            "max_ms": round(max(values), 1),
        }
    stage_summary = {}
    for key, values in stage_values.items():
        if not values:
            continue
        stage_summary[key] = {
            "count": len(values),
            "avg_ms": round(sum(values) / len(values), 1),
            "max_ms": round(max(values), 1),
        }

    return {
        "event_count": len(entries),
        "vram_layout_count": vram_layout_count,
        "vram_swap_count": vram_swap_count,
        "duration_ms": duration_summary,
        "stage_ms": stage_summary,
        "peaks": {
            "ram_used_mb": round(max(ram), 1) if ram else None,
            "swap_used_mb": round(max(swap), 1) if swap else None,
            "vram_allocated_mb": round(max(vram_alloc), 1) if vram_alloc else None,
            "vram_reserved_mb": round(max(vram_res), 1) if vram_res else None,
            "cpu_temp_c": round(max(cpu_temp), 1) if cpu_temp else None,
        },
    }


@debug_bp.get("/debug/state")
def debug_state():
    """System overview: pipeline readiness, OCR service, DB counts, active settings."""
    s = get_settings()
    db = get_database()

    ocr = _ocr_health(s)
    items = db.get_items_metadata()

    return jsonify({
        "device": get_device(),
        "pipelines": {
            "remember": "loaded" if _pm._remember_pipeline is not None else "not loaded",
            "scan": "loaded" if _pm._scan_pipeline is not None else "not loaded",
        },
        "ocr": ocr,
        "db": {
            "item_count": len(items),
            "sighting_count": db.count_sightings(),
            "feedback": db.count_feedback(),
        },
        "settings": {
            "enable_depth": s.enable_depth,
            "enable_ocr": s.enable_ocr,
            "enable_dedup": s.enable_dedup,
            "enable_learning": s.enable_learning,
            "similarity_threshold": s.similarity_threshold,
            "darkness_threshold": s.darkness_threshold,
            "blur_sharpness_threshold": s.blur_sharpness_threshold,
            "ocr_backend": s.ocr_backend,
        },
    })


@debug_bp.post("/debug/echo")
def debug_echo():
    """Echo the request back as JSON.

    Sends no headers, runs no inference. Use this when /remember or /scan returns
    an unexpected 400 - hit /debug/echo with the same request to see exactly what
    the server received: field names, file sizes, content-type, etc.
    """
    _SKIP_HEADERS = {"x-api-key", "authorization", "cookie"}
    headers = {k: v for k, v in request.headers if k.lower() not in _SKIP_HEADERS}

    files_info = []
    for field_name, f in request.files.items():
        data = f.read()
        files_info.append({
            "field": field_name,
            "filename": f.filename or "",
            "mimetype": f.mimetype or "",
            "size_bytes": len(data),
        })

    form_fields = {k: v for k, v in request.form.items()}

    json_body = None
    raw_preview = None
    ct = request.content_type or ""
    if "application/json" in ct:
        parsed, err = read_json_dict(request)
        json_body = parsed if err is None else {"error": err[0].get("error"), "detail": err[0].get("detail")}
    elif not request.files:
        raw = request.get_data(as_text=True)
        raw_preview = raw[:500] if raw else None

    return jsonify({
        "method": request.method,
        "content_type": ct,
        "headers": headers,
        "form_fields": form_fields,
        "files": files_info,
        "json_body": json_body,
        "raw_body_preview": raw_preview,
    })


@debug_bp.post("/debug/image")
def debug_image():
    """Upload an image; returns quality metadata without running ML inference.

    Returns the same luminance and blur values the pipeline uses to decide
    whether to reject an image. Use this to verify images are arriving with
    the expected properties before debugging a failed /remember or /scan.
    """
    if "image" not in request.files:
        return jsonify({"error": "missing field: image"}), 400

    raw = request.files["image"].read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        return jsonify({"error": f"could not open image: {exc}"}), 400

    s = get_settings()
    lum = mean_luminance(image)
    blur = _blur_score(image)

    return jsonify({
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "file_bytes": len(raw),
        "luminance": round(lum, 2),
        "is_dark": lum < s.darkness_threshold,
        "darkness_threshold": s.darkness_threshold,
        "blur_score": round(blur, 2),
        "is_blurry": blur < s.blur_sharpness_threshold,
        "blur_sharpness_threshold": s.blur_sharpness_threshold,
    })


@debug_bp.get("/debug/db")
def debug_db():
    """Dump database contents: all items, last 20 sightings, feedback counts."""
    db = get_database()
    return jsonify({
        "item_count": len(db.get_items_metadata()),
        "items": db.get_items_metadata(),
        "sighting_count": db.count_sightings(),
        "recent_sightings": db.get_sightings(limit=20),
        "feedback": db.count_feedback(),
    })


@debug_bp.get("/debug/logs")
def debug_logs():
    """Return recent log entries from app, important, or crash logs.

    Query params:
        n          - number of entries to return (default 50, max 500)
        source     - app | important | crash (default app)
        event      - exact event filter for app/important logs
        level      - exact level filter for app/important logs
        tag        - exact tag filter for app/important logs
        contains   - case-insensitive text filter
        since_line - skip the first N lines before filtering
    """
    n = min(max(request.args.get("n", 50, type=int), 1), 500)
    source = (request.args.get("source") or "app").strip().lower()
    event = request.args.get("event") or None
    level = request.args.get("level") or None
    tag = request.args.get("tag") or None
    contains = request.args.get("contains") or None
    since_line = max(request.args.get("since_line", 0, type=int), 0)

    if source not in {"app", "important", "crash"}:
        return jsonify({"error": "source must be one of: app, important, crash"}), 400

    if source == "crash":
        entries = _tail_crash_blocks(_CRASH_LOG, n=n, contains=contains)
        return jsonify({
            "source": source,
            "path": str(_CRASH_LOG),
            "count": len(entries),
            "filters": {"contains": contains},
            "entries": entries,
        })

    path = _APP_LOG if source == "app" else _IMPORTANT_LOG
    entries = _tail_json_log(
        path,
        n,
        since_line=since_line,
        event=event,
        level=level,
        tag=tag,
        contains=contains,
    )
    return jsonify({
        "source": source,
        "path": str(path),
        "count": len(entries),
        "filters": {
            "event": event,
            "level": level,
            "tag": tag,
            "contains": contains,
            "since_line": since_line,
        },
        "entries": entries,
    })


@debug_bp.get("/debug/perf")
def debug_perf():
    """Return current and recent performance metrics without SSH log parsing."""
    n = min(max(request.args.get("n", 100, type=int), 1), 500)
    app_entries = _tail_json_log(_APP_LOG, n=2000)
    perf_entries = [
        e for e in app_entries
        if e.get("tag") in {"perf", "vram"} or e.get("event") == "vram_layout"
    ]
    recent_perf = perf_entries[-n:]
    important = _tail_json_log(_IMPORTANT_LOG, n=2000)
    warning_count = sum(
        1 for e in important if str(e.get("level", "")).upper() == "WARNING"
    )
    error_count = sum(
        1 for e in important if str(e.get("level", "")).upper() in {"ERROR", "CRITICAL"}
    )
    crash_recent = _tail_crash_blocks(_CRASH_LOG, n=20)
    return jsonify({
        "metrics_now": collect_system_metrics(),
        "summary": _summarize_perf(perf_entries),
        "recent_perf_entries": recent_perf,
        "important_counts": {
            "warning": warning_count,
            "error_or_critical": error_count,
        },
        "crash_count_recent_20": len(crash_recent),
    })


@debug_bp.get("/debug/test-remember")
def debug_test_remember():
    """Run detection on a built-in test image (wallet_1ft_table.jpg). No DB write.

    Proves the remember pipeline (GroundingDINO) is loaded and working.
    Uses detect_score() - returns detection quality without embedding or saving.
    """
    if not _TEST_IMAGE.exists():
        return jsonify({"error": "test image not found", "path": str(_TEST_IMAGE)}), 404

    from visual_memory.api.pipelines import get_remember_pipeline
    pipeline = get_remember_pipeline()
    try:
        result = pipeline.detect_score(_TEST_IMAGE, _TEST_PROMPT)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({
        "image": _TEST_IMAGE.name,
        "prompt": _TEST_PROMPT,
        "detected": result["detected"],
        "score": result.get("score"),
        "blur_score": result.get("blur_score"),
        "second_pass_prompt": result.get("second_pass_prompt"),
    })


@debug_bp.get("/debug/test-scan")
def debug_test_scan():
    """Run scan pipeline on a built-in test image (wallet_1ft_table.jpg). No sightings saved.

    Proves YOLOE + embedding matching are working. If the DB is empty, matches
    will be empty but the pipeline still runs successfully.
    """
    if not _TEST_IMAGE.exists():
        return jsonify({"error": "test image not found", "path": str(_TEST_IMAGE)}), 404

    pipeline = get_scan_pipeline()
    db = get_database()
    try:
        image = Image.open(_TEST_IMAGE).convert("RGB")
        result = pipeline.run(image)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({
        "image": _TEST_IMAGE.name,
        "db_item_count": len(db.get_items_metadata()),
        **result,
    })


@debug_bp.post("/debug/wipe")
def debug_wipe():
    """Selectively wipe database contents and reset weights or settings.

    Body: {"confirm": true, "target": "<target>"}

    target options:
        "all"           - items + sightings + feedback + projection head weights
        "items"         - taught objects only (scan pipeline reloads immediately)
        "sightings"     - location history only
        "feedback"      - feedback records only
        "weights"       - projection head from DB and file; scan pipeline reloads
        "settings"      - ML settings (enable_learning, thresholds) reset to defaults
        "user-settings" - user preferences (performance_mode, voice_speed) reset to defaults
    """
    body, err = read_json_dict(request)
    if err is not None:
        body_out, status = err
        return jsonify(body_out), status
    if not body.get("confirm"):
        return jsonify({"error": 'send {"confirm": true} to wipe data'}), 400

    target = body.get("target", "all")
    valid = {"all", "items", "sightings", "feedback", "weights", "settings", "user-settings"}
    if target not in valid:
        return jsonify({"error": f"unknown target; valid: {sorted(valid)}"}), 400

    db = get_database()
    report: dict = {"wiped": True, "target": target}

    if target in ("all", "items"):
        report["items_deleted"] = db.clear_items()
        get_scan_pipeline().reload_database()

    if target in ("all", "sightings"):
        report["sightings_deleted"] = db.clear_sightings()

    if target in ("all", "feedback"):
        report["feedback_deleted"] = db.clear_feedback()

    if target in ("all", "weights"):
        db.clear_projection_head()
        s = get_settings()
        weights_path = Path(s.projection_head_path)
        if not weights_path.is_absolute():
            weights_path = Path(__file__).resolve().parents[4] / weights_path
        if weights_path.exists():
            weights_path.unlink()
            report["weights_file_deleted"] = True
        else:
            report["weights_file_deleted"] = False
        get_scan_pipeline().reload_head()
        report["weights_db_cleared"] = True

    if target == "settings":
        db.reset_ml_settings()
        defaults = _Settings()
        s = get_settings()
        for key in _ML_PATCHABLE:
            setattr(s, key, getattr(defaults, key))
        get_scan_pipeline().set_enable_learning(s.enable_learning)
        get_scan_pipeline().set_head_weight(s.projection_head_weight, s.projection_head_ramp_at)
        report["ml_settings_reset"] = True

    if target == "user-settings":
        db.reset_user_settings_db()
        _pm._user_settings = None
        report["user_settings_reset"] = True

    return jsonify(report)


@debug_bp.patch("/debug/config")
def debug_patch_config():
    """Directly set any primitive field on the Settings object.

    Changes apply immediately in-memory but are not persisted across restarts.
    For ML learning fields (enable_learning, projection_head_weight, etc.),
    use PATCH /settings instead - that endpoint also propagates the change to
    the scan pipeline and persists it.

    Returns the applied values and a list of fields that would take effect
    immediately vs fields that only matter at model-load time (model paths, etc.).
    """
    data, err = read_json_dict(request)
    if err is not None:
        body, status = err
        return jsonify(body), status
    if not data:
        return jsonify({
            "settable_fields": {k: t.__name__ for k, t in _SETTABLE_TYPES.items()}
        })

    persist = bool(data.pop("persist", False))

    errors = {}
    applied = {}
    for key, value in data.items():
        if key not in _SETTABLE_TYPES:
            errors[key] = "unknown or non-primitive field"
            continue
        expected = _SETTABLE_TYPES[key]
        try:
            if expected is bool:
                if isinstance(value, bool):
                    cast = value
                elif isinstance(value, str):
                    cast = value.lower() in ("1", "true", "yes")
                else:
                    cast = bool(value)
            else:
                cast = expected(value)
            applied[key] = cast
        except (TypeError, ValueError):
            errors[key] = f"expected {expected.__name__}"

    if errors:
        return jsonify({"errors": errors}), 400

    s = get_settings()
    for key, value in applied.items():
        setattr(s, key, value)

    # Propagate ML learning fields to pipeline so they take effect immediately
    pipeline = get_scan_pipeline()
    if "enable_learning" in applied:
        pipeline.set_enable_learning(s.enable_learning)
    if "projection_head_weight" in applied or "projection_head_ramp_at" in applied:
        pipeline.set_head_weight(s.projection_head_weight, s.projection_head_ramp_at)

    if persist:
        import dataclasses as _dc
        snapshot = {
            f.name: getattr(s, f.name)
            for f in _dc.fields(s)
            if isinstance(getattr(s, f.name), (bool, int, float, str))
        }
        get_database().save_ml_settings(snapshot)

    return jsonify({
        "applied": applied,
        "persisted": persist,
        "note": "in-memory only unless persist=true. For ML fields use PATCH /settings.",
    })
