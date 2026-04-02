# VisualMemory Backend Architecture

> Reference documentation.
> App context: Navigation aid for blind users. Announces nearby significant objects with distance + direction.

---

## Project Overview

Blind users cannot quickly visually re-scan their environment the way sighted people naturally do. This makes spatial recall slow and cognitively draining.

Spaitra restores this ability by creating a personal, spatial memory of the user’s physical world.


## Core Actions

### Teach 

> “These are my house keys on the dining room table.”

Single-shot memory creation.

- Detect object
- Generate visual + text embeddings
- Store spatial position
- Add to personal memory index

The user can confirm or correct detections to improve personalization.

---

### Scan 

Recognizes previously taught objects in the current scene.

Example narration:
> “Wallet 3 feet to your left. House keys look down, ahead. Tax returns 3 feet ahead.”

- Objects are ordered spatially (left -> right)
- Results returned as structured JSON

---

### Ask 

> “Where did I leave my wallet?”  
> “What’s the receipt that has my office chair on it?”

Semantic memory retrieval.

- Search stored embeddings
- Match object or document
- Return last known spatial location
- Optionally read document content aloud or export

---

### Main Differentiators
- Embedding-based semantic search across personal memory
- Spatial grounding (distance + direction of important objects)
- Personalized detection head (ProjectionHead on DINOv3) to improve accuracy with use
- Continuous adaptation to the user’s environment
- Minimalist, voice-based interface

---

## UX Design

The interface is voice-first. One core action per screen. There is no tutorial on first launch; onboarding is the user's first Teach + Scan + Ask session with actual items in a real room. They understand the product because they used it, not because they read instructions.

Feedback model decisions that affect the backend:
- Only explicit negatives are logged as training data. The user says "wrong" to flag a bad detection. Silence is not recorded.
- No feedback is collected during onboarding. The ProjectionHead does not train until the user has organic sessions with clean, representative data.
- Unconfirmed positives are excluded because they can corrupt triplet training over time.

See [UX.md](UX.md) for the full onboarding flow and UX patterns.

### Accessibility Priorities

- Blind-first interaction quality takes precedence over visual polish.
- Large, forgiving touch targets and clear spacing matter more than color, animation, or other visual state changes.
- "Visual differentiation" is secondary only for low-vision users, sighted debugging, and demos. It is not the primary accessibility mechanism.
- Primary state confirmation should come from speech, listening prompts, haptics where supported, and fixed button placement the user can memorize.
- For the mobile UI, ease of pressing the `Spaitra`, `Settings`, and `Scan` controls is a release blocker. Small icons, dense toolbars, and layout churn are not acceptable for production.

---

## Setup

### Local Development

```bash
gh repo clone tsa-softwaredev-26/TSA-soft-dev-backend-2026
cd TSA-soft-dev-backend-2026
python -m venv .venv-core
source .venv-core/bin/activate
pip install -e ".[core]"
hf auth login
python setup_weights.py
```

OCR runs in a separate environment:

```bash
python -m venv .venv-ocr
source .venv-ocr/bin/activate
pip install -e ".[ocr]"
python -m services.ocr.run
```

Run the core backend from the core environment:

```bash
source .venv-core/bin/activate
python -m services.core.run
```

`pip install -e ".[core]"` installs Torch, depth-pro, Flask, and the core model stack.
`pip install -e ".[ocr]"` installs PaddleOCR and the OCR service stack only.
`setup_weights.py` downloads Depth Pro (~2GB), YOLOE (~80MB), CLIP (~180MB),
DINOv3 (~1.2GB), and GroundingDINO (~900MB). DINOv3 and GroundingDINO are gated
on HuggingFace - request access and run `hf auth login` first.
`DepthEstimator` resolves the checkpoint path at import time using `Path(__file__)`.

### Server Deployment

See [DEPLOY.md](DEPLOY.md) for the full server setup guide (Debian, systemd,
gunicorn, GPU PyTorch, external OCR service, srv.us tunnel).

---

## Core Structure

```
services/
├── core/
│   ├── app.py                        # Core app surface
│   ├── run.py                        # Dev entry point (python -m services.core.run)
│   ├── wsgi.py                       # Gunicorn entry point (services.core.wsgi:application)
│   └── requirements.txt              # Core-only install surface
├── ocr/
│   ├── app.py                        # FastAPI PaddleOCR service
│   ├── run.py                        # Dev entry point (python -m services.ocr.run)
│   └── requirements.txt              # OCR-only install surface
│
src/visual_memory/
├── pipelines/
│   ├── remember_mode/pipeline.py
│   └── scan_mode/pipeline.py
│
├── engine/
│   ├── model_registry.py               # ModelRegistry singleton; lazy-loads all models; prepare_for_remember() / prepare_for_scan() for VRAM swapping
│   ├── embedding/
│   │   ├── embed_image.py              # ImageEmbedder (DINOv3, 1024-dim)
│   │   ├── embed_text.py              # CLIPTextEmbedder (CLIP text encoder, 512-dim)
│   │   └── embed_combined.py          # make_combined_embedding() - 1536-dim output
│   ├── speech_recognition/
│   │   ├── base.py                    # BaseSpeechRecognizer ABC
│   │   └── whisper_recognizer.py      # WhisperRecognizer (active STT)
│   ├── text_recognition/
│   │   ├── base.py                    # BaseTextRecognizer ABC
│   │   └── http_recognizer.py         # HTTPOCRRecognizer (active)
│   ├── object_detection/
│   │   ├── detect_all.py              # YOLOE
│   │   └── prompt_based.py            # GroundingDINO
│   └── depth/estimator.py             # Depth Pro wrapper
│
├── utils/
│   ├── audio_utils.py                 # audio validation + decode/resample helpers for STT
│   ├── image_utils.py
│   ├── similarity_utils.py
│   ├── logger.py                      # JSON structured logging via loguru; LogTag constants; crash handler
│   ├── metrics.py                     # collect_system_metrics() - RAM/swap/VRAM/thermals
│   ├── logparse.py                    # CLI filter + stats tool (python -m visual_memory.utils.logparse)
│   ├── voice_state_context.py         # mode-aware context contract for STT / routing
│   └── voice_state_policy.py          # state-aware speech bias policy
│
├── learning/
│   ├── projection_head.py             # ProjectionHead (residual linear, identity at init)
│   ├── trainer.py                     # ProjectionTrainer + triplet_loss (CLI: python -m visual_memory.learning.trainer)
│   └── feedback_store.py              # FeedbackStore (SQLite-backed via DatabaseStore.feedback table)
│
├── benchmarks/
│   ├── full_benchmark.py              # multi-stage system benchmark (retrieval, detection, depth)
│   ├── format_results.py              # reads results.csv (+ optional results.json metadata), writes BENCHMARKS.md
│   ├── check_dataset.py               # pre-flight: verify all 120 images are in benchmarks/images/
│   └── redact_receipt.py              # gitignored - OCR-based receipt redaction (personal tool)
│
├── config/
│   ├── settings.py                     # All ML tuning thresholds
│   └── user_settings.py                # User preferences: PerformanceMode, UserSettings (JSON-persisted)
├── api/
│   ├── app.py                          # create_app() factory - blueprints, auth, upload cap
│   ├── pipelines.py                    # Lazy singletons: get_remember_pipeline(), get_scan_pipeline(), get_feedback_store(), get_user_settings(), warm_all()
│   ├── run.py                          # Compatibility shim; prefer services/core/run.py
│   ├── voice_session.py                # Per-WebSocket state machine state
│   └── routes/
│       ├── health.py                   # GET /health
│       ├── remember.py                 # POST /remember
│       ├── scan.py                     # POST /scan
│       ├── feedback.py                 # POST /feedback
│       ├── retrain.py                  # POST /retrain
│       ├── settings_route.py           # GET /settings, PATCH /settings (ML tuning)
│       ├── user_settings_route.py      # GET /user-settings, PATCH /user-settings (user prefs)
│       ├── find.py                     # GET /find - last-seen location query (Ask Mode)
│       ├── items.py                    # GET /items, DELETE /items/<label>, POST /items/<label>/rename
│       ├── sightings.py               # POST /sightings - user-confirmed location update
│       ├── crop.py                    # GET /crop - fetch cropped image from cached scan
│       ├── transcribe.py              # POST /transcribe + shared STT helper
│       ├── voice.py                   # Unified voice HTTP route
│       ├── voice_ws.py                # Stateful Socket.IO voice transport
│       └── debug.py                   # Optional debug endpoints; enabled only when ENABLE_DEBUG_ROUTES=1
└── tests/                             # Test scripts + test data
    ├── scripts/                        # Runnable .py test scripts
    ├── input_images/                   # Object test images
    ├── demo_database/                  # Reference crops for ScanPipeline
    └── text_demo/                      # OCR test images + ground_truth/
```

---

## Voice and STT Architecture

### Current production path

Frontend voice capture belongs in the frontend app only:
- `composeApp/src/commonMain/kotlin/.../ui/HomeScreen.kt` - hold/release button behavior
- `composeApp/src/commonMain/kotlin/.../ui/AppViewModel.kt` - UI intent -> socket event bridge
- `composeApp/src/androidMain/kotlin/.../platform/PlatformVoiceController.android.kt`
- `composeApp/src/iosMain/kotlin/.../platform/PlatformVoiceController.ios.kt`

Speech-to-text itself belongs in the backend, not the mobile client:
- `src/visual_memory/utils/audio_utils.py` - validate, decode, and resample incoming audio bytes
- `src/visual_memory/api/routes/transcribe.py` - shared STT entrypoint (`transcribe_audio_bytes`)
- `src/visual_memory/engine/speech_recognition/whisper_recognizer.py` - Whisper model load, context prompt injection, and transcription
- `src/visual_memory/engine/model_registry.py` - lifecycle and warm/cold preparation for Whisper
- `src/visual_memory/api/routes/voice_ws.py` - WebSocket event handling (`chat_start`, `audio`, `navigate`, `shortcut_*`) and stateful routing after STT
- `src/visual_memory/api/voice_session.py` - per-connection runtime state for mode-aware voice behavior
- `src/visual_memory/utils/voice_state_context.py` and `src/visual_memory/utils/voice_state_policy.py` - state-aware context biasing for Whisper
- `src/visual_memory/config/settings.py` - STT model, language, sample rate, and context-bias settings

### STT responsibility split

- Frontend responsibility: capture audio, stop/release correctly, send bytes, play TTS, expose large reliable controls.
- Backend responsibility: decode audio, transcribe, apply state-aware biasing, route intent, return narration and authoritative state.
- Production rule: do not move STT into the mobile client for v1. Keep one backend STT implementation so behavior, context biasing, and evaluation stay consistent across Android and iOS.

---

## VRAM Management (`SAVE_VRAM`)

`SAVE_VRAM` controls model parking between GPU and CPU RAM.

- Wrappers (`ImageEmbedder`, `CLIPTextEmbedder`, `GroundingDinoDetector`, `YoloeDetector`, `DepthEstimator`) support `to_cpu()` / `to_gpu()` so weights can move without reloading from disk.
- `ModelRegistry` swaps layouts at the start of pipeline `run()` calls:

| Method | GPU (active) | CPU (parked) |
|---|---|---|
| `prepare_for_remember()` | GDino + DINOv3 + CLIP | YOLOE + Depth Pro |
| `prepare_for_scan()` | YOLOE + Depth Pro + DINOv3 + CLIP | GDino |

- DINOv3 and CLIP stay on GPU in both modes (~480 MB combined).
- Use `SAVE_VRAM=1` for GPUs with `< 8 GB` VRAM (needs ~16 GB system RAM; swap cost: GDino ~1-2 s, Depth Pro ~3-5 s).
- Keep `SAVE_VRAM=0` on GPUs with `>= 8 GB` VRAM to avoid per-request swap overhead.
- `warm_all()` starts in scan layout so first scan has no swap penalty.

---

## Module Reference

### `pipelines/remember_mode/pipeline.py` - `RememberPipeline`
- `run(image_path, prompt) -> dict` - full pipeline: detect -> embed -> OCR -> DB write
- `detect_score(image_path, prompt) -> dict` - detection only, no embed or DB write; used by multi-image route to rank candidates
- `_detect_with_fallback(image, prompt)` - returns `(detection, second_pass_prompt)`: tries original prompt first, then `_SECOND_PASS_TEMPLATES` if nothing found and `detection_second_pass_enabled=True`
- `add_to_database(embedding, metadata)` - persists combined embedding to SQLite via DatabaseStore

`run()` result shape:
```json
{
  "success": true,
  "message": "...",
  "result": {
    "label": "wallet",
    "confidence": 0.52,
    "detection_quality": "low | medium | high",
    "detection_hint": "human-readable guidance string",
    "blur_score": 143.7,
    "is_blurry": false,
    "is_dark": false,
    "darkness_level": 112.4,
    "replaced_previous": false,
    "second_pass": false,
    "second_pass_prompt": null,
    "box": [x1, y1, x2, y2],
    "ocr_text": "",
    "ocr_confidence": 0.0
  }
}
```

Teach behavior:
- Label is always the user's original prompt string, not the GDINO-internal token (matters when second-pass prompts like "a wallet" are used).
- Auto-replace: if any items with this label already exist, they are deleted before the new embedding is stored. No confirmation. `replaced_previous: true` in the result lets the frontend say "Updated existing memory" instead of "New memory saved".
- Historical avg confidence is captured before the delete, so the quality tier is still informed by past teaches.

Detection quality tiers:
- When the label has prior teaches: score is normalized by historical avg confidence for that label, so inherently hard-to-detect labels are not penalized vs their baseline.
- Without history: absolute thresholds (`detection_quality_low_max`, `detection_quality_high_min`) apply.
- Blur (`is_blurry`) is determined by Laplacian variance of the full image vs `blur_sharpness_threshold` and is used for quality messaging (it does not block storing a detection).
- `detection_hint` is a function of both quality tier and blur, giving the most actionable user-facing message.

Second-pass detection:
- Triggers only when the first `detector.detect()` call returns None.
- Tries templates in `_SECOND_PASS_TEMPLATES`: `"a {prompt}"`, `"{prompt} object"`, `"close up of a {prompt}"`.
- Result includes `second_pass: true` and `second_pass_prompt` so the client can log or surface it.
- If all second-pass templates also fail, remember mode runs a third pass via local Ollama prompt suggestions (`_detect_with_ollama_fallback`) and retries with those variants.

Multi-image (POST /remember with `images[]`):
- Frontend sends N frames (burst or sequential), backend runs `detect_score` on all, picks highest confidence, runs full pipeline only on the winner.
- `images_tried` and `images_with_detection` added to response when multi-image path is taken.

### `pipelines/scan_mode/pipeline.py` - `ScanPipeline`
- `__init__(focal_length_px: float, db_path: str | Path = None)` - loads models + fetches DB embeddings on init
- `run(query_image: PIL.Image, scan_id: str | None = None, focal_length_px: float | None = None) -> dict`
- `scan_id`: if provided, caches (anchor_emb, query_emb) per label for /feedback lookup
- `focal_length_px`: per-call override; falls back to `self.focal_length_px` if None
- Returns `{"matches": [...], "count": int}` (plus `is_dark`, `darkness_level`, `message` on dark images)
- Each match (depth enabled): `{"label": str, "similarity": float, "distance_ft": float, "direction": str, "narration": str, "ocr_text": str (optional)}`
- Each match (depth disabled): `{"label": str, "similarity": float, "box": list, "ocr_text": str (optional)}`
- Auto-sightings: every successful match calls `db.add_sighting(label, direction, similarity)` so `/find` works immediately without user confirmation. User-confirmed POST /sightings adds `room_name` and spatial data on top.
- `reload_database()` - re-fetches all items from DB; call after /remember adds an entry
- `_load_head() -> bool` - loads projection head: DB first (`user_state.projection_head`), file fallback; used by `__init__` and `reload_head()`
- `reload_head() -> bool` - re-reads projection head (DB first, file fallback) after /retrain; returns True if loaded
- `set_enable_learning(bool)` - toggle projection head application at runtime (called by PATCH /settings)
- `set_head_weight(weight, ramp_at)` - update blend ceiling and ramp target at runtime
- `get_cached_embeddings(scan_id, label) -> (anchor_emb, query_emb) | None`
- `_apply_head(emb) -> emb` - blends raw and projected embedding; alpha ramps 0->weight as `_triplet_count` reaches `_head_ramp_at`
- `_triplet_count` - set by /retrain after training; drives automatic weight scaling
- Database raw base embeddings stored; ProjectionHead applied on-the-fly at match time; DB projected once per `run()` call (outside box loop)
- `_head_trained=False` (no weights file) -> `_apply_head` is a no-op, zero pipeline cost

### `config/user_settings.py` - `UserSettings`
- User-facing preferences, separate from ML tuning params in settings.py
- Persisted as JSON text in `user_state.user_settings` (SQLite, same DB as items)
- Loaded once at process start via `get_user_settings()` which calls `get_database()`
- `PerformanceMode`: FAST | BALANCED | ACCURATE (string enum)
- `PerformanceConfig.for_mode(mode)` - returns `{depth_enabled, target_latency}` for each mode
- `UserSettings` fields: `performance_mode` (BALANCED), `voice_speed` (1.0, range 0.5-2.0), `learning_enabled` (True), `button_layout` ("default" | "swapped", stub)
- `save(db)` / `load(db)` - DB round-trip with safe defaults on missing/corrupt row
- `to_dict()` - serializes for API response; includes derived `performance_config` block
- Exposed via GET /user-settings, PATCH /user-settings
- PATCH /user-settings propagates `learning_enabled` to ScanPipeline and `performance_mode: fast` disables detection second pass

### `config/settings.py` - `Settings`
- Single dataclass, all ML tuning params in one place
- Detection: `grounding_dino_model`, `box_threshold`, `text_threshold`, `yoloe_confidence`, `yoloe_iou`
- Embedding: `image_embedder_model` (`dinov3-vitl16`), `embedder_model` (`clip-vit-base-patch32`)
- Matching: `similarity_threshold` (0.14), `dedup_iou_threshold` (0.5), `narration_high_confidence` (0.6)
- OCR: `ocr_backend` ("http"), `ocr_service_url`, `ocr_health_url` (derived from ocr_service_url if empty), `ocr_timeout_seconds`, `ocr_languages`, `ocr_min_confidence` (0.3), `text_similarity_threshold` (0.4)
- Toggles (env-overridable): `enable_depth` (`ENABLE_DEPTH`), `enable_ocr` (`ENABLE_OCR`), `enable_dedup` (`ENABLE_DEDUP`)
- Personalization: `projection_head_path` ("models/projection_head.pt"), `projection_head_dim` (1536)
- Learning: `enable_learning` (`ENABLE_LEARNING`), `min_feedback_for_training` (10), `projection_head_weight` (1.0), `projection_head_ramp_at` (50), `projection_head_epochs` (20)
- Detection quality (remember mode): `detection_quality_low_max` (0.40), `detection_quality_high_min` (0.65)
- Image sharpness: `blur_sharpness_threshold` (120.0) - Laplacian variance below this = blurry
- Second pass: `detection_second_pass_enabled` (True) - retry with reformulated prompts on failed detect
- Darkness: `darkness_threshold` (30.0) - mean luminance below this = too dark; both modes; conservative, only genuine dark rooms (no lights on)
- API: `api_host` ("127.0.0.1"), `api_port` (5000)

### `engine/object_detection/prompt_based.py` - `GroundingDinoDetector`
- Model: `IDEA-Research/grounding-dino-base`
- `detect(image, prompt)` - returns `{'box', 'score', 'label'}` or None
- `detect_batch(images, prompts)` - single forward pass; returns `List[Optional[dict]]`
- Used in: Remember Mode

### `engine/object_detection/detect_all.py` - `YoloeDetector`
- Model: `yoloe-26l-seg-pf.pt` (local, prompt-free); device auto-selected by ultralytics
- `detect_all(image)` - returns `(List[[x1,y1,x2,y2]], List[float])` or `(None, None)`
- `detect_all_batch(images)` - single forward pass; returns `List[(boxes, scores)]`
- Used in: Scan Mode

### `engine/embedding/embed_image.py` - `ImageEmbedder`
- Model: `facebook/dinov3-vitl16-pretrain-lvd1689m` (gated, ~1GB, requires `transformers>=4.56.0`)
- `embed(image: PIL.Image) -> torch.Tensor` shape `(1, 1024)` - L2-normalized from `pooler_output`
- `batch_embed(images: List[PIL.Image]) -> torch.Tensor` shape `(N, 1024)` - single forward pass
- Self-supervised vision model; object-centric features with less context bleed than CLIP
- Used in: Both modes

### `engine/embedding/embed_text.py` - `CLIPTextEmbedder`
- Model: `openai/clip-vit-base-patch32` text encoder only (`CLIPTextModelWithProjection`, ~180MB)
- `embed_text(text: str) -> torch.Tensor` shape `(1, 512)` - L2-normalized; chunked for long documents
- `batch_embed_text(texts: List[str]) -> torch.Tensor` shape `(N, 512)` - single forward pass; truncates at 77 tokens
- Used in: Both modes (OCR text embedding)

### `engine/embedding/embed_combined.py` - `make_combined_embedding`
- Concatenates normalized image + text embeddings -> `(1, 1536)` combined vector
- Non-document objects pass a zero text vector - image similarity dominates
- Used in: Both pipelines

### `engine/text_recognition/http_recognizer.py` - `HTTPOCRRecognizer`
- Backend: external HTTP OCR microservice
- Input: PIL Image (encoded as PNG multipart form-data, field name: `image`)
- Output: `{"text": str, "confidence": float, "segments": list}`
- Response contract: JSON with at least `text` (string), optional `confidence` (float)

### `learning/projection_head.py` - `ProjectionHead`
- Single residual linear layer (`Linear(1536, 1536, bias=False)`) initialized to zeros
- At init: `linear(x) = 0`, so `forward(x) = normalize(x + 0) = x` - identity transform
- Safe to enable by default before any training; no-op until weights file exists
- `load(path) -> bool` - loads weights if file exists, returns False if not
- `save(path)` - saves state dict; creates parent dirs
- `project(x)` - alias for `forward()`

### `learning/trainer.py` - `ProjectionTrainer` + `triplet_loss`
- `triplet_loss(anchor, positive, negative, margin=0.2)` - cosine distance, `relu(pos_dist - neg_dist + margin).mean()`
- `ProjectionTrainer(head, lr=1e-4)` - Adam optimizer with `weight_decay=1e-4`
- `train_step(anchor, positive, negative, weight=1.0) -> float` - single gradient step
- `train(triplets, epochs=20) -> float` - applies linear recency bias (oldest=0.5, newest=1.0); returns avg final loss
- CLI: `python -m visual_memory.learning.trainer [--db-path data/memory.db] [--output models/projection_head.pt] [--epochs 20] [--lr 1e-4] [--dim 1536]`
- `--feedback-dir` is accepted as a deprecated alias for `--db-path` for backwards compat; logs a warning

### `learning/feedback_store.py` - `FeedbackStore`
- `FeedbackStore(db: DatabaseStore)` - thin wrapper; all persistence delegated to `DatabaseStore.feedback` table
- `record_positive(anchor_emb, query_emb, label)` - calls `db.add_feedback(label, "positive", ...)`
- `record_negative(anchor_emb, query_emb, label)` - calls `db.add_feedback(label, "negative", ...)`
- `load_triplets() -> List[(anchor, positive, negative)]` - calls `db.load_feedback_triplets()`; pairs each positive with each negative per label (cartesian product)
- `count() -> dict` - calls `db.count_feedback()`; `{"positives": int, "negatives": int, "triplets": int}`
- Triplet count is per-label cartesian product, computed in SQL via GROUP BY to avoid loading all blobs

### `utils/device_utils.py`
- `get_device() -> str` - returns "cuda", "mps", or "cpu" in priority order; used by all engine modules

### `utils/image_utils.py`
- `load_image(path)` - PIL Image (RGB, EXIF-corrected, HEIC-compatible)
- `load_folder_images(folder)` - `List[(path, PIL Image)]`
- `crop_object(image, box)` - cropped PIL Image

### `utils/quality_utils.py`
- `mean_luminance(image: PIL.Image) -> float` - mean grayscale pixel value (0-255); used by both pipelines for darkness detection
- `estimate_text_likelihood(image: PIL.Image) -> float` - normalized Laplacian edge density on a 128x128 greyscale resize; returns 0.0-1.0; ~2ms; used as OCR pre-check to skip HTTP OCR calls on plain-color crops (threshold `ocr_text_likelihood_threshold=0.30` in settings.py)
- Pure numpy/PIL - no model loading, no extra deps
- Shared between RememberPipeline and ScanPipeline via `utils` package export

### `utils/similarity_utils.py`
- `cosine_similarity(t1, t2)` - scalar tensor
- `find_match(query, database, threshold)` - `(path, score)` or `(None, 0.0)`
- `deduplicate_matches(matches, iou_threshold)` - filtered list

### `utils/memory_monitor.py`
- Proactive OOM prevention helper for RAM, swap, and VRAM pressure checks
- `check_memory()` returns usage percentages and MB totals
- `is_oom_risk(threshold=0.85)` checks combined RAM+swap pressure
- `cleanup_zombies(max_age_hours=2)` removes old zombie processes owned by the current user
- `suggest_throttle()` recommends slowdowns under memory pressure
- Integrations: benchmark stage guards in `benchmarks/full_benchmark.py` and request middleware in `api/app.py`
- CLI: `python -m visual_memory.utils.memory_monitor --check` and `--cleanup`

### `engine/depth/estimator.py` - `DepthEstimator`
- Model: Apple Depth Pro
- `__init__(focal_length_px=None)` -loads model once
- `estimate(image: PIL.Image) -> torch.Tensor` -depth map in meters
- `get_depth_at_bbox(depth_map, bbox) -> float` -mean depth in feet, inner 50% of bbox
- `get_direction(bbox, img_w) -> str` -5-zone direction string
- `build_narration(label, direction, distance_ft, similarity, bbox=None, img_h=None) -> str | None` -final narration; prepends "look down," or "look up," when object is in the bottom 40% or top 25% of frame

---

## Pipeline Flows

### Remember Mode

- Load image, then run darkness and blur checks.
- Run GroundingDINO detection (batch path when available).
- If initial detection fails, retry with `_SECOND_PASS_TEMPLATES`; if still no hit, return failure with quality metadata.
- Crop the detected object and compute historical confidence baseline for the label.
- Run text-likelihood gating; call OCR only when the crop looks text-bearing.
- Build combined embedding (`DINOv3 1024 + CLIP text 512 = 1536`).
- Write to DB and return success plus quality tier/hint.
- Multi-image `/remember` ranks candidates with `detect_score_batch`, then runs full pipeline once on the best image.

### Scan Mode

- Init loads DB embeddings and projection head state (DB first, file fallback).
- Run darkness check; return early for dark frames.
- Detect candidate boxes with YOLOE.
- Pass 1: crop each box, embed, optionally project via `ProjectionHead`, and score with `find_match()`.
- Deduplicate overlapping/duplicate matches.
- Pass 2 (only if matches exist): run depth once, then annotate each match with distance, direction, and narration.
- Return `{"matches": [...], "count": int}`.

---

## Depth Estimation

### Model: Apple Depth Pro
- Metric depth (absolute meters)
- `f_px=None` -> model infers (~75% error at close range)
- `f_px=tensor` -> calibrated (~26% error) 
- Checkpoint: `checkpoints/depth_pro.pt` -absolute path resolved via `Path(__file__)` in estimator.py

### Focal Length
```
f_px = (focalLengthMm / sensorWidthMm) * imageWidthPx
```
Pass from Android camera API. iPhone 15 Plus reference: `f_mm=6.24`, `sensor_width=8.64mm` -> `f_px=3094.0`

### Test Results (18 images, iPhone 15 Plus)
- Average error: 27.8%, acceptable for navigation

---

## Benchmarks

Full system evaluation across retrieval, detection, and depth estimation.

### Dataset

```
benchmarks/dataset.csv      -- 120-row dataset (committed)
benchmarks/images/          -- user-captured images (gitignored)
benchmarks/ground_truth/    -- receipt OCR ground truth from redact_receipt.py (gitignored)
benchmarks/results.csv      -- generated output (gitignored)
benchmarks/results.json     -- generated output with metadata (gitignored)
BENCHMARKS.md               -- formatted report at project root (gitignored)
benchmarks/CAPTURE_GUIDE.md -- personal capture + run guide (gitignored)
```

**10 objects x 12 conditions = 120 images.**
Conditions: 3 distances (1ft/3ft/6ft) x 2 lighting (bright/dim) x 2 backgrounds (clean/messy).
Objects: wallet_a, wallet_b, bottle_a, bottle_b, sunglasses_a, sunglasses_b, receipt_a, receipt_b, keys_a, keys_b.
Labels are per-instance (wallet_a vs wallet_b) to test instance-level discrimination.

### Stages

1. **Embed** - DINOv3 image embedding + optional OCR-service text embedding for each image.
2. **Split** - 60/40 train/test per label (seeded RNG).
3. **Database** - Individual train embeddings, identical format to production ScanPipeline.
4. **Train** - ProjectionTrainer generates triplets from train set; trains ProjectionHead.
5. **Retrieval** - `find_match()` with production threshold on baseline and personalized embeddings.
6. **Detection** - GroundingDinoDetector on each test image with the object's dino_prompt.
7. **Depth** - DepthEstimator on images where GDINO detected the object and gt_distance > 0.

### Scripts

```bash
# Validate images before running
python -m visual_memory.benchmarks.check_dataset

# Redact receipts (once per receipt, interactive)
python -m visual_memory.benchmarks.redact_receipt a
python -m visual_memory.benchmarks.redact_receipt b

# Fast smoke test (~5-10 min)
python -m visual_memory.benchmarks.full_benchmark \
    --dataset benchmarks/dataset.csv --images benchmarks/images \
    --seed 42 --no-depth --no-ocr --epochs 5

# Full run (~2-4 hours)
python -m visual_memory.benchmarks.full_benchmark \
    --dataset benchmarks/dataset.csv --images benchmarks/images --seed 42

# Format into BENCHMARKS.md
python -m visual_memory.benchmarks.format_results
```

---

## Tests

Run from project root:

```bash
# Fast API + unit coverage (default suite)
python -m visual_memory.tests.scripts.run_all

# Integration/system checks (real models; optional depth)
python -m visual_memory.tests.scripts.run_tests
DEPTH=1 python -m visual_memory.tests.scripts.run_tests
```

---

## API

Run locally: `python -m services.core.run`
Production (Ubuntu/NVIDIA): `gunicorn -w 1 -b 0.0.0.0:5000 "visual_memory.api.app:create_app()"`
Single worker is required - model state is process-local.

**Environment variables:**
- `API_KEY=<secret>` - enforce `X-API-Key` header on all routes except `/health` (required for production, minimum length enforced)
- `DB_ENCRYPTION_KEY=<secret>` - enable SQLCipher encryption for `data/memory.db` (current production runtime is Python 3.12 with SQLCipher enabled)
- `ENABLE_DEPTH=0` - skip Depth Pro (~2GB VRAM savings)
- `ENABLE_OCR=0` - skip OCR requests
- `OCR_SERVICE_URL=http://127.0.0.1:8001/ocr` - external OCR endpoint
- `OCR_HEALTH_URL=http://127.0.0.1:8001/health` - OCR service health check URL (derived from OCR_SERVICE_URL if unset)
- `OCR_MAX_CONCURRENCY=1` - OCR request capacity guardrail (over-capacity returns 429 `server_busy`)
- `OCR_RATE_LIMIT_PER_MIN=120` - per-client OCR rate limit (over-limit returns 429 `rate_limited`)
- `OCR_THROTTLE_RETRY_AFTER_SECONDS=2` - retry guidance for throttled OCR calls
- `ENABLE_LEARNING=0` - disable projection head application in scan (also patchable at runtime via PATCH /settings)
- `api_host` / `api_port` controlled via `settings.py` (defaults: 127.0.0.1:5000)

Models are loaded once at startup via `warm_all()` in `create_app()`. Upload cap: 50MB.

**Endpoints:** GET /health, POST /remember, POST /scan, POST /feedback, POST /retrain, GET /retrain/status, GET /settings, PATCH /settings, GET /user-settings, PATCH /user-settings, GET /find, POST /ask, POST /item/ask, GET /crop, GET /items, DELETE /items/<label>, POST /items/<label>/rename, POST /sightings. Optional debug endpoints are available only when `ENABLE_DEBUG_ROUTES=1`.

See `UX.md` for current onboarding narration and state-driven UX behavior.

---

## Current Capabilities

- Production API supports teach (`/remember`), scan (`/scan`), ask (`/ask`, `/item/ask`), find (`/find`), feedback/retrain, item management, settings, and voice endpoints.
- Core stack is in place: GroundingDINO (teach), YOLOE (scan), DINOv3+CLIP combined embeddings, OCR microservice integration, and optional Depth Pro distance estimation.
- Scan uses optional personalization via ProjectionHead; training and feedback are persisted in SQLite.
- Runtime/user settings are patchable via API and persist across restarts.
- Voice path is backend-authoritative with state-aware routing and Whisper-based transcription.
- Deploy path is systemd-based (`spaitra-core` + `spaitra-ocr`) with model warmup at startup.

---
## Production Priorities

- Keep benchmark harness behavior aligned with runtime paths and thresholds before tuning releases.
- Finish frontend blind-first polish (stable controls, haptics, clear spoken state guidance, handedness layout).
- Harden operational readiness: monitoring/alerts, backup and restore procedures, secret rotation, and rollback checklist.
- Lock release gates on measured quality (voice, OCR, retrieval, depth) and accessibility pass results.

---

## Design Decisions

- **Embedder: DINOv3 for images, CLIP text encoder for OCR (March 2026)** - Started with DINOv3 (1024-dim) + sentence-transformers (384-dim). Tried switching to CLIP ViT-B/32 alone (512-dim shared image+text space) to simplify the stack and unify embedding spaces. Reverted image embeddings back to DINOv3 after benchmarking: CLIP image intra/inter-class ratio 0.961 (intra=0.769, inter=0.800) on full-scene images - objects on the same table appeared more similar cross-class than the same object at different distances. DINOv3 is self-supervised and object-centric, producing better visual discrimination for the scan use case. CLIP text encoder kept for OCR matching - lightweight text-only projection (~180MB), good semantic alignment for product labels. sentence-transformers removed entirely. Combined vector: 1536-dim (DINOv3 1024 + CLIP text 512). DINOv3 is gated on HuggingFace - requires access request.
- **YOLOE prompt-free for scan** - Broad detection; similarity threshold handles bloat.
- **Grounding DINO for remember** - Handles vague natural language; much stronger than prompted YOLOE for this use case.
- **Depth Pro** - Only monocular model with metric (absolute) depth; no scale factor needed.
- **OCR moved out of core backend (March 2026)** - OCR is now an external HTTP microservice so backend deployment avoids Paddle dependency conflicts. The core backend sends crop images to `OCR_SERVICE_URL` (`/ocr`) and consumes a stable JSON response (`text`, `confidence`). `/ocr/batch` was tested and removed after reproducible p95 regressions on server benchmarks.
- **Combined embedding over hybrid matching** - `max(image_sim, text_sim)` could false-match two white documents with different text (image similarity alone passes threshold). The combined embedding `normalize(img) || normalize(text)` means cosine similarity is approximately the average of both sub-similarities, so both image AND text must match. Non-document objects get a zero text slot; image similarity dominates unchanged.

---

## Tuning Process

### How tuning is run

When tuning ML/runtime parameters on single-GPU systems:

1. Stop live services: `systemctl stop spaitra-core spaitra-ocr`
2. Clear GPU memory: `nvidia-smi --query-gpu=memory.free --format=csv`
3. Run tuning workloads in isolation (no concurrent load)
4. Monitor GPU utilization and VRAM: `nvidia-smi dmon`
5. Validate stability before restart: `systemctl start spaitra-core spaitra-ocr && sleep 5 && systemctl status spaitra-core spaitra-ocr`

Before merging tuning changes:

1. Run benchmark/test coverage for the affected area on server
2. Check for regressions in latency and accuracy
3. Persist and verify settings through `PATCH /settings` + restart (`warm_all()` reapplies)
4. Commit benchmark artifacts/settings snapshots under `benchmarks/` when generated

### Tuned parameters (current)

- **NumPy OCR text-likelihood gate** (`estimate_text_likelihood` + settings):
  - `ocr_text_likelihood_threshold=0.30`, `ocr_text_likelihood_upper_threshold=0.85`
  - Rescue path: `0.10` + min luminance `40.0` + min blur `130.0`
  - **Why:** reduce OCR false positives and OCR call load on textless crops while still catching receipt/document-like crops.
- **Remember blur quality threshold:**
  - `blur_sharpness_threshold=120.0`
  - **Why:** adjusted from benchmark blur-score distribution to reduce over-flagging blur in user messaging.
- **Darkness gate (remember + scan):**
  - `darkness_threshold=30.0`
  - **Why:** conservative block for genuinely dark scenes where detection reliability is poor.
- **Grounding DINO detection thresholds (remember):**
  - `box_threshold=0.30`, `text_threshold=0.25`
  - **Why:** tuned for practical detectability/precision tradeoff in teach/remember flows (limited sweep artifacts currently documented).
- **YOLOE detection thresholds (scan detector stage):**
  - `yoloe_confidence=0.35`, `yoloe_iou=0.45`
  - **Why:** broad proposal recall at detector stage, with downstream matching/dedup filtering.
- **Combined embedding text weighting:**
  - `combined_text_weight=1.10`, `combined_text_weight_high_confidence_boost=0.10`
  - **Why:** improve retrieval on text-heavy objects without fully dominating image signal.

### Tuning status

- **Scan similarity thresholds:** benchmark runs compare against production settings by default. Holdout auto-tuning is opt-in via `--auto-tune-thresholds` so benchmark reports stay comparable to live thresholds.
- **Scan margin gate:** stays enabled by default for false-positive control in production and benchmark runs.
