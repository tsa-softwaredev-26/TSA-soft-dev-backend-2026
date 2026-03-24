# VisualMemory Backend Architecture

> Reference doc for continuous development.
> App context: Navigation aid for blind users. Announces nearby significant objects with distance + direction.

---

## Project Overview

Blind users cannot quickly visually re-scan their environment the way sighted people naturally do. This makes spatial recall slow and cognitively draining.

Spaitra restores this ability by creating a personal, spatial memory of the user’s physical world.


## Core Modes

### Teach Mode

> “These are my house keys on the dining room table.”

Single-shot memory creation.

- Detect object
- Generate visual + text embeddings
- Store spatial position
- Add to personal memory index

The user can confirm or correct detections to improve personalization.

---

### Scan Mode

Recognizes previously taught objects in the current scene.

Example narration:
> “Wallet 3 feet to your left. House keys 5 feet to your right. Tax returns 3 feet ahead.”

- Objects are ordered spatially (left -> right)
- Results returned as structured JSON

---

### Ask Mode

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
- Personalized self-training detection (head on top of DINOv3) to improve accuracy with use
- Continuous adaptation to the user’s environment
- Minimalist, voice-based interface

## Setup

Request access to both gated models on Hugging Face, then git clone and run setup:
- [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base)
- [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)


```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
hf auth login
python setup_weights.py
```

`pip install -e .` installs all dependencies including depth-pro from GitHub.
`setup_weights.py` downloads the Depth Pro checkpoint (~2GB) and YOLOE weights into the project root.
`DepthEstimator` resolves the checkpoint path at import time using `Path(__file__)`.
DINOv3 (`facebook/dinov3-vitl16-pretrain-lvd1689m`, gated - request access on HuggingFace) and CLIP text encoder (`openai/clip-vit-base-patch32`) are downloaded automatically on first use via `transformers`.

---

## Core Structure

```
src/visual_memory/
├── pipelines/
│   ├── remember_mode/pipeline.py
│   └── scan_mode/pipeline.py
│
├── engine/
│   ├── embedding/
│   │   ├── embed_image.py              # ImageEmbedder (DINOv3, 1024-dim)
│   │   ├── embed_text.py              # CLIPTextEmbedder (CLIP text encoder, 512-dim)
│   │   └── embed_combined.py          # make_combined_embedding() - 1536-dim output
│   ├── text_recognition/
│   │   ├── base.py                    # BaseTextRecognizer ABC
│   │   └── paddle_recognizer.py       # PaddleOCRRecognizer (active)
│   ├── object_detection/
│   │   ├── detect_all.py              # YOLOE
│   │   └── prompt_based.py            # GroundingDINO
│   └── depth/estimator.py             # Depth Pro wrapper
│
├── utils/
│   ├── image_utils.py
│   ├── similarity_utils.py
│   └── logger.py                      # JSON structured logging
│
├── learning/
│   ├── projection_head.py             # ProjectionHead (residual linear, identity at init)
│   ├── trainer.py                     # ProjectionTrainer + triplet_loss (CLI: python -m visual_memory.learning.trainer)
│   └── feedback_store.py              # FeedbackStore (.pt file persistence, DB-ready interface)
│
├── benchmarks/
│   ├── full_benchmark.py              # 7-phase system benchmark (retrieval, detection, depth)
│   ├── format_results.py              # reads results.json, writes BENCHMARKS.md at root
│   ├── check_dataset.py               # pre-flight: verify all 120 images are in benchmarks/images/
│   └── redact_receipt.py              # gitignored - OCR-based receipt redaction (personal tool)
│
├── config/
│   ├── settings.py                     # All ML tuning thresholds
│   └── user_settings.py                # User preferences: PerformanceMode, UserSettings (JSON-persisted)
├── api/
│   ├── app.py                          # create_app() factory - blueprints, auth, upload cap
│   ├── pipelines.py                    # Lazy singletons: get_remember_pipeline(), get_scan_pipeline(), get_feedback_store(), get_user_settings(), warm_all()
│   ├── run.py                          # Local dev entry point (python src/visual_memory/api/run.py)
│   └── routes/
│       ├── health.py                   # GET /health
│       ├── remember.py                 # POST /remember
│       ├── scan.py                     # POST /scan
│       ├── feedback.py                 # POST /feedback
│       ├── retrain.py                  # POST /retrain
│       ├── settings_route.py           # GET /settings, PATCH /settings (ML tuning)
│       ├── user_settings_route.py      # GET /user-settings, PATCH /user-settings (user prefs)
│       ├── find.py                     # GET /find - last-seen location query (Ask Mode)
│       └── sightings.py                # DELETE /sightings/<id> - remove a false-positive sighting
└── tests/                             # Test scripts + test data
    ├── scripts/                        # Runnable .py test scripts
    ├── input_images/                   # Object test images
    ├── demo_database/                  # Reference crops for ScanPipeline
    └── text_demo/                      # OCR test images + ground_truth/
```

---

## Module Reference

### `pipelines/remember_mode/pipeline.py` - `RememberPipeline`
- `run(image_path: str | Path, prompt: str) -> dict`
- Returns `{"success": bool, "message": str, "result": {"label": str, "confidence": float, "box": [x1,y1,x2,y2], "ocr_text": str, "ocr_confidence": float} | None}`
- `add_to_database(embedding, metadata)` - persists combined embedding to SQLite via DatabaseStore

### `pipelines/scan_mode/pipeline.py` - `ScanPipeline`
- `__init__(focal_length_px: float, db_path: str | Path = None)` - loads models + fetches DB embeddings on init
- `run(query_image: PIL.Image, scan_id: str | None = None, focal_length_px: float | None = None) -> dict`
- `scan_id`: if provided, caches (anchor_emb, query_emb) per label for /feedback lookup
- `focal_length_px`: per-call override; falls back to `self.focal_length_px` if None
- Returns `{"matches": [...], "count": int}`
- Each match (depth enabled): `{"label": str, "similarity": float, "distance_ft": float, "direction": str, "narration": str, "ocr_text": str (optional)}`
- Each match (depth disabled): `{"label": str, "similarity": float, "box": list, "ocr_text": str (optional)}`
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
- `UserSettings` fields: `performance_mode` (BALANCED), `voice_speed` (1.0, range 0.5-2.0), `auto_update_location` (False), `learning_enabled` (True), `button_layout` ("default" | "swapped", stub)
- `save(db)` / `load(db)` - DB round-trip with safe defaults on missing/corrupt row
- `to_dict()` - serializes for API response; includes derived `performance_config` block
- Exposed via GET /user-settings, PATCH /user-settings
- PATCH /user-settings propagates `learning_enabled` to ScanPipeline immediately

### `config/settings.py` - `Settings`
- Single dataclass, all ML tuning params in one place
- Detection: `grounding_dino_model`, `box_threshold`, `text_threshold`, `yoloe_confidence`, `yoloe_iou`
- Embedding: `image_embedder_model` (`dinov3-vits16`), `embedder_model` (`clip-vit-base-patch32`)
- Matching: `similarity_threshold` (0.2), `dedup_iou_threshold` (0.5), `narration_high_confidence` (0.6)
- OCR: `ocr_backend` ("paddle"), `ocr_languages`, `ocr_min_confidence` (0.3), `text_similarity_threshold` (0.4)
- Toggles (env-overridable): `enable_depth` (`ENABLE_DEPTH`), `enable_ocr` (`ENABLE_OCR`), `enable_dedup` (`ENABLE_DEDUP`)
- Personalization: `projection_head_path` ("models/projection_head.pt"), `projection_head_dim` (1536)
- Learning: `enable_learning` (`ENABLE_LEARNING`), `min_feedback_for_training` (10), `projection_head_weight` (1.0), `projection_head_ramp_at` (50), `projection_head_epochs` (20)
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

### `engine/text_recognition/paddle_recognizer.py` - `PaddleOCRRecognizer`
- Model: PaddleOCR-VL-1.5 with PP-DocLayoutV3 layout detection
- Input: PIL Image (saved to temp PNG, passed as file path)
- Output: `{"text": str, "confidence": float, "segments": list}`
- Inference: ~3-18s per image
- Extracts text from `parsing_res_list[].block_content` in JSON export

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
- CLI: `python -m visual_memory.learning.trainer [--feedback-dir feedback/] [--output models/projection_head.pt] [--epochs 20] [--lr 1e-4]`

### `learning/feedback_store.py` - `FeedbackStore`
- `FeedbackStore(store_dir=Path("feedback"))` - creates dir on init
- `record_positive(anchor_emb, query_emb, label)` - saves `.pt` file (type="positive")
- `record_negative(anchor_emb, query_emb, label)` - saves `.pt` file (type="negative")
- `load_triplets() -> List[(anchor, positive, negative)]` - pairs each positive with each negative per label
- `count() -> dict` - `{"positives": int, "negatives": int, "triplets": int}`
- DB contract and Flask endpoint contract documented in module docstring

### `utils/device_utils.py`
- `get_device() -> str` - returns "cuda", "mps", or "cpu" in priority order; used by all engine modules

### `utils/image_utils.py`
- `load_image(path)` - PIL Image (RGB, EXIF-corrected, HEIC-compatible)
- `load_folder_images(folder)` - `List[(path, PIL Image)]`
- `crop_object(image, box)` - cropped PIL Image

### `utils/similarity_utils.py`
- `cosine_similarity(t1, t2)` - scalar tensor
- `find_match(query, database, threshold)` - `(path, score)` or `(None, 0.0)`
- `deduplicate_matches(matches, iou_threshold)` - filtered list

### `engine/depth/estimator.py` - `DepthEstimator`
- Model: Apple Depth Pro
- `__init__(focal_length_px=None)` -loads model once
- `estimate(image: PIL.Image) -> torch.Tensor` -depth map in meters
- `get_depth_at_bbox(depth_map, bbox) -> float` -mean depth in feet, inner 50% of bbox
- `get_direction(bbox, img_w) -> str` -5-zone direction string
- `build_narration(label, direction, distance_ft, similarity) -> str | None` -final output

---

## Pipeline Flows

### Remember Mode
```
image_path + text prompt
    -> load_image()
    -> GroundingDinoDetector.detect(image, prompt)
    -> crop_object(image, box)
    -> TextRecognizer.recognize(crop)        -> OCR text
    -> ImageEmbedder.embed(crop)              -> image embedding (1024-dim)
    -> CLIPTextEmbedder.embed_text(ocr_text)  -> text embedding (512-dim)
    -> make_combined_embedding()              -> combined (1536-dim)
    -> add_to_database()                    <- stub, awaiting DB integration
    -> return {"success": bool, "result": {...}}
```

### Scan Mode
```
init:
    database_dir -> load_folder_images() -> embed each -> database_embeddings[]
    ProjectionHead.load("models/projection_head.pt") -> _head_trained (True/False)

run(query_image):
    -> YoloeDetector.detect_all()
    -> PASS 1: for each box -> crop -> embed (combined 1536-dim)
        -> if _head_trained: project query + all DB embeddings via ProjectionHead
        -> find_match() -> collect matches
    -> deduplicate_matches()
    -> PASS 2: DepthEstimator.estimate(image) -> depth_map  (once, only if matches found)
    -> for each match:
        -> get_depth_at_bbox() -> distance_ft
        -> get_direction()     -> direction
        -> build_narration()   -> narration string
    -> return {"matches": [...], "count": int}
```

---

## Depth Estimation

### Model: Apple Depth Pro
- Metric depth (absolute meters)
- `f_px=None` -> model infers (~75% error at close range)
- `f_px=tensor` -> calibrated (~26% error) -always use when available
- Checkpoint: `checkpoints/depth_pro.pt` -absolute path resolved via `Path(__file__)` in estimator.py

### Focal Length
```
f_px = (focalLengthMm / sensorWidthMm) * imageWidthPx
```
Pass from Android camera API. iPhone 15 Plus reference: `f_mm=6.24`, `sensor_width=8.64mm` -> `f_px=3094.0`

### Test Results (18 images, iPhone 15 Plus)
- Average error: 27.8% -acceptable for navigation

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
Objects: wallet_a, wallet_b, book_a, book_b, sunglasses_a, sunglasses_b, receipt_a, receipt_b, keys_a, keys_b.
Labels are per-instance (wallet_a vs wallet_b) to test instance-level discrimination.

### Phases

1. **Embed** - DINOv3 image embedding + optional PaddleOCR text embedding for each image.
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

### Metrics

| Metric | What it measures |
|--------|-----------------|
| Baseline accuracy | Retrieval using raw DINOv3+CLIP embeddings at production threshold |
| Personalized accuracy | Retrieval after ProjectionHead trained on benchmark train set |
| Accuracy delta (pp) | Improvement from projection head; positive = head helps |
| Mean sim gap | Per-image average improvement in cosine similarity score |
| Triplet final loss | Training convergence; lower = more distinct embedding clusters |
| Detection rate | GroundingDINO hit rate per object and per condition |
| Mean depth abs error (ft) | Depth Pro accuracy on detected objects with known ground truth |
| Mean depth % error | Same as above normalized by ground truth distance |

---

## Test Scripts

Run from project root (`python -m visual_memory.tests.scripts.<name>`).

```bash
# Integration test runner - DEPTH=0 (default) skips Depth Pro; DEPTH=1 includes it (~2GB, memory-heavy)
python -m visual_memory.tests.scripts.run_tests
DEPTH=1 python -m visual_memory.tests.scripts.run_tests
VERBOSE=1 python -m visual_memory.tests.scripts.run_tests   # show OCR text diff

# OCR benchmark - extract text from text_demo/ images, check accuracy
python -m visual_memory.tests.scripts.test_ocr_benchmark

# Standalone CLI scripts
python -m visual_memory.tests.scripts.test_remember <image_path> "<prompt>"
python -m visual_memory.tests.scripts.test_scan <image_path> [--db <dir>] [--focal <f_px>]
python -m visual_memory.tests.scripts.test_text_recognition [image_path]
python -m visual_memory.tests.scripts.test_estimator

# Detector probe
python -m visual_memory.tests.scripts.probe_detector <image_path> --prompt "<prompt>"
python -m visual_memory.tests.scripts.probe_detector --batch wallet
python -m visual_memory.tests.scripts.probe_detector <image_path> --detector yoloe

# Projection head unit tests (CPU-only, no model loading, ~2s)
python -m visual_memory.tests.scripts.test_projection_head

# Train projection head from collected feedback
python -m visual_memory.learning.trainer
python -m visual_memory.learning.trainer --epochs 50 --lr 5e-5
```

---

## API

Run locally: `python src/visual_memory/api/run.py`
Production (Ubuntu/NVIDIA): `gunicorn -w 1 -b 0.0.0.0:5000 "visual_memory.api.app:create_app()"`
Single worker is required - model state is process-local.

**Environment variables:**
- `API_KEY=<secret>` - enforce `X-API-Key` header on all routes except `/health` (opt-in; unset = no auth)
- `ENABLE_DEPTH=0` - skip Depth Pro (~2GB VRAM savings)
- `ENABLE_OCR=0` - skip PaddleOCR
- `ENABLE_LEARNING=0` - disable projection head application in scan (also patchable at runtime via PATCH /settings)
- `api_host` / `api_port` controlled via `settings.py` (defaults: 127.0.0.1:5000)

Models are loaded once at startup via `warm_all()` in `create_app()`. Upload cap: 50MB.

**Endpoints:** GET /health, POST /remember, POST /scan, POST /feedback, POST /retrain, GET /settings, PATCH /settings, GET /user-settings, PATCH /user-settings, GET /find, DELETE /sightings/<id>.

See `FRONTEND_GUIDE.md` for full request/response schemas, field reference, and integration patterns.

---

## Known Benchmark Data

### Detection score data (ad-hoc, pre-full-benchmark)

| model               | image                | prompt | max sigmoid       |
|---------------------|----------------------|--------|-------------------|
| grounding-dino-tiny | wallet_6ft_table.jpg | wallet | 0.1655            |
| grounding-dino-tiny | skipper.HEIC         | wallet | 0.2286 (FP)       |
| grounding-dino-base | wallet_6ft_table.jpg | wallet | 0.2036 (FN)       |
| grounding-dino-base | skipper.HEIC         | wallet | 0.3308 (FP)       |

6ft detection out of range for GDINO-base (FP > FN, no safe threshold). Use 1ft/3ft images.

### DINOv3 + CLIPText similarity data (March 2026)

Run via `python -m visual_memory.tests.scripts.benchmark_embedder`.

**Section A - Intra vs inter-class similarity (DINOv3 image embeddings, full-scene input_images/)**

| Embedder | Intra-class sim | Inter-class sim | Ratio |
|----------|-----------------|-----------------|-------|
| DINOv3   | 0.5133          | 0.5151          | 0.996 |

Note: all 9 images share the same context (objects on a table). Full-scene intra/inter gap is not meaningful.
Pipeline uses YOLOE crops before embedding - see pipeline scan test below.

**Section A - Pipeline scan match (wallet_1ft_table.jpg YOLOE crops vs cropped_wallet.png reference)**

| Detection | DINOv3 sim |
|-----------|------------|
| crop 1    | 0.3156     |
| crop 2    | 0.3058     |
| crop 3    | 0.2385     |

All 3 crops above `similarity_threshold=0.2`. Threshold kept at **0.2**.

**Section B - CLIPText similarity on OCR ground truth (text_demo/ images)**

All 4 text_demo ground truth files share the same "The quick brown fox..." sentence.
All pairwise similarities = 1.0000. Cross-text gap cannot be measured from this test set.
`text_similarity_threshold=0.3` kept - real OCR outputs will differ.

**Section C - Combined embedding similarity (DINOv3 image + CLIPText, 1536-dim)**

- Similar-text cluster (marker/pen/pencil/typed): similarity 0.78-0.95
- Separate cluster (random_printed_notes, malarkey): similarity 0.18-0.30 vs similar-text cluster
- Threshold 0.2 separates clusters cleanly.

---

## Current State

- Core engine complete and tested (depth, detection, embedding, OCR)
- Both pipelines produce structured JSON dicts; combined 1536-dim embeddings (DINOv3 + CLIP text)
- `run_tests.py` passes (March 2026): remember:detect, scan:match, text_recognition x2 (DEPTH=0 skips estimator)
- `learning/` module complete: ProjectionHead, ProjectionTrainer, FeedbackStore - all tested CPU-only
- `test_projection_head.py` passes: identity-at-init, triplet loss, store roundtrip, training convergence
- ProjectionHead wired into ScanPipeline with auto-scaling blend (`_apply_head`); no-op until `models/projection_head.pt` exists
- Flask API: POST /remember, POST /scan, POST /feedback, POST /retrain, GET /settings, PATCH /settings, GET /health
- /retrain trains ProjectionHead from FeedbackStore and hot-reloads weights into ScanPipeline; guarded by `min_feedback_for_training`
- /settings exposes runtime toggles (enable_learning, projection_head_weight, etc.) without restart
- Database: SQLite via DatabaseStore; both pipelines persist/query embeddings
- ScanPipeline reloads DB after /remember; scan_id embedding cache for /feedback (last 50)
- OCR backend: PaddleOCR-VL-1.5 (`ocr_backend = "paddle"`)
- Image embedder: DINOv3 ViT-S/16 (`facebook/dinov3-vits16-pretrain-lvd1689m`, gated on HuggingFace)
- Text embedder: CLIP text encoder only (`openai/clip-vit-base-patch32`, 512-dim, ~180MB)

---

## TODO

### Engine / Architecture
- [ ] Run full benchmark - 120 images not yet captured; see `benchmarks/CAPTURE_GUIDE.md` (~1-2 hrs)
- [ ] Wire `batch_embed_text()` into pipeline OCR paths once PaddleOCR supports batch (currently CPU-sequential; no gain until backend changes)
- [ ] Wire `detect_all_batch()` into ScanPipeline when processing multiple images per request
- [ ] Wire `detect_batch()` into RememberPipeline if multi-prompt remember is added
- [ ] Remove `engine/embedding/projection_head/` empty dir (leftover from earlier experiment)
- [ ] `str | Path` type hints in pipeline files require Python 3.10+; `pyproject.toml` allows 3.8+ (pre-existing, low priority)

### Database
- [x] `user_state` table: store serialized ProjectionHead weights per user in DB; `/retrain` saves to DB after training
- [x] `ScanPipeline._load_head()` - loads projection head from DB first, falls back to file; used by `__init__` and `reload_head()`
- [x] `sightings` table: every successful `/scan` match writes label, timestamp, direction, distance_ft, similarity, crop_path; `GET /find?label=<q>` queries it; `DELETE /sightings/<id>` removes a false-positive
- [ ] Fuzzy label search in `/find`: embed the query string via CLIPTextEmbedder (available as `get_scan_pipeline().text_embedder`), embed each label from `get_known_labels()`, return labels above cosine similarity threshold - handles "wallet" matching "my brown wallet" etc.
- [ ] History-aware sightings retrieval: add time-range params to `/find` (`since`, `before`), recency-weighted ranking; sightings are retained indefinitely by design (personal spatial memory, not a recent-activity log) - prune only after search quality is optimized


### Learning / Personalization
- [ ] Async /retrain - training blocks the request thread (seconds to minutes depending on triplet count); move to a background thread with a GET /retrain/status polling endpoint before server deployment
- [ ] Persist runtime settings overrides (enable_learning, projection_head_weight, etc.) to DB so they survive restarts; currently in-memory only
- [ ] Test suite: add `test_learning_pipeline.py` covering FeedbackStore roundtrip, triplet loading, trainer convergence, `_apply_head` blending at various triplet counts, `reload_head`, and full /feedback -> /retrain -> /settings endpoint flow

---

## Server Transition Notes

Steps needed before deploying to a Linux/NVIDIA server (can be completed by an agent in ~5 min each):

1. **Async /retrain** - wrap training in `threading.Thread`, store status in a module-level dict, add `GET /retrain/status` route returning `{"running": bool, "last_result": {...}}`. Change `/retrain` to start the thread and return `{"started": true}` immediately.
2. **Persist ML settings** - PATCH /settings currently writes to an in-memory Settings instance only. On `PATCH /settings`, also write changed fields to DB (e.g., a `settings` TEXT column in `user_state`). On startup, read from DB and apply via the same setters. (User prefs already persisted via `user_state.user_settings`.)
3. **Move FeedbackStore to DB** - `FeedbackStore.record_positive/record_negative` currently write `.pt` files to `feedback/`. Replace with `INSERT INTO feedback (label, type, anchor_blob, query_blob, timestamp)` in the user DB. Update `load_triplets()` to `SELECT` and deserialize. This removes a flat-file dependency and enables per-user feedback isolation.

---

## Future Plans

- **Input Enhancement in Remember Mode** - Run user prompt through a lightweight LLM to expand vague descriptions before Grounding DINO. `"remember my airpods"` -> best detection wins from expanded candidates.
- **Bloat Prevention** - Duplicate entry detection, pruning unused entries, user confirmation before overwriting similar embeddings.
- **HNSW index** - Marginal benefit below ~10k entries; defer until scale requires it.
- **Pipeline batching** - ScanPipeline can call `batch_embed()` and `detect_all_batch()` instead of per-crop loops for full GPU utilization on the server.

---

## Design Decisions

- **Embedder: DINOv3 for images, CLIP text encoder for OCR (March 2026)** - Started with DINOv3 (1024-dim) + sentence-transformers (384-dim). Tried switching to CLIP ViT-B/32 alone (512-dim shared image+text space) to simplify the stack and unify embedding spaces. Reverted image embeddings back to DINOv3 after benchmarking: CLIP image intra/inter-class ratio 0.961 (intra=0.769, inter=0.800) on full-scene images - objects on the same table appeared more similar cross-class than the same object at different distances. DINOv3 is self-supervised and object-centric, producing better visual discrimination for the scan use case. CLIP text encoder kept for OCR matching - lightweight text-only projection (~180MB), good semantic alignment for product labels. sentence-transformers removed entirely. Combined vector: 1536-dim (DINOv3 1024 + CLIP text 512). DINOv3 is gated on HuggingFace - requires access request.
- **YOLOE prompt-free for scan** - Broad detection; similarity threshold handles bloat.
- **Grounding DINO for remember** - Handles vague natural language; much stronger than prompted YOLOE for this use case.
- **Depth Pro** - Only monocular model with metric (absolute) depth; no scale factor needed.
- **Actionability over accuracy** - "May be a wallet two feet to your right, focus to verify" is actionable. A raw confidence percentage is not.
- **PaddleOCR over EasyOCR** - Switched from EasyOCR to PaddleOCR-VL-1.5 (Feb 2026). EasyOCR averaged 9.7% word-overlap on the `text_demo/` test set at 200-435s/image. PaddleOCR-VL-1.5 runs in 3-18s/image (~15x faster) and correctly extracts layout-aware text from `parsing_res_list[].block_content` in its JSON output. Text extraction was initially broken due to the engine searching wrong JSON paths; fixed by targeting the correct field.
- **Combined embedding over hybrid matching** - `max(image_sim, text_sim)` could false-match two white documents with different text (image similarity alone passes threshold). The combined embedding `normalize(img) || normalize(text)` means cosine similarity is approximately the average of both sub-similarities, so both image AND text must match. Non-document objects get a zero text slot; image similarity dominates unchanged.
- **Single venv** - All dependencies including PaddleOCR live in one `pip install -e .` install.
- **Projection head: identity-at-init residual design** - A single `Linear(1536, 1536, bias=False)` initialized to zeros. At init `output = normalize(x + 0) = x`, so the head is a strict no-op before any training. Safe to wire in by default with zero runtime cost. After triplet training, the head learns a small residual correction that pulls user-confirmed matches closer and pushes mismatches apart. Raw base embeddings are stored; projection applied on-the-fly at match time, so retraining only requires reloading weights - no re-embedding the database.
