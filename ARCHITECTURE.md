# VisualMemory Backend Architecture

> Reference doc for continuous development.
> App context: Navigation aid for blind users. Announces nearby significant objects with distance + direction.

---

## Project Overview

Python backend for a visual memory app built for blind users The system narrates the environment by detecting and locating previously-remembered objects in a new scan image.

**Two modes:**

**Remember Mode** — User specifies an object by text prompt. System detects + crops it, embeds it, stores it (database hook ready, not yet implemented).

**Scan Mode** — User takes a photo. System detects all objects, embeds each crop, searches the database for matches. Every match = a significant object. Returns structured JSON per match, including depth and direction.

---

## Setup
Request permission for DINOv2 and Grounding DINO on Hugging Face, then:
```bash
pip install -e .
python setup_weights.py
```

`pip install -e .` installs all dependencies including depth-pro from GitHub.
`setup_weights.py` downloads the Depth Pro checkpoint (~2GB) from `apple/DepthPro` on HuggingFace
directly into `checkpoints/depth_pro.pt` at the project root. Works on Windows, macOS, and Linux.

`DepthEstimator` resolves the checkpoint path at import time using `Path(__file__)`, so it works
from any working directory without symlinks or environment variables.
---

## Core Structure:

```
src/visual_memory/
├── pipelines/
│   ├── remember_mode/pipeline.py
│   └── scan_mode/pipeline.py
│
├── engine/
│   ├── embedding/embed_image.py        # CLIP Vision embedder
│   ├── object_detection/
│   │   ├── detect_all.py               # YOLOE
│   │   └── prompt_based.py             # GroundingDINO
│   └── depth/estimator.py              # Depth Pro wrapper
│
├── utils/
│   ├── image_utils.py
│   └── similarity_utils.py
│
├── config/settings.py                  # All tunable thresholds
├── api/                                # Reserved for Flask/FastAPI
└── cli/                                # Integration test scripts
```
---

## Module Reference
### `pipelines/remember_mode/pipeline.py` - `RememberPipeline`
- `run(image_path, prompt) -> dict`
- Returns `{"success": bool, "message": str, "result": {"label", "confidence", "box"} | None}`
- `add_to_database(embedding, metadata)` — stub, does nothing (awaiting DB implementation)

### `pipelines/scan_mode/pipeline.py` - `ScanPipeline`
- `__init__(database_dir: Path, focal_length_px: float)` — loads models + embeds database on init
- `run(query_image: PIL.Image) -> dict`
- Returns `{"matches": [...], "count": int}`
- Each match: `{"label", "similarity", "distance_ft", "direction", "narration"}`
- Database is re-embedded on each init (temporary — persistent DB is future work)

### `config/settings.py` - `Settings`
- Single dataclass, all ML tuning params in one place
- All detector/pipeline defaults are pulled from here — edit here to tune
- Fields: `grounding_dino_model`, `box_threshold`, `text_threshold`, `yoloe_confidence`, `yoloe_iou`, `embedder_model`, `similarity_threshold`, `dedup_iou_threshold`, `narration_high_confidence`

### `engine/object_detection/prompt_based.py` - `GroundingDinoDetector`
- Model: `IDEA-Research/grounding-dino-base` (upgraded from tiny Feb 2026)
- Input: PIL Image + text prompt
- Output: `{'box': [x1,y1,x2,y2], 'score': float, 'label': str}` or `None`
- Used in: Remember Mode

### `engine/object_detection/detect_all.py` - `YoloeDetector`
- Model: `yoloe-26l-seg-pf.pt` (local, prompt-free)
- Input: PIL Image
- Output: `(List[[x1,y1,x2,y2]], List[float])` or `(None, None)`
- Thresholds: `conf=0.35`, `iou=0.45`
- Used in: Scan Mode

### `engine/embedding/embed_image.py` - `ImageEmbedder`
- Model: `facebook/dinov3-vitl16-pretrain-lvd1689m`
- Input: PIL Image
- Output: `torch.Tensor` shape `(1, embedding_dim)`
- Used in: Both modes

### `utils/image_utils.py`
- `load_image(path)` - PIL Image (RGB, EXIF-corrected, HEIC-compatible)
- `load_folder_images(folder)` - `List[(path, PIL Image)]`
- `crop_object(image, box)` - cropped PIL Image

### `utils/similarity_utils.py`
- `cosine_similarity(t1, t2)` - scalar tensor
- `find_match(query, database, threshold)` - `(path, score)` or `(None, 0.0)`
- `deduplicate_matches(matches, iou_threshold)` - filtered list
- Linear search, fine for expected scale

### `engine/depth/estimator.py` - `DepthEstimator`
- Model: Apple Depth Pro
- `__init__(focal_length_px=None)` - loads model once
- `estimate(image: PIL.Image) -> torch.Tensor` — depth map in meters
- `get_depth_at_bbox(depth_map, bbox) -> float` - mean depth in feet, inner 50% of bbox
- `get_direction(bbox, img_w) -> str` - 5-zone direction string
- `build_narration(label, direction, distance_ft, similarity) -> str | None` - final output

---

## Pipeline Flows

### Remember Mode
```
image_path + text prompt
    → load_image()
    → GroundingDinoDetector.detect(image, prompt)
    → crop_object(image, box)
    → ImageEmbedder.embed(crop)
    → add_to_database()          ← stub, awaiting DB integration
    → return {"success": bool, "result": {...}}
```

### Scan Mode
```
init:
    database_dir → load_folder_images() → embed each → database_embeddings[]

run(query_image):
    → YoloeDetector.detect_all()
    → PASS 1: for each box → crop → embed → find_match() → collect matches
    → deduplicate_matches()
    → PASS 2: DepthEstimator.estimate(image) → depth_map  (once, only if matches found)
    → for each match:
        → get_depth_at_bbox() → distance_ft
        → get_direction()     → direction
        → build_narration()   → narration string
    → return {"matches": [...], "count": int}
```

---

## Depth Estimation

### Model: Apple Depth Pro
- Metric depth (absolute meters)
- `f_px=None` -> model infers (~75% error at close range)
- `f_px=tensor` -> calibrated (~26% error) — always use when available
- Checkpoint: `checkpoints/depth_pro.pt` — absolute path resolved via `Path(__file__)` in estimator.py

### Focal Length
```
f_px = (focalLengthMm / sensorWidthMm) * imageWidthPx
```
Pass from Android camera API. iPhone 15 Plus reference: `f_mm=6.24`, `sensor_width=8.64mm` -> `f_px=3094.0`

### Test Results (18 images, iPhone 15 Plus)
- Average error: 27.8% — acceptable for navigation
- Floor shots have higher error (camera angled down, straight-line > measured horizontal)

---

## Narration Output

```
High confidence (similarity ≥ 0.6):   "Wallet ahead, 2 feet away."
Low confidence (similarity 0.4–0.6):  "May be a wallet slightly right, focus to verify."
Below threshold (< 0.4):              → not announced
```

### Direction Logic (5-zone)
```python
if nx < -0.5:  return "to your left"
if nx < -0.15: return "slightly left"
if nx < 0.15:  return "ahead"
if nx < 0.5:   return "slightly right"
return                "to your right"
```

---

## CLI Test Scripts

Run from project root.

```bash
# Remember Mode
python -m visual_memory.cli.tests.test_remember <image_path> "<prompt>"

# Scan Mode
python -m visual_memory.cli.tests.test_scan <image_path> [--db <database_dir>] [--focal <f_px>]

# Integration test runner
python -m visual_memory.cli.tests.run_tests

# Probe detector — compact summary (single image)
python -m visual_memory.cli.tests.probe_detector <image_path> --prompt "<prompt>"

# Probe detector — full Stage 1–8 breakdown
python -m visual_memory.cli.tests.probe_detector <image_path> --prompt "<prompt>" --verbose

# Probe detector — batch distance sweep
python -m visual_memory.cli.tests.probe_detector --batch wallet
python -m visual_memory.cli.tests.probe_detector --batch airpods --prompt "airpods"
python -m visual_memory.cli.tests.probe_detector --batch all --prompt "wallet"

# Probe detector — YOLOE instead of GroundingDINO
python -m visual_memory.cli.tests.probe_detector <image_path> --prompt "<prompt>" --detector yoloe
```

`probe_detector` replaces `inspect_scores` + `debug_detection_stages`. Compact mode shows
raw sigmoid stats at threshold=0 with markers at the configured threshold. Batch mode sweeps
all `{object}_{1ft,3ft,6ft}_{floor,table}.jpg` variants and prints a summary table — use this
to confirm at which distance GroundingDINO reliably detects the target.

---

## API Plan 

Both pipelines return plain Python dicts - JSON-serializable, no torch tensors in output.

### POST /remember
**Input:** `image` (file), `prompt` (string), optional: `focal_length_px` (float)
**Pipeline call:** `RememberPipeline().run(image_path, prompt)`
**Response:**
```json
{
  "success": true,
  "message": "Object detected and embedded successfully.",
  "result": {
    "label": "wallet",
    "confidence": 0.87,
    "box": [120, 200, 340, 410]
  }
}
```

### POST /scan
**Input:** `image` (file), `focal_length_px` (float, required for accuracy), optional: `database_dir` (str)
**Pipeline call:** `ScanPipeline(database_dir, focal_length_px).run(image)`
**Response:**
```json
{
  "matches": [
    {
      "label": "wallet",
      "similarity": 0.72,
      "distance_ft": 2.3,
      "direction": "ahead",
      "narration": "Wallet ahead, 2 feet away."
    }
  ],
  "count": 1
}
```

## Current State

- Core engine complete and tested (depth, detection, embedding)
- Both pipelines return structured JSON dicts
- Test scripts functional at `src/visual_memory/cli/tests`
- `api/` directory empty, awaiting implementation -
- Database is flat folder of images, re-embedded on each `ScanPipeline` init - temporary
- All testing via local image files until connected to frontend

---
## TODO
- [ ] Switch embedder from DINOv3 to CLIP - +0.37 similarity improvement measured in testing; update `Settings.embedder_model` and retune `similarity_threshold` (0.3 -> 0.5–0.6)
- [ ] Validate end-to-end scan pipeline with real device images

### Next: Database 
- [ ] Implement `RememberPipeline.add_to_database()` 
- [ ] Replace folder-based `load_folder_images()` + re-embed in `ScanPipeline` with DB query function
- [ ] Get embeddings for similarity: `get_all_embeddings() -> List[(id, label, embedding)]`

### Cleanup
- [ ] Shorten this doc after API + DB phases complete
---



## Future Plans:
    Input Enhancement in Remember Mode:
        Run user prompt through a lightweight LLM to expand vague descriptions before Grounding DINO.
        `"remember my airpods"` → `["white round earbuds", "white round earbud case", ...]` -> best detection out of 3-5 wins.

    GPU Server Migration:
        `device='cuda'` across all models,
        Batch processing

    Bloat Prevention (once sqlite is implemented)
    Duplicate entry detection, pruning unused entries, user confirmation before overwriting similar embeddings.

### Optional Plans (extra time)
Phone IMU reports exact tilt angle to  compensate depth for angled shots. Android/iOS concern, not backend.
HNSW index for search - not difficult to implement, but benefit is marginal for >10k images

## Design Decisions and Why

- **DINOv3 for embeddings** - CLIP switch planned. CLIP captures semantic identity better, and better handles real world lighting changes, object upsidedown, etc. Has downside of making two wallets that look different similar, but not an issue for this app's scope
- **YOLOE prompt-free for scan** - broad detection, similarity handles bloat
- **Grounding DINO for remember** - handles vague natural language well, much stronger than prompted YOLOE for this use
- **Depth Pro** — only monocular model with metric (absolute) depth, no scale factor needed
- **Actionability over information/accuracy** - "May be a wallet two feet to your right, focus to verify" is actionable, not just informational: "Wallet detected two feet to your right, 23% confidence"
