# VisualMemory — Backend Architecture

> Reference doc for continuous development. Branch: `depth-perception`.
> App context: Navigation aid for blind users. Announces nearby significant objects with distance + direction.
> Last updated: Feb 2026.

---

## Project Overview

Python backend for a visual memory app built for **blind users**. The system narrates the environment by detecting and locating previously-remembered objects in a new scan image.

**Two modes:**

**Remember Mode** — User specifies an object by text prompt. System detects + crops it, embeds it, stores it. This is how "significant objects" are registered.

**Scan Mode** — User takes a photo. System detects all objects, embeds each crop, searches the database for matches. Every match = a significant object. Returns narration string per match, including depth and direction.

> Depth estimation is part of Scan Mode, not a separate mode.

---

## Setup

```bash
pip install -e .
python setup_weights.py
```

`pip install -e .` installs all dependencies including depth-pro from GitHub.
`setup_weights.py` downloads the Depth Pro checkpoint (~2GB) and creates a symlink at `checkpoints/` so `create_model_and_transforms()` works from any directory.
`checkpoints/` is gitignored — every teammate runs this once after cloning.

---

## Directory Structure

```
VisualMemory/
├── remember_mode/
│   └── main.py                     # Prompt → detect → crop → embed pipeline
├── scan_mode/
│   └── main.py                     # detect all → embed → match → depth + narration
├── depth_estimation/
│   ├── __init__.py
│   ├── estimator.py                # DepthEstimator class + narration logic
│   └── test_estimator.py           # manual test — configure paths and run
├── object_detection/
│   ├── detect_all_objects.py       # YoloeDetector (prompt-free, all objects)
│   ├── prompt_based_detector.py    # GroundingDinoDetector (text prompt → best box)
│   └── yoloe-26l-seg-pf.pt        # Local YOLOE weights (gitignored)
├── embed_image/
│   └── embed_image.py              # DINOv3 ViT-Large embedder
├── utils/
│   ├── image_utils.py              # load_image, load_folder_images, crop_object
│   └── similarity_utils.py         # cosine_similarity, find_closest_match
├── depth_demo/                     # throwaway test suite — delete after integration confirmed
│   ├── depth_test.py
│   ├── ground_truth.json
│   └── ml-depth-pro/               # gitignored
├── checkpoints/ → symlink          # gitignored, created by setup_weights.py
└── setup_weights.py
```

---

## Module Reference

### `object_detection/prompt_based_detector.py` — `GroundingDinoDetector`
- Model: `IDEA-Research/grounding-dino-tiny`
- Input: PIL Image + text prompt
- Output: `{'box': [x1,y1,x2,y2], 'score': float, 'label': str}` or `None`
- Returns single best detection (highest confidence)
- Device: auto (CUDA > MPS > CPU)
- Used in: Remember Mode

### `object_detection/detect_all_objects.py` — `YoloeDetector`
- Model: `yoloe-26l-seg-pf.pt` (local, prompt-free)
- Input: image file path
- Output: `(List[[x1,y1,x2,y2]], List[float])` or `(None, None)`
- Thresholds: `conf=0.35`, `iou=0.45`
- Used in: Scan Mode

### `embed_image/embed_image.py` — `ImageEmbedder`
- Model: `facebook/dinov3-vitl16-pretrain-lvd1689m`
- Input: PIL Image
- Output: `torch.Tensor` shape `(1, embedding_dim)`
- `device_map="auto"` handles MPS/CUDA
- Used in: Both modes

### `utils/image_utils.py`
- `load_image(path)` → PIL Image (RGB, EXIF-corrected, HEIC-compatible)
- `load_folder_images(folder)` → `List[(path, PIL Image)]`
- `crop_object(image, box)` → cropped PIL Image

### `utils/similarity_utils.py`
- `cosine_similarity(t1, t2)` → scalar tensor
- `find_closest_match(query, database, threshold)` → `(path, score)` or `(None, 0.0)`
- Linear search, fine for expected scale

### `depth_estimation/estimator.py` — `DepthEstimator`
- Model: Apple Depth Pro
- `__init__(focal_length_px=None)` — loads model once
- `estimate(image: PIL.Image) -> torch.Tensor` — depth map in meters, call once per image
- `get_depth_at_bbox(depth_map, bbox) -> float` — mean depth in feet, inner 50% of bbox
- `get_direction(bbox, img_w) -> str` — 5-zone direction string
- `build_narration(label, direction, distance_ft, similarity) -> str | None` — final output
- All spatial + narration logic lives here, not in utils

---

## Mode Pipelines

### Remember Mode (`remember_mode/main.py`)
```
text prompt
    → GroundingDinoDetector.detect(image, prompt)
    → crop_object(image, box)
    → ImageEmbedder.embed(crop)     ← commented out, not yet storing
    → show cropped object
```



### Scan Mode 
```
database folder → load_folder_images() → embed each → database_embeddings[]

query image
    → YoloeDetector.detect_all()
    → DepthEstimator.estimate(image) → depth_map   ← once per image
    → for each detected box:
        → crop → embed → find_closest_match()
        → if match:
            → get_depth_at_bbox(depth_map, box) → distance_ft
            → get_direction(box, img_w)          → direction
            → build_narration(label, direction, distance_ft, similarity)
    → print narration per matched object
```

---

## Depth Estimation

### Model: Apple Depth Pro
- Metric depth (absolute meters) — no scale factor needed
- `f_px=None` → model infers (~75% error at close range)
- `f_px=tensor` → calibrated (~26% error) — always use when available
- Checkpoint: `checkpoints/depth_pro.pt` via symlink

### Focal Length
# f_px = (focalLengthMm / sensorWidthMm) * imageWidthPx
Pass from Android camera API: `f_px = (f_mm / sensor_width_mm) * image_width_px`
iPhone 15 Plus reference: `f_mm=6.24`, `sensor_width=8.64mm` ->`f_px=3094.0`

### Test Results (18 images, iPhone 15 Plus)
- Average error: 27.8% — acceptable for navigation
- Floor shots have higher error (camera angled down, straight-line > measured horizontal)
- At 3ft = ~10 inches off, at 1ft = ~3 inches off

---

## Narration Output

### Philosophy
Dead simple output. One distance, one direction, one confidence signal.
User tilts phone naturally to scan — no camera angle constraints.

### Output
```
High confidence (similarity ≥ 0.6):
    "Wallet ahead, 2 feet away."
    "Mouse to your left, 3 feet away."

Low confidence (similarity 0.4–0.6):
    "May be a wallet slightly right, focus to verify."

Below threshold (similarity < 0.4):
    → do not announce
```

### Direction Logic
Previously clock position (e.g. "3 o'clock") — switched to 5-zone plain directions.
Removes mental translation step. Could revert to clock if finer granularity is needed.

```python
# nx = normalized bbox center x, -1=far left, 1=far right
if nx < -0.5:  return "to your left"
if nx < -0.15: return "slightly left"
if nx < 0.15:  return "ahead"
if nx < 0.5:   return "slightly right"
return                "to your right"
```

---

## Current State

- `depth_estimation/` package complete and tested
- `scan_mode/main.py` integrated with depth — two-pass design (match first, depth only if matches found)
- Pipeline not yet run end-to-end — next step
- `depth_demo/` still present — delete after scan_mode confirmed working
- All testing local image files only — no live camera, no server
- Will be exposed via Flask (teammates), then Android via KMP

---

## TODO

### Run and validate scan_mode end-to-end
- [ ] `pip install -e .` to register depth_estimation package
- [ ] Run `python scan_mode/main.py` — success = narration printed per match with distance + direction
- [ ] Tune confidence thresholds based on live output

### Cleanup (after confirmed working)
- [ ] Delete `depth_demo/`
- [ ] Commit

---

## Future Plans (Do Not Implement Yet)

### IMU Tilt Compensation
Phone IMU (accelerometer + gyroscope) reports exact tilt angle. Could compensate depth when camera is angled, removing need for natural scanning. Android/iOS concern, not backend. Out of scope for TSA presentation.

### Input Enhancement — Remember Mode
Pass user query through lightweight + quick LLM to expand vague descriptions before Grounding DINO.
- `"remember my airpods"` → `["white round earbuds", "white round earbud case", "round earbud case"]`
- Run detection with each, pick best result.

### Persistent Embedding Cache
Cache embeddings to `.npz` — only needed once Flask + Android integration begins.

### GPU Server Migration
- `device='cuda'` across all models
- Batch processing in embedder + detector
- HNSW index (hnswlib/FAISS) for similarity search

---

## Key Design Decisions

- **DINOv3 for embeddings** — better visual similarity than CLIP, no text alignment needed
- **YOLOE prompt-free for scan** — broad detection, similarity handles disambiguation
- **Grounding DINO for remember** — handles vague natural language well
- **Depth Pro** — only monocular model with metric (absolute) depth, no scale factor needed
- **Calibrated focal length** — 26% error vs 75% inferred; pass f_px from Android camera API
- **5-zone directions over clock** — maps directly to body movement, no mental translation; could revert if granularity needed
- **No vertical zone** — user's natural scanning handles it; chest-height assumption broke for floor objects and wheelchair users
- **Depth map once per image** — reused across all matched objects
- **Straight-line distance** — consistent regardless of camera angle
- **Confidence narration** — "May be a wallet... focus to verify" is actionable, not just informational
- **HEIC support** via `pillow-heif` in `load_image()`
