# VisualMemory — Backend Architecture

> Reference doc for continuous development. Branch: `depth-perception`.
> App context: Navigation aid for blind users. Announces nearby significant objects with distance + direction.
> Last updated: Feb 2026.

---

## Project Overview

Python backend for a visual memory app built for **blind users**. The system narrates the environment by detecting and locating previously-remembered objects in a new scan image.

**Two modes:**

**Remember Mode** — User specifies an object by text prompt. System detects + crops it, embeds it, stores it. This is how "significant objects" are registered.

**Scan Mode** — User takes a photo. System detects all objects, embeds each crop, searches the database for matches. Every match = a significant object. Returns narration string per match, including depth from depth estimation.

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
├── depth_estimation/               # [NEXT] depth + spatial output module
│   ├── __init__.py
│   └── estimator.py                # DepthEstimator class + narration logic
├── object_detection/
│   ├── detect_all_objects.py       # YoloeDetector (prompt-free, all objects)
│   ├── prompt_based_detector.py    # GroundingDinoDetector (text prompt → best box)
│   └── yoloe-26l-seg-pf.pt        # Local YOLOE weights (gitignored)
├── embed_image/
│   └── embed_image.py              # DINOv3 ViT-Large embedder
├── utils/
│   ├── image_utils.py              # load_image, load_folder_images, crop_object
│   └── similarity_utils.py         # cosine_similarity, find_closest_match
├── depth_demo/                     # throwaway test suite — delete after integration
│   ├── depth_test.py               # 18-image test runner
│   ├── ground_truth.json           # ground truth distances for all test images
│   ├── test_images/                # 18 test images (gitignored)
│   └── ml-depth-pro/               # gitignored, local clone for testing
├── checkpoints/ → symlink          # gitignored, created by setup_weights.py
└── setup_weights.py                # one-time setup script
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

### `depth_estimation/estimator.py` — `DepthEstimator` [TO BUILD]
- Model: Apple Depth Pro (`apple/ml-depth-pro`)
- `__init__(focal_length_px=None)` — loads model once, stores f_px
- `estimate(image: PIL.Image) -> depth_map` — runs Depth Pro, returns tensor (H, W) in meters
- `get_depth_at_bbox(depth_map, bbox) -> float` — mean depth in inner 50% of bbox
- `get_clock_position(bbox, img_w, img_h) -> str` — "3 o'clock"
- `build_narration(label, clock, distance_ft, similarity) -> str` — final narration string
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

### Scan Mode — current
```
database folder → load_folder_images() → embed each → database_embeddings[]

query image
    → YoloeDetector.detect_all()
    → for each box: crop → embed → find_closest_match()
    → return all matched objects
```

### Scan Mode — target
```
database folder → load_folder_images() → embed each → database_embeddings[]

query image
    → YoloeDetector.detect_all()
    → DepthEstimator.estimate(image) → depth_map   ← once per image
    → for each detected box:
        → crop → embed → find_closest_match()
        → if match:
            → get_depth_at_bbox(depth_map, box) → distance_ft
            → get_clock_position(box, w, h)     → clock
            → build_narration(label, clock, distance_ft, similarity)
    → print narration per matched object
```

---

## Depth Estimation

### Model: Apple Depth Pro
- Metric depth (absolute meters) — no scale factor needed
- `f_px=None` → model infers focal length
- `f_px=tensor` → use known value (more accurate, ~26% error vs ~75% inferred at close range)
- Checkpoint: `checkpoints/depth_pro.pt` via symlink

### Focal Length
Pass from Android camera API: `f_px = (f_mm / sensor_width_mm) * image_width_px`
iPhone 15 Plus reference used to demo: `f_mm=6.24`, `sensor_width=8.64mm`.

### Official API 
```python
import depth_pro

model, transform = depth_pro.create_model_and_transforms()
model.eval()

image, _, f_px = depth_pro.load_rgb(image_path)
image = transform(image)

prediction = model.infer(image, f_px=f_px)   # f_px as torch.float32 tensor
depth = prediction["depth"]                   # metric meters, shape (H, W)
```

### Test Results (18 images, iPhone 15 Plus)
Average error: 27.8%
Floor shots result in higher error
Accuracy correlated to how straight the camera is angled
Error is acceptable. At 3ft = ~10 inches off, at 1ft = ~3 inches off.

---

## Narration Output

### Philosophy
Mix simplicity in output, inspired by Steve Jobs philosphy.
User's natural behavior will tilt the phone until they find it with the distance and clock position.

### Output (to be tuned)
```
High confidence (similarity ≥ 0.6):
    "Wallet at 3 o'clock, 2 feet away."

Low confidence (similarity 0.4–0.6):
    "May be a wallet at 3 o'clock, focus to verify."

Below threshold (similarity < 0.4):
    → do not announce
```

### Clock Position Logic
```python
angle = math.degrees(math.atan2(nx, -ny)) % 360  # 0=12 o'clock, clockwise
hour  = round(angle / 30) % 12 or 12
```

---

## Current State

- Depth estimation tested and validated (27.8% avg error, acceptable)
- All testing local image files only — no live camera, no server
- Database re-embedded on every run — intentional for simplicity during testing
- Backend called directly as Python script
- Will be exposed via Flask (teammates), then Android via KMP

---

## TODO

### Next: Build `depth_estimation/` module
- [ ] Create `depth_estimation/__init__.py`
- [ ] Create `depth_estimation/estimator.py` — `DepthEstimator` class:
  - `__init__(focal_length_px=None)` — loads Depth Pro once
  - `estimate(image: PIL.Image) -> torch.Tensor` — depth map in meters
  - `get_depth_at_bbox(depth_map, bbox) -> float` — mean depth, inner 50% bbox
  - `get_clock_position(bbox, img_w, img_h) -> str` — clock position string
  - `build_narration(label, clock, distance_ft, similarity) -> str` — narration string
- [ ] Use symlink checkpoint path (not hardcoded `ml-depth-pro/` path from depth_demo)
- [ ] Add `depth_estimation` to `pyproject.toml` packages

### Then: Integrate into Scan Mode
- [ ] Import `DepthEstimator` in `scan_mode/main.py`
- [ ] Init `DepthEstimator` with focal length
- [ ] Call `estimate()` once per query image
- [ ] Per matched object: `get_depth_at_bbox()` + `get_clock_position()` + `build_narration()`
- [ ] Print narration per matched object
- [ ] Tune confidence thresholds and demo the fully pipeline with a database

### Then: Cleanup
- [ ] Delete `depth_demo/` once integration is tested
- [ ] Commit

---

## Future Plans (Do Not Implement Yet)

### IMU Tilt Compensation
Phone IMU (accelerometer + gyroscope) reports exact tilt angle. Could compensate vertical depth when camera is angled up/down, removing the need for natural scanning. Android/iOS concern, not backend. Out of scope for TSA presentation.

### Input Enhancement — Remember Mode
Pass user query through cheap LLM (Claude Haiku, GPT-4o-mini) to expand vague descriptions before Grounding DINO.
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
- **Calibrated focal length** — 26% error vs 75% inferred; pass f_px from camera API
- **Clock position** — 12 zones, pure trig, more actionable than left/right for blind users
- **No vertical zone in narration** — user's natural scanning handles it; chest-height assumption broke for floor objects and wheelchair users
- **Depth map once per image** — reused across all matched objects
- **Straight-line distance** — what Depth Pro actually measures; consistent regardless of camera angle
- **Narration confidence signal** — "May be a wallet... focus to verify" vs silent drop; actionable not just informational
