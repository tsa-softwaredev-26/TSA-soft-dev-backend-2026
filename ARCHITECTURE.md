# VisualMemory — Backend Architecture

> Reference doc for continuous development. Branch: `depth-perception`.
> App context: Navigation aid for blind users. Announces nearby significant objects with distance + direction.
> Last updated: Feb 2026.

---

## Project Overview

Python backend for a visual memory app built for **blind users**. The system narrates the environment by detecting and locating previously-remembered objects in a new scan image.

**Two modes:**

**Remember Mode** — User specifies an object by text prompt. System detects + crops it, embeds it, stores it. This is how "significant objects" are registered.

**Scan Mode** — User takes a photo. System detects all objects, embeds each crop, searches the database for matches. Every match = a significant object the user cares about. Returns distance + spatial position for each, narrated aloud.

> Depth estimation is part of Scan Mode, not a separate mode.

---

## Setup

```bash
pip install -e .
python setup_weights.py
```

`pip install -e .` installs all dependencies including depth-pro from GitHub.
`setup_weights.py` downloads the Depth Pro checkpoint (~2GB) and creates a symlink at `checkpoints/` so `create_model_and_transforms()` works from any directory without modifying Apple's source.
`checkpoints/` is gitignored — every teammate runs this once after cloning.

---

## Directory Structure

```
VisualMemory/
├── remember_mode/
│   └── main.py                     # Prompt → detect → crop → embed pipeline
├── scan_mode/
│   └── main.py                     # detect all → embed → match → depth + narration
├── object_detection/
│   ├── detect_all_objects.py       # YoloeDetector (prompt-free, all objects)
│   ├── prompt_based_detector.py    # GroundingDinoDetector (text prompt → best box)
│   └── yoloe-26l-seg-pf.pt        # Local YOLOE weights
├── embed_image/
│   └── embed_image.py              # DINOv3 ViT-Large embedder
├── utils/
│   ├── image_utils.py              # load_image, load_folder_images, crop_object
│   └── similarity_utils.py         # cosine_similarity, find_closest_match
├── depth_demo/                     # [CURRENT] Depth Pro testing — throwaway, will become depth_estimation/
│   ├── depth_test.py
│   ├── input_images/
│   └── ml-depth-pro/               # gitignored, cloned for local testing only
├── checkpoints/ → symlink          # gitignored, created by setup_weights.py
├── setup_weights.py                # one-time setup script
└── old_depth_demo/                 # old experiments, reference only
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
- Returns all detections above threshold (`conf=0.35`, `iou=0.45`)
- Used in: Scan Mode

### `embed_image/embed_image.py` — `ImageEmbedder`
- Model: `facebook/dinov3-vitl16-pretrain-lvd1689m`
- Input: PIL Image
- Output: `torch.Tensor` shape `(1, embedding_dim)`
- `device_map="auto"` handles MPS/CUDA automatically
- Used in: Both modes

### `utils/image_utils.py`
- `load_image(path)` → PIL Image (RGB, EXIF-corrected, HEIC-compatible)
- `load_folder_images(folder)` → `List[(path, PIL Image)]`
- `crop_object(image, box)` → cropped PIL Image

### `utils/similarity_utils.py`
- `cosine_similarity(t1, t2)` → scalar tensor
- `find_closest_match(query, database, threshold)` → `(path, score)` or `(None, 0.0)`
- Linear search, fine for expected database size (<5k items)

---

## Mode Pipelines

### Remember Mode
```
text prompt
    → GroundingDinoDetector.detect(image, prompt)
    → crop_object(image, box)
    → ImageEmbedder.embed(crop)     ← commented out, not yet storing
    → show cropped object
```

### Scan Mode — current
```
database folder → load_folder_images() → ImageEmbedder.embed() each → database_embeddings[]

query image
    → YoloeDetector.detect_all()
    → for each box: crop → embed → find_closest_match()
    → return all matched objects
```

### Scan Mode — target (with depth + narration)
```
[same through matching]
    → run DepthEstimator.estimate(image) once → depth_map
    → for each matched object:
        → get_depth_at_bbox(depth_map, box)   → straight-line distance (meters)
        → get_clock_position(box, w, h)        → "3 o'clock"
        → get_vertical_zone(box, h)            → "knee level"
        → get_spatial_offsets(box, w, h, dist) → horizontal + vertical distances via trig
    → narration string per object:
        "Wallet at 3 o'clock, 3 feet away, knee level"
```

> Depth map computed once per image, reused for all matched objects.

---

## Depth Estimation

### Model: Apple Depth Pro
- Outputs metric depth (absolute meters) — no calibration scale factor needed
- `f_px=None` → model infers focal length from image
- `f_px=tensor` → use known focal length (more accurate when available)
- ~2GB checkpoint, downloaded via `setup_weights.py`

### Test Results (42cm airpod case, iPhone 15 Plus)
| Mode | Focal px | Mean depth | Error |
|------|----------|------------|-------|
| Inferred | 4290.6 | 0.735m | 75.0% |
| Calibrated (iPhone 15 Plus) | 3094.0 | 0.530m | 26.2% |

**Calibrated wins.** iPhone 15 Plus specs: `f_mm=6.24`, `sensor_width=8.64mm`.
`f_px = (f_mm / sensor_width_mm) * image_width_px`

Still need to test at 1ft, 3ft, 6ft to confirm calibrated consistently beats inferred across distances before committing to it in the pipeline.

### Depth Pro API (official pattern — do not deviate)
```python
import depth_pro

model, transform = depth_pro.create_model_and_transforms()
model.eval()

image, _, f_px = depth_pro.load_rgb(image_path)
image = transform(image)

prediction = model.infer(image, f_px=f_px)
depth = prediction["depth"]            # metric meters, shape (H, W)
focallength_px = prediction["focallength_px"]
```

### Checkpoint Path (temp, during depth_demo testing)
```python
# TODO: DELETE when moving to depth_estimation module — use symlink instead
config = depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT
config.checkpoint_uri = Path(__file__).parent / "ml-depth-pro" / "checkpoints" / "depth_pro.pt"
model, transform = depth_pro.create_model_and_transforms(config=config)
```

---

## Spatial Output Design

### Design Philosophy
Complex backend, dead simple narration. One number, one direction, one height word.
The user is processing audio while moving — cognitive load is the enemy.

### Camera Assumption
User holds phone at chest height pointing forward.
This is a stated design constraint (same as BlindSquare, Google Maps audio nav).
Makes vertical positioning self-calibrating — no need to know user height.
Works identically for wheelchair users since position is relative to camera, not ground.

### Output Components
All derived from bbox center + Depth Pro straight-line distance:

**1. Straight-line distance** — from Depth Pro, metric meters converted to feet.

**2. Clock position** — horizontal direction, 12 zones, pure trig on bbox center:
```python
angle = math.degrees(math.atan2(nx, -ny)) % 360  # 0 = 12 o'clock, clockwise
hour = round(angle / 30) % 12 or 12
```

**3. Vertical zone** — camera-relative height from bbox center y position:
```
0–15%  → overhead
15–35% → high / shelf
35–50% → chest
50–65% → waist
65–80% → knee
80–100%→ floor
```

**4. Spatial offsets (raw, for testing)** — derived via trig from straight-line distance + bbox angles:
```
horizontal_distance = straight_line * sin(horizontal_angle)
vertical_distance   = straight_line * sin(vertical_angle)
```

### Final Narration String
```
"Wallet at 3 o'clock, 3 feet away, knee level"
```

### Example (full internal output for testing)
```python
{
    "label": "wallet",
    "straight_line_m": 0.91,
    "straight_line_ft": 3.0,
    "clock_position": "3 o'clock",
    "vertical_zone": "knee level",
    "horizontal_offset_ft": 1.5,   # raw, testing only
    "vertical_offset_ft": 0.8,     # raw, testing only
    "narration": "Wallet at 3 o'clock, 3 feet away, knee level"
}
```

---

## TODO: Depth Integration

### Testing (depth_demo — in progress)
- [x] `depth_test.py` running, calibrated at 26.2% error at 42cm
- [ ] Test at 1ft (30cm), 3ft (91cm), 6ft (183cm) — confirm calibrated beats inferred at all distances
- [ ] Add spatial output: clock position, vertical zone, trig offsets
- [ ] Print full example output dict + narration string

### Build `depth_estimation/` module (after testing)
- [ ] `DepthEstimator` class — loads model once, exposes `estimate()` and `get_depth_at_bbox()`
- [ ] `utils/direction_utils.py` — `get_clock_position()`, `get_vertical_zone()`, `get_spatial_offsets()`
- [ ] Replace hardcoded checkpoint path with symlink (`checkpoints/` at project root)
- [ ] Remove `depth_demo/` once module is complete

### Integrate into Scan Mode
- [ ] `DepthEstimator.estimate(image)` once per query image
- [ ] Per matched object: depth at bbox + clock position + vertical zone → narration string
- [ ] Final output dict per object (see example above)

---

## Current State

- All testing local image files only — no live camera, no server
- Database re-embedded on every run (intentional for simplicity during testing)
- Backend called directly as Python script
- Will be exposed via Flask (teammates), then Android via KMP

---

## Future Plans (Do Not Implement Yet)

### IMU Tilt Compensation
The phone's IMU (accelerometer + gyroscope) reports the exact tilt angle of the device. This could be used to compensate vertical depth calculations when the user points the camera up or down instead of straight ahead — removing the chest-height assumption entirely. This is an Android/iOS concern, not backend. Out of scope for TSA presentation.

### Input Enhancement — Remember Mode
Pass user query through a cheap LLM (Claude Haiku, GPT-4o-mini) to expand vague descriptions before running Grounding DINO.
- Input: `"remember my airpods"`
- LLM output: `["white round earbuds", "white round earbud case", "round earbud case"]`
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
- **YOLOE prompt-free for scan** — broad detection, let similarity handle disambiguation
- **Grounding DINO for remember** — handles vague natural language well
- **Depth Pro** — only monocular model that outputs metric (absolute) depth without calibration
- **Calibrated focal length over inferred** — 26.2% vs 75% error at close range; pass f_px from camera metadata when available
- **No phone spec index** — too fragile across Android variants; use camera API directly
- **Clock position** — 12 zones, pure trig, more actionable than left/right for blind users
- **Camera-relative vertical zones** — works for any user height or wheelchair, no height input needed
- **Chest-height assumption** — stated design constraint, same as industry standard nav apps
- **Depth map once per image** — reused across all matched objects
- **Straight-line distance in narration** — most intuitive for a blind user navigating toward an object
- **HEIC support** via `pillow-heif` in `load_image()`
