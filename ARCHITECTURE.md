# VisualMemory Backend Architecture

> Reference doc for continuous development.
> App context: Navigation aid for blind users. Announces nearby significant objects with distance + direction.

---

## Project Overview

Python backend for a visual memory app built for blind users. The system narrates the environment by detecting and locating previously-remembered objects in a new scan image.

**Two modes:**

**Remember Mode** — User specifies an object by text prompt. System detects + crops it, embeds it, stores it (database hook ready, not yet implemented).

**Scan Mode** — User takes a photo. System detects all objects, embeds each crop, searches the database for matches. Every match = a significant object. Returns structured JSON per match, including depth and direction.

---

## Setup

Request permission for Grounding DINO on Hugging Face, then git clone and paste setup inside TSA-soft-dev-backend-2026:

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
CLIP (`openai/clip-vit-base-patch32`, ~600MB) is downloaded automatically on first use via `transformers`.

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
│   │   ├── embed_image.py              # CLIPEmbedder (image + text, shared space)
│   │   └── embed_combined.py          # make_combined_embedding()
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
├── config/settings.py                  # All tunable thresholds
├── api/                                # Reserved for Flask/FastAPI
└── tests/                             # Test scripts + test data
    ├── scripts/                        # Runnable .py test scripts
    ├── input_images/                   # Object test images
    ├── demo_database/                  # Reference crops for ScanPipeline
    └── text_demo/                      # OCR test images + ground_truth/
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
- Fields: `grounding_dino_model`, `box_threshold`, `text_threshold`, `yoloe_confidence`, `yoloe_iou`, `embedder_model`, `similarity_threshold`, `dedup_iou_threshold`, `narration_high_confidence`, `ocr_backend`, `text_similarity_threshold`

### `engine/object_detection/prompt_based.py` - `GroundingDinoDetector`
- Model: `IDEA-Research/grounding-dino-base`
- Input: PIL Image + text prompt
- Output: `{'box': [x1,y1,x2,y2], 'score': float, 'label': str}` or `None`
- Used in: Remember Mode

### `engine/object_detection/detect_all.py` - `YoloeDetector`
- Model: `yoloe-26l-seg-pf.pt` (local, prompt-free)
- Input: PIL Image
- Output: `(List[[x1,y1,x2,y2]], List[float])` or `(None, None)`
- Used in: Scan Mode

### `engine/embedding/embed_image.py` - `CLIPEmbedder`
- Model: `openai/clip-vit-base-patch32` (~600MB, auto-downloaded via `transformers`)
- `embed(image: PIL.Image) -> torch.Tensor` shape `(1, 512)` — L2-normalized image embedding
- `embed_text(text: str) -> torch.Tensor` shape `(1, 512)` — L2-normalized text embedding
- Both outputs live in the same CLIP shared embedding space
- Used in: Both modes (image via `embed()`, OCR text via `embed_text()`)

### `engine/embedding/embed_combined.py` - `make_combined_embedding`
- Concatenates normalized image + text embeddings → `(1, 1024)` combined vector
- Non-document objects pass a zero text vector — image similarity dominates unchanged
- Used in: Both pipelines

### `engine/text_recognition/paddle_recognizer.py` - `PaddleOCRRecognizer`
- Model: PaddleOCR-VL-1.5 with PP-DocLayoutV3 layout detection
- Input: PIL Image (saved to temp PNG, passed as file path)
- Output: `{"text": str, "confidence": float, "segments": list}`
- Inference: ~3-18s per image
- Extracts text from `parsing_res_list[].block_content` in JSON export

### `utils/image_utils.py`
- `load_image(path)` — PIL Image (RGB, EXIF-corrected, HEIC-compatible)
- `load_folder_images(folder)` — `List[(path, PIL Image)]`
- `crop_object(image, box)` — cropped PIL Image

### `utils/similarity_utils.py`
- `cosine_similarity(t1, t2)` — scalar tensor
- `find_match(query, database, threshold)` — `(path, score)` or `(None, 0.0)`
- `deduplicate_matches(matches, iou_threshold)` — filtered list

### `engine/depth/estimator.py` - `DepthEstimator`
- Model: Apple Depth Pro
- `__init__(focal_length_px=None)` — loads model once
- `estimate(image: PIL.Image) -> torch.Tensor` — depth map in meters
- `get_depth_at_bbox(depth_map, bbox) -> float` — mean depth in feet, inner 50% of bbox
- `get_direction(bbox, img_w) -> str` — 5-zone direction string
- `build_narration(label, direction, distance_ft, similarity) -> str | None` — final output

---

## Pipeline Flows

### Remember Mode
```
image_path + text prompt
    → load_image()
    → GroundingDinoDetector.detect(image, prompt)
    → crop_object(image, box)
    → TextRecognizer.recognize(crop)        → OCR text
    → CLIPEmbedder.embed(crop)             → image embedding (512-dim)
    → CLIPEmbedder.embed_text(ocr_text)    → text embedding (512-dim, same space)
    → make_combined_embedding()            → combined (1024-dim)
    → add_to_database()                    ← stub, awaiting DB integration
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
- `f_px=None` → model infers (~75% error at close range)
- `f_px=tensor` → calibrated (~26% error) — always use when available
- Checkpoint: `checkpoints/depth_pro.pt` — absolute path resolved via `Path(__file__)` in estimator.py

### Focal Length
```
f_px = (focalLengthMm / sensorWidthMm) * imageWidthPx
```
Pass from Android camera API. iPhone 15 Plus reference: `f_mm=6.24`, `sensor_width=8.64mm` → `f_px=3094.0`

### Test Results (18 images, iPhone 15 Plus)
- Average error: 27.8% — acceptable for navigation

---

## Narration Output

```
High confidence (similarity ≥ 0.6):             "Wallet ahead, 2 feet away."
Below high confidence (similarity_threshold–0.6): "May be a wallet slightly right, focus to verify."
Below similarity_threshold (default 0.2):         → not announced (no match returned)
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

## Test Scripts

Run from project root (`python -m visual_memory.tests.scripts.<name>`).

```bash
# Lightweight device check — runs in ~2-3 min, skips Depth Pro load
# Use this for routine verification; run_tests.py loads all models including
# Depth Pro (~2GB) which can cause memory pressure on laptops.
python -m visual_memory.tests.scripts.test_devices

# OCR benchmark — extract text from text_demo/ images, check accuracy
python -m visual_memory.tests.scripts.test_ocr_benchmark

# Integration test runner — loads ALL models including Depth Pro (~15 min, memory-heavy)
python -m visual_memory.tests.scripts.run_tests
VERBOSE=1 python -m visual_memory.tests.scripts.run_tests   # show OCR text diff

# Standalone CLI scripts
python -m visual_memory.tests.scripts.test_remember <image_path> "<prompt>"
python -m visual_memory.tests.scripts.test_scan <image_path> [--db <dir>] [--focal <f_px>]
python -m visual_memory.tests.scripts.test_text_recognition [image_path]
python -m visual_memory.tests.scripts.test_estimator

# Detector probe
python -m visual_memory.tests.scripts.probe_detector <image_path> --prompt "<prompt>"
python -m visual_memory.tests.scripts.probe_detector --batch wallet
python -m visual_memory.tests.scripts.probe_detector <image_path> --detector yoloe
```

---

## API Plan

Both pipelines return plain Python dicts — JSON-serializable, no torch tensors in output.

### POST /remember
**Input:** `image` (file), `prompt` (string), optional: `focal_length_px` (float)
**Response:**
```json
{
  "success": true,
  "message": "Object detected and embedded successfully.",
  "result": { "label": "wallet", "confidence": 0.87, "box": [120, 200, 340, 410] }
}
```

### POST /scan
**Input:** `image` (file), `focal_length_px` (float, required), optional: `database_dir` (str)
**Response:**
```json
{
  "matches": [
    { "label": "wallet", "similarity": 0.72, "distance_ft": 2.3, "direction": "ahead",
      "narration": "Wallet ahead, 2 feet away." }
  ],
  "count": 1
}
```

---

## Known Benchmark Data

### Detection score data

| model               | image                | prompt | max sigmoid       |
|---------------------|----------------------|--------|-------------------|
| grounding-dino-tiny | wallet_6ft_table.jpg | wallet | 0.1655            |
| grounding-dino-tiny | skipper.HEIC         | wallet | 0.2286 (FP)       |
| grounding-dino-base | wallet_6ft_table.jpg | wallet | 0.2036 (FN)       |
| grounding-dino-base | skipper.HEIC         | wallet | 0.3308 (FP)       |

6ft detection out of range for GDINO-base (FP > FN, no safe threshold). Use 1ft/3ft images.

### CLIP similarity data (March 2026)

Run via `python -m visual_memory.tests.scripts.benchmark_embedder`. Device: MPS (Apple M-series).

**Section A — Intra-class vs inter-class (CLIP image embeddings, full images)**

| Embedder | Intra-class sim | Inter-class sim | Discrimination ratio |
|----------|-----------------|-----------------|----------------------|
| CLIP     | 0.7688          | 0.8003          | 0.961                |

Note: Ratio < 1.0 because all 9 input images share similar context (objects on table, same lighting).
CLIP ViT-B/32 embeds context as well as object identity; intra/inter gap is small for in-context sets.

**Section A — Scan match test (cropped_wallet.png reference vs full wallet images)**

| Image                   | CLIP sim |
|-------------------------|----------|
| wallet_1ft_table.jpg    | 0.7315   |
| wallet_3ft_table.jpg    | 0.6138   |
| wallet_6ft_table.jpg    | 0.6230   |

Scores well above `similarity_threshold=0.2` at all distances. Threshold kept at **0.2**.

**Section B — CLIP text↔image cross-modal alignment (text_demo images)**

| img \ text   | marker     | pen        | pencil     | typed      |
|---|---|---|---|---|
| marker       | 0.2456 *   | 0.2456     | 0.2456     | 0.2456     |
| pen          | 0.2232     | 0.2232 *   | 0.2232     | 0.2232     |
| pencil       | 0.2142     | 0.2142     | 0.2142 *   | 0.2142     |
| typed        | 0.2245     | 0.2245     | 0.2245     | 0.2245 *   |

- Avg matched (diagonal): 0.2269 — avg mismatched: 0.2269 — gap: 0.0000
- Zero cross-modal gap expected: all 4 test images have near-identical text ("The quick brown fox…")
- `text_similarity_threshold=0.3` is above the 0.2269 matched sim → lower to 0.2 if text matching enabled

**Section C — Combined embedding similarity (image + GT text, 1024-dim)**

| | marker | pen | pencil | typed | random_printed_notes | malarkey |
|---|---|---|---|---|---|---|
| marker | 1.0000 | 0.9524 | 0.9503 | 0.9170 | 0.5046 | 0.5578 |
| pen | 0.9524 | 1.0000 | 0.9822 | 0.9131 | 0.4575 | 0.5916 |
| pencil | 0.9503 | 0.9822 | 1.0000 | 0.9050 | 0.4597 | 0.5755 |
| typed | 0.9170 | 0.9131 | 0.9050 | 1.0000 | 0.5325 | 0.5116 |
| random_printed_notes | 0.5046 | 0.4575 | 0.4597 | 0.5325 | 1.0000 | 0.5811 |
| malarkey | 0.5578 | 0.5916 | 0.5755 | 0.5116 | 0.5811 | 1.0000 |

- Off-diagonal range: 0.4575 – 0.9822 — mean: 0.6928
- marker/pen/pencil/typed cluster high (same text) — random_printed_notes and malarkey separate cleanly

---

## Current State

- Core engine complete and tested (depth, detection, embedding, OCR)
- Both pipelines produce structured JSON dicts including combined image+text embeddings
- `test_devices.py` 4/4 pass with CLIP switch (March 2026): GroundingDINO (mps), YOLOE (cpu), CLIPEmbedder (mps), PaddleOCR (cpu)
- `api/` directory empty, awaiting implementation
- Database is flat folder of images, re-embedded on each `ScanPipeline` init — temporary
- OCR backend: PaddleOCR-VL-1.5 (`settings.ocr_backend = "paddle"`)
- Embedder: CLIP ViT-B/32 (`settings.embedder_model = "openai/clip-vit-base-patch32"`, 512-dim shared image+text space)

---

## TODO

### Engine / Architecture
- [x] Switch embedder from DINOv3 to CLIP — done (March 2026); `Settings.embedder_model = "openai/clip-vit-base-patch32"`; combined embedding now 1024-dim (512+512)
- [x] Fix DepthEstimator device — was defaulting to CPU (depth_pro default); now auto-detects MPS/CUDA/CPU and passes device explicitly to `create_model_and_transforms`
- [x] Run `benchmark_embedder.py` and record similarity data in Known Benchmark Data section — done (March 2026); `similarity_threshold=0.2` kept; wallet scores 0.61–0.73 at all distances
- [ ] Add text chunking for documents longer than 77 CLIP tokens (currently truncated silently — fine for product labels, may be an issue for longer docs post-server migration)
- [ ] Dependency injection for shared model instances — `RememberPipeline` and `ScanPipeline` each instantiate their own CLIP, PaddleOCR, and (in scan) DepthEstimator. Refactor constructors to accept pre-built instances so the caller can share them (e.g. `ScanPipeline(embedder=shared_clip, recognizer=shared_ocr)`). **Post-server migration:** keep and expand — on the Flask server, models should be singletons loaded once at startup and injected into request handlers, not re-instantiated per request. The DI pattern is the same; the scope changes from "test runner" to "app lifetime".
- [ ] Implement `RememberPipeline.add_to_database()` — store combined embedding + metadata in SQLite
- [ ] Replace folder-based `load_folder_images()` + re-embed in `ScanPipeline` with DB query
- [ ] Add API layer (`api/` is empty) — POST /remember and POST /scan endpoints

### Next: Database
- [ ] Schema: `(id, label, combined_embedding BLOB, ocr_text TEXT, image_path TEXT, timestamp)`
- [ ] `add_to_database(combined, metadata)` → INSERT
- [ ] `ScanPipeline._embed_database()` → `SELECT id, label, combined_embedding FROM items`

---

## Future Plans

- **Input Enhancement in Remember Mode** — Run user prompt through a lightweight LLM to expand vague descriptions before Grounding DINO. `"remember my airpods"` → `["white round earbuds", "white round earbud case", ...]` → best detection wins.
- **GPU Server Migration** — `device='cuda'` across all models, batch processing.
- **Bloat Prevention** — Duplicate entry detection, pruning unused entries, user confirmation before overwriting similar embeddings.
- **HNSW index** — Marginal benefit below ~10k entries; defer until scale requires it.

---

## Design Decisions

- **CLIP over DINOv3 + sentence-transformers (March 2026)** — Replaced `facebook/dinov3-vitl16-pretrain-lvd1689m` (1024-dim, vision-only) + `sentence-transformers/all-MiniLM-L6-v2` (384-dim, text-only) with a single `openai/clip-vit-base-patch32` (512-dim, shared image+text space). Benefits: (1) One model replaces two — eliminates `sentence-transformers` dependency and RAM overhead of loading two separate models (~1.3GB → ~600MB total). (2) Semantically stable image embeddings across distance, lighting, and viewing angle. (3) Image and text share the same embedding space — concatenation in `make_combined_embedding` is now fully coherent (both slots are in the same CLIP space). Combined embedding shrinks from 1408-dim to 1024-dim. Benchmark data: run `python -m visual_memory.tests.scripts.benchmark_embedder` and record results here.
- **YOLOE prompt-free for scan** — Broad detection; similarity threshold handles bloat.
- **Grounding DINO for remember** — Handles vague natural language; much stronger than prompted YOLOE for this use case.
- **Depth Pro** — Only monocular model with metric (absolute) depth; no scale factor needed.
- **Actionability over accuracy** — "May be a wallet two feet to your right, focus to verify" is actionable. A raw confidence percentage is not.
- **PaddleOCR over EasyOCR** — Switched from EasyOCR to PaddleOCR-VL-1.5 (Feb 2026). EasyOCR averaged 9.7% word-overlap on the `text_demo/` test set at 200-435s/image. PaddleOCR-VL-1.5 runs in 3-18s/image (~15x faster) and correctly extracts layout-aware text from `parsing_res_list[].block_content` in its JSON output. Text extraction was initially broken due to the engine searching wrong JSON paths; fixed by targeting the correct field.
- **Combined embedding over hybrid matching** — `max(image_sim, text_sim)` could false-match two white documents with different text (image similarity alone passes threshold). The combined embedding `normalize(img) ‖ normalize(text)` means cosine similarity ≈ average of both sub-similarities, so both image AND text must match. Non-document objects get a zero text slot; image similarity dominates unchanged.
- **Single venv** — All dependencies including PaddleOCR live in one `pip install -e .` install.
