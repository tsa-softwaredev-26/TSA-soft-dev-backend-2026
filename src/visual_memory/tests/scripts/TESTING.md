# Testing Guide

## Test media layout

- A compact uncompressed core set remains in `src/visual_memory/tests/input_images/`:
  - `wallet_1ft_table.jpg`
  - `wallet_3ft_table.jpg`
  - `magnesium.heic`
- The remaining demo/validation media is stored as zip bundles in:
  - `src/visual_memory/tests/media_archives/input_images_extra_part*.zip`
  - `src/visual_memory/tests/media_archives/text_demo_images.zip`
  - `src/visual_memory/tests/media_archives/validation_images_part*.zip`
- To restore full local media, unzip from `src/visual_memory/tests/` so paths match test defaults.

## Fast test suite (no models, < 30s)

```bash
# All unit + API tests (default)
python -m visual_memory.tests.scripts.run_all

# Unit only: pure logic, utils, DB - no Flask (~5s)
python -m visual_memory.tests.scripts.run_all --suite unit

# API only: every endpoint via Flask test client + stubs (~25s)
python -m visual_memory.tests.scripts.run_all --suite api

# Filter by feature tag
python -m visual_memory.tests.scripts.run_all --tag remember,scan

# Verbose output (shows response bodies on failure)
TEST_VERBOSITY=2 python -m visual_memory.tests.scripts.run_all

# Stop on first failure
python -m visual_memory.tests.scripts.run_all --fail-fast

# Run a single module
python -m visual_memory.tests.scripts.test_remember_route
python -m visual_memory.tests.scripts.test_scan_route
python -m visual_memory.tests.scripts.test_feedback_retrain
python -m visual_memory.tests.scripts.test_items_crud
```

Run all commands from project root. No model loading required.

---

## System tests (real models)

```bash
# Full test suite (all core models, plus OCR HTTP calls when ENABLE_OCR=1)
python -m visual_memory.tests.scripts.run_tests

# Fast mode - skips scan OCR; 2 OCR calls total (~3-5 min)
FAST=1 python -m visual_memory.tests.scripts.run_tests

# With depth estimation (loads 2GB model - skipped by default)
DEPTH=1 python -m visual_memory.tests.scripts.run_tests

# Show OCR text vs ground truth
VERBOSE=1 python -m visual_memory.tests.scripts.run_tests

# Combine flags
FAST=1 VERBOSE=1 python -m visual_memory.tests.scripts.run_tests
```

---

## Integration tests (run_tests.py)

### Test 1 - RememberPipeline
- Input: `input_images/wallet_1ft_table.jpg`, prompt `"small rectangular wallet"`
- Runs: GroundingDINO detect + DINOv3 embed + OCR HTTP call (wallet crop, usually no text)
- Pass: detection returns a box and label

### Test 2 - ScanPipeline
- Input: `input_images/wallet_3ft_table.jpg`, database: `demo_database/` (6 reference crops)
- Runs: YOLOE detect + DINOv3 embed + similarity match
- OCR: runs on all 6 database images + query (skipped in `FAST=1` mode)
- Pass: at least 1 match with similarity >= 0.2

### Test 3 - DepthEstimator
- Skipped unless `DEPTH=1`
- Verifies `scan.estimator` is loaded (reuses the instance from Test 2)
- Loads Apple Depth Pro (~2GB) - avoid on machines with limited VRAM

### Test 4 - TextRecognizer + CLIPTextEmbedder
- Input: `input_images/magnesium.heic` (supplement label with readable text)
- Runs: OCR HTTP recognize + CLIP text embed
- `FAST=1`: still requires the OCR service for the remaining OCR call
- Pass: valid OCR result dict + embedding shape `(1, 512)`

---

## FAST=1 OCR budget

| Test | Default | FAST=1 |
|------|---------|--------|
| 1 remember (wallet crop) | 1 OCR, no_segments | 1 OCR, no_segments |
| 2 scan DB embed (6 images) | 6 OCRs | 0 (OCR disabled) |
| 2 scan query | 1 OCR | 0 (OCR disabled) |
| 4 magnesium | 1 OCR | 1 OCR (registry reuse) |
| **Total** | **9 OCRs** | **2 OCRs** |

---

## Manual / probe scripts

### test_remember.py - single RememberPipeline run
```bash
python -m visual_memory.tests.scripts.test_remember <image> "<prompt>"
# Example:
python -m visual_memory.tests.scripts.test_remember tests/input_images/wallet_1ft_table.jpg "wallet"
```

### test_scan.py - single ScanPipeline run
```bash
python -m visual_memory.tests.scripts.test_scan <image>
# Example:
python -m visual_memory.tests.scripts.test_scan tests/input_images/wallet_3ft_table.jpg
```

### test_estimator.py - depth accuracy test
```bash
python -m visual_memory.tests.scripts.test_estimator
# Requires DEPTH=1 equivalent - loads Depth Pro. Do not run on battery or limited VRAM.
```

### test_text_recognition.py - OCR + embedding test
```bash
python -m visual_memory.tests.scripts.test_text_recognition
```

---

## HuggingFace auth

DINOv3 is gated. If the model is already cached but auth fails at startup, run with:

```bash
TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m visual_memory.tests.scripts.run_tests
```

Token is read from `~/.cache/huggingface/token` or `HF_TOKEN` env var.

---

## Log inspection

```bash
# Tail recent log entries
python -c "from visual_memory.utils import tail_logs; [print(e) for e in tail_logs(n=20)]"

# Filter by event type
python -c "from visual_memory.utils import tail_logs; [print(e) for e in tail_logs(event='scan_text_match')]"
```

Log file: `logs/app.log` (project root, JSON Lines, gitignored).

---

## Voice evaluation dataset

Use one shared dataset file:
- `src/visual_memory/tests/input_data/voice_eval_dataset.json`

Place recordings here (gitignored):
- `src/visual_memory/tests/input_audio/`

Supported audio extensions by case id (case-insensitive stem match):
- `.m4a`, `.ogg`, `.webm`, `.wav`, `.mp3`

The dataset is maintained for voice route and websocket coverage in the tracked suite.
