# Spaitra Backend

Visual memory backend for the Spaitra iOS app. Teaches the system to recognize
objects, then scans scenes to find them and report distance + direction.

---

## Local Development

For running tests and iterating on models locally (Mac or Linux workstation).
Does not require a GPU - MPS (Apple Silicon) or CPU works.

**Prerequisites:**
- Python >= 3.10
- HuggingFace account with access to both gated models (request access before cloning):
  - [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base)
  - [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
- `gh` CLI installed and authenticated (`gh auth login`)

```bash
gh repo clone tsa-softwaredev-26/TSA-soft-dev-backend-2026
cd TSA-soft-dev-backend-2026
python -m venv venv
source venv/bin/activate
pip install -e .
hf auth login
python setup_weights.py
```

Models download on first run (~5GB total, 10-30 min). Subsequent runs load from cache.

To run the API locally:
```bash
python src/visual_memory/api/run.py
# Starts Flask on 127.0.0.1:5000
```

---

## Server Deployment

For the production Flask/gunicorn API on a Debian server with optional GPU.
See [DEPLOY.md](DEPLOY.md) for full step-by-step instructions.

**Requirements:**
- Debian 12 (Bookworm) or 13 (Trixie)
- NVIDIA GPU with >=8GB VRAM recommended (CUDA 11.8 or 12.x)
- CPU-only is supported; OCR runs in an external service configured via `OCR_SERVICE_URL`

Key steps: install system packages, set up SSH deploy key or PAT for GitHub access,
create a `spaitra` service user, install dependencies, configure `.env`, install the
systemd service, and run a srv.us tunnel for public HTTPS access.

---

## Running Tests

```bash
# Full integration test - run after any engine or pipeline change
python -m visual_memory.tests.scripts.run_tests

# Show OCR text vs ground truth
VERBOSE=1 python -m visual_memory.tests.scripts.run_tests

# CPU-only tests (no model loading, run any time)
python -m visual_memory.tests.scripts.test_projection_head
python -m visual_memory.tests.scripts.test_scan_batching
```

First run loads all models (~2-3 min). Tests 3 and 4 reuse model instances from
test 2, so the full suite is faster than loading each separately.

---

## Modes

- **Teach Mode** - detect and store an object from a photo + text label
- **Scan Mode** - find all remembered objects in a new scene, return distance + direction
- **Ask Mode** - query where a known object was last seen

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for module layout, pipeline internals,
API contract, and tuning reference.
