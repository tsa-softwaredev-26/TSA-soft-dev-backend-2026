# Spaitra Backend

Spaitra is split into two runtime services:

- `services/core`: the main backend API with Torch-based detection, embedding, depth, and retrieval pipelines
- `services/ocr`: a separate PaddleOCR HTTP microservice

Shared application code stays in `src/visual_memory`.

## Repository Layout

```text
services/
  core/   core runtime entrypoints
  ocr/    OCR runtime entrypoints
src/visual_memory/
  shared package used by the core backend
deploy/
  Debian systemd units and env templates
```

## Dependency Model

- Core environment: install `.[core]`
- OCR environment: install `.[ocr]`
- Shared package only: base `pip install -e .`

The core backend does not require `paddleocr` or `paddlepaddle`. The OCR service does not require Torch model packages unless you choose to install them separately.

## Local Development

### 1. Clone the repo

```bash
gh repo clone tsa-softwaredev-26/TSA-soft-dev-backend-2026
cd TSA-soft-dev-backend-2026
```

### 2. Create the core environment

```bash
python3 -m venv .venv-core
source .venv-core/bin/activate
pip install -e ".[core]"
hf auth login
python setup_weights.py
```

Use the core environment for the Flask API, model tests, and weight download.

### 3. Create the OCR environment

```bash
python3 -m venv .venv-ocr
source .venv-ocr/bin/activate
pip install -e ".[ocr]"
```

### 4. Run the OCR service

In one shell:

```bash
source .venv-ocr/bin/activate
python -m services.ocr.run
```

Default OCR address: `http://127.0.0.1:8001/ocr`

### 5. Run the core backend

In a second shell:

```bash
source .venv-core/bin/activate
export OCR_SERVICE_URL=http://127.0.0.1:8001/ocr
python -m services.core.run
```

Default core address: `http://127.0.0.1:5000`

### 6. Verify service-to-service communication

```bash
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:5000/health
curl -X POST -F image=@src/visual_memory/tests/text_demo/typed.jpeg \
  http://127.0.0.1:8001/ocr
```

If you disable OCR in the core backend, the API still runs:

```bash
ENABLE_OCR=0 python -m services.core.run
```

## Production-Style Local Run

Core backend with gunicorn:

```bash
source .venv-core/bin/activate
gunicorn -w 1 -b 127.0.0.1:5000 services.core.wsgi:application
```

OCR service with uvicorn:

```bash
source .venv-ocr/bin/activate
uvicorn services.ocr.app:app --host 127.0.0.1 --port 8001
```

## Tests

Run core integration tests from the core environment:

```bash
source .venv-core/bin/activate
python -m visual_memory.tests.scripts.run_tests
VERBOSE=1 python -m visual_memory.tests.scripts.run_tests
python -m visual_memory.tests.scripts.test_projection_head
python -m visual_memory.tests.scripts.test_scan_batching
```

OCR-related tests require the OCR service to be reachable at `OCR_SERVICE_URL`.

## Deployment

Use [DEPLOY.md](docs/DEPLOY.md) for the Debian production setup with:

- isolated `venv-core` and `venv-ocr`
- `spaitra-core.service` and `spaitra-ocr.service`
- step-by-step environment and verification commands

## Architecture

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for pipeline internals and package structure.
