# Spaitra Deployment Guide

Target: **Debian 12 or newer**, Python 3.11 or newer, optional NVIDIA GPU.

Two services run on the same host and communicate over localhost:

| Service | Runtime | Port |
|---|---|---|
| `spaitra-core` | Flask + gunicorn, Torch models, SQLite | 5000 |
| `spaitra-ocr` | FastAPI + uvicorn, PaddleOCR | 8001 |

---

## Quick-start checklist

- [ ] System packages installed
- [ ] `spaitra` service user created
- [ ] Repo cloned
- [ ] `venv-core` created, `pip install -e ".[core]"` done
- [ ] `venv-ocr` created, `pip install -e ".[ocr]"` done
- [ ] GPU PyTorch installed (GPU hosts only)
- [ ] HuggingFace authenticated, `setup_weights.py` completed (includes Ollama model pull)
- [ ] Ollama daemon installed and running (optional; API degrades gracefully without it)
- [ ] `/opt/spaitra/.env` created and edited
- [ ] `/opt/spaitra/.ocr.env` created and edited
- [ ] `API_KEY` generated and saved securely
- [ ] `DB_ENCRYPTION_KEY` generated and saved securely (enable on Python <= 3.12)
- [ ] Smoke tests pass (manual service run + curl)
- [ ] systemd units installed and enabled
- [ ] `deploy/secure_permissions.sh` applied
- [ ] Auto-retrain cron installed
- [ ] Public tunnel running (`spaitra-tunnel.service`)
- [ ] End-to-end health check passes

---

## 1. System packages

```bash
apt-get update
apt-get install -y \
    python3 python3-venv python3-dev \
    git gh build-essential curl \
    libsqlcipher-dev \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgomp1 \
    libjpeg-dev libpng-dev libtiff-dev libwebp-dev \
    ffmpeg
```

> **Python version:** any Python ≥ 3.11 works. On Debian 12+ `python3` is 3.11.
> On Debian 13 it may be 3.12 or 3.13; that is fine. Do not pin to 3.11 specifically.

**GPU hosts only:** make sure NVIDIA drivers are installed and `nvidia-smi` works before
continuing. The PyTorch GPU wheel is installed in step 4.

---

## 2. Service user

```bash
adduser --system --group --home /opt/spaitra spaitra
mkdir -p /opt/spaitra
chown spaitra:spaitra /opt/spaitra
```

---

## 3. Clone the repository

Run as the `spaitra` user. Use a GitHub PAT (recommended) or an SSH deploy key.

### PAT flow (recommended)

Create a PAT at github.com/settings/tokens with `repo` scope. Then:

```bash
su - spaitra -s /bin/bash
cd /opt/spaitra
echo "https://YOUR_GITHUB_USERNAME:YOUR_PAT@github.com" > ~/.git-credentials
chmod 640 ~/.git-credentials
git config --global credential.helper store
git clone https://github.com/tsa-softwaredev-26/TSA-soft-dev-backend-2026.git
```

> The clone creates `/opt/spaitra/TSA-soft-dev-backend-2026/`. All paths below
> assume this **subdirectory layout**. Adjust `REPO` if you cloned elsewhere.

Set a shell variable for convenience; used throughout this guide:

```bash
export REPO=/opt/spaitra/TSA-soft-dev-backend-2026
```

### SSH deploy key flow (alternative)

```bash
su - spaitra -s /bin/bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
ssh-keygen -t ed25519 -C "spaitra-server" -f ~/.ssh/deploy_key -N ""
cat ~/.ssh/deploy_key.pub   # add this to GitHub → repo → Settings → Deploy keys
cat >> ~/.ssh/config <<'EOF'
Host github.com
    IdentityFile /opt/spaitra/.ssh/deploy_key
    StrictHostKeyChecking accept-new
EOF
chmod 600 ~/.ssh/config
cd /opt/spaitra
git clone git@github.com:tsa-softwaredev-26/TSA-soft-dev-backend-2026.git
export REPO=/opt/spaitra/TSA-soft-dev-backend-2026
```

### Non-interactive git after setup

If you need a non-spaitra user (e.g. a dev account) to run git operations:

```bash
# Make the credentials file group-readable:
chmod 640 /opt/spaitra/.git-credentials
# Run git with spaitra's home so the credential helper finds it:
sg spaitra -c "HOME=/opt/spaitra git -C $REPO -c safe.directory=$REPO pull"
```

---

## 4. Python environments

All commands run as `spaitra` from inside `$REPO`.

```bash
su - spaitra -s /bin/bash
export REPO=/opt/spaitra/TSA-soft-dev-backend-2026
cd $REPO
```

### Core backend

```bash
python3 -m venv venv-core
source venv-core/bin/activate
pip install --upgrade pip
pip install -e ".[core]"
deactivate
```

**GPU hosts; install the matching PyTorch CUDA wheel:**

```bash
source $REPO/venv-core/bin/activate
bash $REPO/deploy/install_gpu_deps.sh
deactivate
```

The script reads `nvidia-smi` to detect the CUDA version and selects the right
PyTorch index automatically. It supports CUDA 11.x through 13.x and will error
clearly if the driver is missing or the CUDA version is unsupported.

**CPU-only hosts:**

```bash
source $REPO/venv-core/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
deactivate
```

### OCR service

```bash
python3 -m venv venv-ocr
source venv-ocr/bin/activate
pip install --upgrade pip
pip install -e ".[ocr]"
deactivate
```

> Keep the two venvs fully isolated. Do not cross-install OCR packages into
> `venv-core` or Torch into `venv-ocr`.

---

## 5. HuggingFace auth and model weights

The core backend uses two gated HuggingFace models. Before downloading:

1. Accept the license for `facebook/dinov3-vitl16-pretrain-lvd1689m` at
   `https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m`
2. Accept the license for `IDEA-Research/grounding-dino-base` at
   `https://huggingface.co/IDEA-Research/grounding-dino-base`

Then authenticate and download (~5.3 GB total):

```bash
source $REPO/venv-core/bin/activate
hf auth login              # paste your HF token when prompted
cd $REPO
python setup_weights.py    # downloads Depth Pro, YOLOE, CLIP, DINOv3, GDino
deactivate
```

Weight locations after download:

```
$REPO/checkpoints/depth_pro.pt                           (~2 GB)
$REPO/src/visual_memory/engine/object_detection/yoloe-26l-seg-pf.pt  (~80 MB)
~/.cache/huggingface/hub/                                (CLIP, DINOv3, GDino)
```

---

## 6. Ollama (optional; enables natural language search)

Ollama powers the `/ask` and `/item/ask` natural language query parsing. The API
degrades gracefully without it (embedding-only search, keyword intent detection).

### Install Ollama daemon

```bash
curl -fsSL https://ollama.com/install.sh | sh
# Verify:
ollama --version
systemctl status ollama   # should be active/running after install
```

### Pull the model

`setup_weights.py` handles this automatically if `ollama` is in `$PATH`:

```bash
source $REPO/venv-core/bin/activate
python setup_weights.py   # step 6/6 pulls llama3.2:1b (~1.3 GB)
deactivate
```

To pull manually:

```bash
ollama pull llama3.2:1b
ollama list   # verify model appears
```

### Verify it works

```bash
ollama run llama3.2:1b "respond with JSON: {\"intent\": \"find\"}"
# Should print a JSON response and exit cleanly.
```

### Troubleshooting

| Symptom | Fix |
|---|---|
| `ollama: command not found` | Re-run install script; confirm `/usr/local/bin/ollama` exists |
| `pull failed: connection refused` | `systemctl start ollama` |
| Slow /ask responses | Lower `OLLAMA_TIMEOUT_SECONDS` in `.env` (default: 5.0 s) |
| Remote Ollama host | Set `OLLAMA_HOST=http://<host>:11434` in `.env` |

The backend circuit breaker opens after 3 consecutive Ollama failures and pauses
all LLM calls for 60 seconds. This prevents a stalled daemon from adding latency
to every request. The API continues to work in embedding-only mode during that window.

Security note: `/ask` applies an unsafe-query gate before retrieval. Queries with
prompt-injection or harmful markers return HTTP 400 with `blocked: true` and
`reason: "unsafe_query"` instead of running semantic match.

---

## 7. VLM item description (moondream2)

`POST /item/ask` now supports live `describe` responses with a 3-tier fallback:

1. moondream2 VLM (`method: "vlm"`)
2. stored visual attributes from teach-time analysis (`method: "attributes"`)
3. minimal label and OCR summary (`method: "minimal"`)

moondream2 is loaded from HuggingFace via `transformers` and `moondream==0.2.0`.
This path is local and does not use Ollama.

Performance-mode behavior for describe:
- fast: VLM disabled, fallback uses attributes or minimal
- balanced: VLM enabled with shorter timeout
- accurate: VLM enabled with longer timeout

---

## 8. Environment files

### Core backend; `/opt/spaitra/.env`

```bash
cp $REPO/deploy/env.example /opt/spaitra/.env
nano /opt/spaitra/.env
```

Required changes:
- Set `API_KEY` to a long random string (e.g. `openssl rand -hex 32`)
- Set `DB_ENCRYPTION_KEY` to a long random string (e.g. `openssl rand -hex 32`) when running Python <= 3.12
- Keep both keys in a secure backup location before restarting services

GPU VRAM guide:

| GPU VRAM | `ENABLE_DEPTH` | `SAVE_VRAM` | Notes |
|---|---|---|---|
| ≥ 8 GB | `1` | `0` | All models warm on GPU, no swap overhead |
| 6-8 GB | `1` | `1` | Swaps GDino↔YOLOE+Depth between calls (~1-5 s per swap) |
| < 6 GB | `0` | `0` | Disable depth; depth model alone is ~2 GB |
| CPU-only | `0` | `0` | No GPU, all inference on CPU |

Full reference; all supported variables:

```dotenv
# Auth
API_KEY=replace-with-a-long-random-api-key
# Enable SQLCipher at rest encryption on Python <= 3.12.
# On Python 3.13 this is currently unavailable because pysqlcipher3 fails to build.
# Leave empty on Python 3.13+ until SQLCipher Python bindings catch up.
DB_ENCRYPTION_KEY=

# Bind address (keep 127.0.0.1; expose via tunnel or reverse proxy)
API_HOST=127.0.0.1
API_PORT=5000

# Feature flags
ENABLE_DEPTH=1          # Depth Pro distance estimation (needs ~2 GB VRAM)
ENABLE_OCR=1            # OCR via the spaitra-ocr microservice
ENABLE_DEDUP=1          # Deduplicate overlapping scan detections
ENABLE_LEARNING=1       # Apply personalized projection head during scan

# OCR microservice
OCR_SERVICE_URL=http://127.0.0.1:8001/ocr
OCR_TIMEOUT_SECONDS=10.0

# VRAM management (enable on GPUs < 8 GB)
# Swaps model weights between GPU and CPU RAM between pipeline calls.
# Requires ~16 GB system RAM. See ARCHITECTURE.md → VRAM Management.
SAVE_VRAM=0

# Ollama LLM (optional; enables natural language /ask and /item/ask parsing)
# Leave OLLAMA_HOST commented out to use the default localhost:11434.
# OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_TIMEOUT_SECONDS=5.0
```

### OCR service; `/opt/spaitra/.ocr.env`

```bash
cp $REPO/deploy/ocr.env.example /opt/spaitra/.ocr.env
# defaults are fine for a single-host deployment; no changes required
```

Reference:

```dotenv
OCR_HOST=127.0.0.1
OCR_PORT=8001
OCR_LANG=en             # language pack; en covers most Latin-script text
OCR_MIN_CONFIDENCE=0.3  # drop OCR segments below this score
OCR_USE_ANGLE_CLS=0     # angle correction; keep 0 unless text is frequently rotated
OCR_ENABLE_MKLDNN=0     # keep 0 on CPU hosts to avoid Paddle runtime crash
OCR_MAX_SIDE=1280       # pre-resize large images to keep OCR memory bounded
OCR_REQUEST_TIMEOUT_SECONDS=40
OCR_MAX_CONCURRENCY=1
OCR_TIMEOUT_KEEP_ALIVE_SECONDS=5
OCR_THROTTLE_RETRY_AFTER_SECONDS=2
OCR_RATE_LIMIT_PER_MIN=120
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
API_KEY=replace-with-the-same-core-api-key
```

---

## 8. Smoke test (before systemd)

Test each service manually before handing off to systemd.

### OCR service

```bash
cd $REPO
source venv-ocr/bin/activate
set -a; . /opt/spaitra/.ocr.env; set +a
python -m services.ocr.run &
sleep 3

curl http://127.0.0.1:8001/health
curl -s -o /dev/null -w "No key -> %{http_code}\n" \
  -X POST -F image=@$REPO/src/visual_memory/tests/text_demo/typed.jpeg \
  http://127.0.0.1:8001/ocr
KEY=$(grep ^API_KEY= /opt/spaitra/.env | cut -d= -f2-)
curl -s -X POST -F image=@$REPO/src/visual_memory/tests/text_demo/typed.jpeg \
  -H "X-API-Key: $KEY" \
  http://127.0.0.1:8001/ocr | python3 -m json.tool

kill %1
deactivate
```

### Core service

```bash
cd $REPO
source venv-core/bin/activate
set -a; . /opt/spaitra/.env; set +a
python -m services.core.run &
sleep 10   # models load on startup; allow extra time on first run

curl http://127.0.0.1:5000/health
# Test with API key:
KEY=$(grep ^API_KEY= /opt/spaitra/.env | cut -d= -f2-)
curl -s http://127.0.0.1:5000/debug/state -H "X-API-Key: $KEY" | python3 -m json.tool

kill %1
deactivate
```

---

## 9. Install systemd units

The service files in `deploy/` assume the flat layout (`/opt/spaitra/venv-core`).
If your venvs are inside the repo subdir, edit the two paths before copying:

```bash
# For subdirectory layout; create adjusted copies:
sed 's|/opt/spaitra/venv-core|'"$REPO"'/venv-core|g; s|/opt/spaitra/venv-ocr|'"$REPO"'/venv-ocr|g; s|WorkingDirectory=/opt/spaitra$|WorkingDirectory='"$REPO"'|' \
  $REPO/deploy/spaitra-core.service > /etc/systemd/system/spaitra-core.service

sed 's|/opt/spaitra/venv-ocr|'"$REPO"'/venv-ocr|g; s|WorkingDirectory=/opt/spaitra$|WorkingDirectory='"$REPO"'|' \
  $REPO/deploy/spaitra-ocr.service > /etc/systemd/system/spaitra-ocr.service

# For flat layout; copy directly:
# cp $REPO/deploy/spaitra-core.service /etc/systemd/system/
# cp $REPO/deploy/spaitra-ocr.service /etc/systemd/system/

systemctl daemon-reload
systemctl enable --now spaitra-ocr
systemctl enable --now spaitra-core

systemctl status spaitra-ocr spaitra-core --no-pager
journalctl -u spaitra-core -n 50 --no-pager

# Harden file permissions once env files and services are in place
sudo -u spaitra bash $REPO/deploy/secure_permissions.sh "$REPO"
```

> **File ownership after git pull:** if you ran `git pull` as root, source files may
> be owned by root and unreadable by the service. Fix with:
> `chown -R spaitra:spaitra $REPO/src $REPO/services`

---

## 10. Auto-retrain cron

The projection head personalizes scan results as the user gives feedback (confirm /
correct detections). Once enough feedback accumulates (default: 10 triplets), a nightly
cron job triggers retraining via the `/retrain` API endpoint.

Create the retrain script:

```bash
mkdir -p /opt/spaitra/bin
cat > /opt/spaitra/bin/auto_retrain.sh <<'SCRIPT'
#!/usr/bin/env bash
# Trigger projection head retraining via the core API.
# Runs nightly via cron. Exits 0 whether training ran or was skipped
# (insufficient data). Exits 1 only on a hard connection failure.
set -euo pipefail

API_KEY_FILE=/opt/spaitra/.env
API_BASE=http://127.0.0.1:5000

KEY=$(grep '^API_KEY=' "$API_KEY_FILE" | cut -d= -f2-)
RESP=$(curl -sf -X POST "$API_BASE/retrain" \
    -H "X-API-Key: $KEY" 2>&1) || {
    echo "[auto_retrain] ERROR: curl failed; is spaitra-core running?" >&2
    exit 1
}
echo "[auto_retrain] $(date -Iseconds) $RESP"
SCRIPT

chown spaitra:spaitra /opt/spaitra/bin /opt/spaitra/bin/auto_retrain.sh
chmod 750 /opt/spaitra/bin/auto_retrain.sh
```

Create the log directory:

```bash
mkdir -p $REPO/logs
chown spaitra:spaitra $REPO/logs
```

Install the crontab for the `spaitra` user:

```bash
# Write to a temp file first, then install atomically:
cat > /tmp/spaitra-crontab <<'CRON'
# Spaitra scheduled jobs
# Trigger projection head retraining nightly at 2 AM.
# The /retrain endpoint silently skips if fewer than 10 feedback triplets exist.
# Output is appended to logs/retrain.log for debugging.
0 2 * * * /opt/spaitra/bin/auto_retrain.sh >> /opt/spaitra/TSA-soft-dev-backend-2026/logs/retrain.log 2>&1
CRON

crontab -u spaitra /tmp/spaitra-crontab
crontab -u spaitra -l   # verify
```

Test the script manually before waiting for the nightly run:

```bash
su - spaitra -s /bin/bash -c "/opt/spaitra/bin/auto_retrain.sh"
# Expected output: {"started": false, "reason": "insufficient_data", ...}
# This is correct; it means the script runs and the API responds.
# Training will start automatically once 10+ feedback triplets accumulate.
```

---

## 11. Public access with srv.us

`srv.us` creates a stable HTTPS public URL that forwards to the local gunicorn port.
No account or configuration required.

### Install srv.us

```bash
curl -fsSL https://install.srv.us | bash
# verify:
srv.us --version
```

### Install as a systemd service

```bash
cat > /etc/systemd/system/spaitra-tunnel.service <<'EOF'
[Unit]
Description=Spaitra srv.us tunnel
After=spaitra-core.service
Requires=spaitra-core.service

[Service]
Type=simple
User=spaitra
ExecStart=/usr/local/bin/srv.us --port 5000
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now spaitra-tunnel
```

Find the assigned public hostname:

```bash
journalctl -u spaitra-tunnel -n 20 --no-pager
# Look for a line like: https://abc123.srv.us → localhost:5000
```

The hostname persists across restarts for the same binary/port combination.

---

## 12. End-to-end verification

Run after all services are up:

```bash
KEY=$(grep ^API_KEY= /opt/spaitra/.env | cut -d= -f2-)

# Health
curl -sf http://127.0.0.1:8001/health && echo " OCR OK"
curl -sf http://127.0.0.1:5000/health && echo " Core OK"

# Auth
curl -s -o /dev/null -w "No key → %{http_code} (expect 401)\n" \
  http://127.0.0.1:5000/debug/state
curl -s -o /dev/null -w "With key → %{http_code} (expect 200)\n" \
  http://127.0.0.1:5000/debug/state -H "X-API-Key: $KEY"
curl -s -o /dev/null -w "OCR no key → %{http_code} (expect 401)\n" \
  -X POST -F image=@$REPO/src/visual_memory/tests/text_demo/typed.jpeg \
  http://127.0.0.1:8001/ocr
curl -s -o /dev/null -w "OCR with key → %{http_code} (expect 200)\n" \
  -X POST -F image=@$REPO/src/visual_memory/tests/text_demo/typed.jpeg \
  -H "X-API-Key: $KEY" \
  http://127.0.0.1:8001/ocr

# System state (GPU, pipelines, OCR reachability)
curl -sf http://127.0.0.1:5000/debug/state -H "X-API-Key: $KEY" | python3 -m json.tool

# Pipeline smoke tests
curl -sf http://127.0.0.1:5000/debug/test-remember -H "X-API-Key: $KEY" | python3 -m json.tool
curl -sf http://127.0.0.1:5000/debug/test-scan    -H "X-API-Key: $KEY" | python3 -m json.tool

# Retrain endpoint (expect insufficient_data on a fresh install)
curl -sf -X POST http://127.0.0.1:5000/retrain -H "X-API-Key: $KEY"
```

### OCR soak check (required)

Run this after deploys that touch OCR, OCR env, or service config:

```bash
KEY=$(grep ^API_KEY= /opt/spaitra/.env | cut -d= -f2-)
for i in 1 2 3 4 5 6 7 8; do
  curl -s -o /dev/null -w "trial_${i} code=%{http_code} time=%{time_total}\n" \
    --max-time 90 -H "Expect:" \
    -X POST -F image=@$REPO/src/visual_memory/tests/text_demo/typed.jpeg \
    -H "X-API-Key: $KEY" \
    http://127.0.0.1:8001/ocr
done

journalctl -u spaitra-ocr --since "5 min ago" --no-pager \
  | egrep -i "timeout|504|oom|killed|internal server error" | tail -n 20 | cat
```

Success criteria:
- all soak requests return `200`
- no new timeout/504/oom lines after the soak start time
- if capacity is exceeded, OCR returns `429` with `Retry-After` and `server_busy` payload
- if caller exceeds sustained rate, OCR returns `429` with `rate_limited` payload

---

## 13. Updating the deployment

```bash
su - spaitra -s /bin/bash
cd /opt/spaitra/TSA-soft-dev-backend-2026
git pull

# Fix ownership if pull was run as root:
# sudo chown -R spaitra:spaitra src/ services/

source venv-core/bin/activate
pip install -e ".[core]"
deactivate

source venv-ocr/bin/activate
pip install -e ".[ocr]"
deactivate
exit

sudo systemctl restart spaitra-ocr
sudo systemctl restart spaitra-core
```

### DB migration runbook: plaintext -> encrypted

Use this only when turning on `DB_ENCRYPTION_KEY` for an existing plaintext DB.

```bash
sudo systemctl stop spaitra-core

# Backup
sudo cp /opt/spaitra/TSA-soft-dev-backend-2026/data/memory.db \
  /opt/spaitra/TSA-soft-dev-backend-2026/data/memory.db.plaintext.bak

# Export plaintext DB into encrypted DB using SQLCipher
DBK=$(sudo grep '^DB_ENCRYPTION_KEY=' /opt/spaitra/.env | cut -d= -f2-)
sudo /opt/spaitra/TSA-soft-dev-backend-2026/venv-core/bin/python - <<PY
from sqlcipher3 import dbapi2 as sqlite3
old="/opt/spaitra/TSA-soft-dev-backend-2026/data/memory.db.plaintext.bak"
enc="/opt/spaitra/TSA-soft-dev-backend-2026/data/memory.db"
key="""$DBK"""
conn = sqlite3.connect(old)
escaped = key.replace("'", "''")
conn.execute(f"ATTACH DATABASE '{enc}' AS encrypted KEY '{escaped}'")
conn.execute("SELECT sqlcipher_export('encrypted')")
conn.execute("DETACH DATABASE encrypted")
conn.close()
PY

sudo chown spaitra:spaitra /opt/spaitra/TSA-soft-dev-backend-2026/data/memory.db
sudo chmod 600 /opt/spaitra/TSA-soft-dev-backend-2026/data/memory.db
sudo systemctl restart spaitra-core
```

Validation:

```bash
KEY=$(grep ^API_KEY= /opt/spaitra/.env | cut -d= -f2-)
curl -s -o /dev/null -w "core health=%{http_code}\n" http://127.0.0.1:5000/health
curl -s -o /dev/null -w "core debug=%{http_code}\n" \
  -H "X-API-Key: $KEY" \
  http://127.0.0.1:5000/debug/state
```

Rollback:
- stop `spaitra-core`
- restore `memory.db.plaintext.bak` to `memory.db`
- clear or correct `DB_ENCRYPTION_KEY` in `/opt/spaitra/.env`
- restart `spaitra-core`

If model weights changed (new checkpoint in `setup_weights.py`):

```bash
su - spaitra -s /bin/bash -c "
  source /opt/spaitra/TSA-soft-dev-backend-2026/venv-core/bin/activate
  cd /opt/spaitra/TSA-soft-dev-backend-2026
  python setup_weights.py
"
```

---

## 14. Troubleshooting

### Service fails to start

```bash
journalctl -u spaitra-core -n 100 --no-pager
journalctl -u spaitra-ocr  -n 100 --no-pager
```

Common causes:

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` | Wrong `WorkingDirectory` or venv path in service file |
| `No such file: .env` | `/opt/spaitra/.env` missing or wrong path |
| `401 Unauthorized` from OCR | Missing or wrong `X-API-Key` from core, or key mismatch between `/opt/spaitra/.env` and `/opt/spaitra/.ocr.env` |
| OCR returns `429` with `server_busy` | Service is throttling by design. Reduce concurrent callers or raise `OCR_MAX_CONCURRENCY` carefully. Respect `Retry-After` on clients. |
| OCR returns `429` with `rate_limited` | Caller exceeded request budget. Respect `Retry-After` and tune `OCR_RATE_LIMIT_PER_MIN` for expected client concurrency. |
| OCR `500` with `ConvertPirAttribute2RuntimeAttribute` | Set `OCR_ENABLE_MKLDNN=0` in `/opt/spaitra/.ocr.env`, restart `spaitra-ocr` |
| OCR service restarts or times out on large photos | Lower `OCR_MAX_SIDE` (for example `1600`) and restart `spaitra-ocr` |
| `500` at startup with `DB_ENCRYPTION_KEY is set but SQLCipher is unavailable` | On Python 3.13, clear `DB_ENCRYPTION_KEY` until SQLCipher bindings catch up (current `pysqlcipher3` build fails on 3.13); on <=3.12 reinstall core deps with `libsqlcipher-dev` |
| `Failed to unlock encrypted database` | `DB_ENCRYPTION_KEY` mismatch; restore the correct key or start with a fresh DB |
| `PermissionError` on `logs/app.log` during tests or startup | `sudo chown -R spaitra:spaitra $REPO/logs && sudo chmod 750 $REPO/logs && sudo find $REPO/logs -maxdepth 1 -type f -name "*.log" -exec chmod 640 {} \;` |
| `Permission denied` on model files | `chown -R spaitra:spaitra $REPO/checkpoints` |
| `CUDA out of memory` | Enable `SAVE_VRAM=1` in `.env` and restart |
| Source files owned by root | `chown -R spaitra:spaitra $REPO/src $REPO/services` |

### Service-level memory and restart guardrails (OCR)

Add the following to `spaitra-ocr.service` for stricter failure behavior:

```ini
MemoryMax=8G
MemoryHigh=7G
StartLimitIntervalSec=300
StartLimitBurst=5
```

Then reload and restart:

```bash
sudo systemctl daemon-reload
sudo systemctl restart spaitra-ocr
```

### GPU / VRAM issues

```bash
# Check CUDA is visible to Python:
$REPO/venv-core/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"

# Current VRAM usage:
nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total --format=csv,noheader

# If CUDA not available after driver install, reinstall PyTorch:
source $REPO/venv-core/bin/activate
bash $REPO/deploy/install_gpu_deps.sh
```

### OCR errors

```bash
# Test OCR directly:
curl http://127.0.0.1:8001/health
curl -X POST -F image=@$REPO/src/visual_memory/tests/text_demo/typed.jpeg \
  http://127.0.0.1:8001/ocr

# If core is logging OCR 500 errors, check the OCR service logs:
journalctl -u spaitra-ocr -n 50 --no-pager
```

### Retrain log

```bash
# View retrain history:
tail -50 $REPO/logs/retrain.log

# Check feedback count:
KEY=$(grep ^API_KEY= /opt/spaitra/.env | cut -d= -f2-)
curl -sf http://127.0.0.1:5000/retrain/status -H "X-API-Key: $KEY"
```

---

## 15. Deployment layout

```text
/opt/spaitra/
├── .env                          ← core backend secrets (not in git)
├── .ocr.env                      ← OCR service config (not in git)
├── .git-credentials              ← GitHub PAT for git operations (not in git)
├── bin/
│   └── auto_retrain.sh           ← nightly retrain cron script
└── TSA-soft-dev-backend-2026/    ← git repository
    ├── venv-core/                ← core Python environment
    ├── venv-ocr/                 ← OCR Python environment
    ├── checkpoints/
    │   └── depth_pro.pt          ← Depth Pro weights (~2 GB)
    ├── data/
    │   └── memory.db             ← SQLCipher-encrypted DB when DB_ENCRYPTION_KEY is set
    ├── logs/
    │   └── retrain.log           ← auto-retrain cron output
    ├── models/
    │   └── projection_head.pt    ← trained head checkpoint (created after /retrain)
    ├── deploy/                   ← service file templates and env examples
    ├── services/                 ← gunicorn/uvicorn entrypoints
    └── src/visual_memory/        ← shared application package

/etc/systemd/system/
├── spaitra-core.service
├── spaitra-ocr.service
└── spaitra-tunnel.service
```

### Optional memory cleanup cron
Use the helper script to run zombie cleanup every 30 minutes:

```bash
*/30 * * * * /opt/spaitra/TSA-soft-dev-backend-2026/deploy/memory_cleanup.sh >> /var/log/spaitra/memory_cleanup.log 2>&1
```
