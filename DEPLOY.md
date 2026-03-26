# Spaitra Backend - Debian Deployment Guide

Target: Debian 12 (bookworm), Python 3.11, optional NVIDIA GPU.
Public access via srv.us tunnel pointing to gunicorn on localhost:5000.

---

## Automated Setup

For a fresh Debian 12 server, the setup script automates all steps below:

```bash
# As root, from the cloned repo:
bash deploy/setup_server.sh
# Set TORCH_CUDA=cu121 (or cu118) for GPU servers:
TORCH_CUDA=cu121 bash deploy/setup_server.sh
```

The script installs packages, creates the `spaitra` user, sets up the venv,
prompts for HuggingFace login, downloads weights, and installs the systemd service.

---

## 1. System Prerequisites

```bash
apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    git build-essential wget curl \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    libgomp1 \
    libjpeg-dev libpng-dev libtiff-dev libwebp-dev \
    ffmpeg
```

For GPU servers, install CUDA drivers first (check your card with `nvidia-smi`):
```bash
# CUDA 12.x (recommended for modern cards)
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update && apt-get install -y cuda-toolkit-12-4
```

---

## 2. User and Directory Setup

```bash
adduser --system --group --home /opt/spaitra spaitra
mkdir -p /opt/spaitra && chown spaitra:spaitra /opt/spaitra
```

---

## 3. Clone and Install

```bash
su - spaitra -s /bin/bash
cd /opt/spaitra

git clone https://github.com/tsa-softwaredev-26/TSA-soft-dev-backend-2026.git .

python3.11 -m venv venv
source venv/bin/activate
```

Install PyTorch first (choose one based on your server):

```bash
# GPU server (CUDA 12.x):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# GPU server (CUDA 11.x):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU-only server:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Install the rest of the package:
```bash
pip install -e .
```

For GPU PaddlePaddle (optional, replaces the CPU version installed above):
```bash
# CUDA 12.x:
pip install paddlepaddle-gpu==3.0.0.post120
# CUDA 11.x:
pip install paddlepaddle-gpu==3.0.0
```

---

## 4. HuggingFace Auth (required for gated models)

Two models require accepted access on HuggingFace before they will download:

- `facebook/dinov3-vits16-pretrain-lvd1689m` (image embedder)
- `IDEA-Research/grounding-dino-base` (remember mode detector)

Request access on huggingface.co, then authenticate:
```bash
pip install huggingface_hub
huggingface-cli login
# paste your HuggingFace token when prompted
```

---

## 5. Download Model Weights

```bash
python setup_weights.py
# Downloads: checkpoints/depth_pro.pt (~2GB) and YOLOE weights (~80MB)
# DINOv3 + GroundingDINO download lazily on first warm_all() call
```

First startup will download DINOv3 (~1.2GB) and GroundingDINO (~900MB) from HuggingFace.
Total disk needed: ~5GB. Expect 10-30 min on first start.

---

## 6. Environment Configuration

```bash
cp deploy/env.example .env
# Edit .env - minimum required:
#   API_KEY=<strong-random-secret>
#   ENABLE_DEPTH=0   (skip Depth Pro if no GPU or limited VRAM)
```

Key variables:

| Variable | Default | Notes |
|----------|---------|-------|
| `API_KEY` | (none) | Required in production. All routes except /health need `X-API-Key` header. |
| `ENABLE_DEPTH` | `1` | Set to `0` on CPU-only or low-VRAM servers. |
| `ENABLE_OCR` | `1` | Set to `0` to skip PaddleOCR entirely (faster startup). |
| `ENABLE_LEARNING` | `1` | Set to `0` to disable projection head. |

---

## 7. Systemd Service

```bash
# As root:
cp /opt/spaitra/deploy/spaitra.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable spaitra
systemctl start spaitra

# Watch startup logs (first run downloads models - takes a few minutes):
journalctl -u spaitra -f
```

Service is healthy when you see:
```
{"event": "warm_all_complete", ...}
```

Test locally: `curl http://127.0.0.1:5000/health`
Expected: `{"status": "ok"}`

---

## 8. srv.us Tunnel

srv.us provides a stable public HTTPS URL that tunnels to your local port.
Install the srv.us client per their documentation, then run:

```bash
srv.us --port 5000
```

To keep it running permanently, create a second systemd service:

```ini
# /etc/systemd/system/spaitra-tunnel.service
[Unit]
Description=Spaitra srv.us tunnel
After=spaitra.service
Requires=spaitra.service

[Service]
Type=simple
User=spaitra
ExecStart=/usr/local/bin/srv.us --port 5000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
systemctl enable --now spaitra-tunnel
```

The public URL shown by srv.us is your frontend's base URL. Set it in the app as:
```
Base URL: https://<your-srv-us-subdomain>.srv.us
```

---

## 9. Updating the Server

```bash
su - spaitra -s /bin/bash
cd /opt/spaitra
source venv/bin/activate
git pull
pip install -e .
systemctl restart spaitra
```

---

## 10. Troubleshooting

**Service won't start:**
```bash
journalctl -u spaitra --no-pager -n 50
```

Common causes:
- Missing `.env` file (`cp deploy/env.example .env`)
- HuggingFace token not set (`huggingface-cli login`)
- CUDA version mismatch (reinstall torch with correct index URL)
- Out of disk space during model download

**HuggingFace 403 on model download:**
- Check your token has read access: `huggingface-cli whoami`
- Confirm you have accepted the model's license on the model page

**gunicorn timeout (503):**
- Increase `--timeout` in `deploy/spaitra.service` (default is 120s)
- First requests after cold start are slow (model warm-up)

**Depth Pro CUDA OOM:**
- Set `ENABLE_DEPTH=0` in `.env` and restart

**PaddleOCR slow on CPU:**
- Expected: 3-18s per image crop with text. The text likelihood pre-check
  skips OCR for crops without text; set `ENABLE_OCR=0` to disable entirely.

---

## Directory Layout (after setup)

```
/opt/spaitra/
├── .env                        # secrets + feature toggles
├── venv/                       # Python virtualenv
├── checkpoints/depth_pro.pt    # downloaded by setup_weights.py
├── data/memory.db              # SQLite database (auto-created on first run)
├── feedback/                   # training feedback files (auto-created)
├── models/                     # projection head weights (auto-created on first /retrain)
├── logs/app.log                # JSON lines application log
└── src/visual_memory/...       # source code
```
