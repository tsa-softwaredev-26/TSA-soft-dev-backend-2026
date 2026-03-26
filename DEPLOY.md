# Spaitra Backend - Deployment Guide

Target: Debian 12 (Bookworm) or 13 (Trixie), Python 3.11, optional NVIDIA GPU.
Public access via srv.us tunnel pointing to gunicorn on localhost:5000.

All steps below are manual. Run them in order on a fresh server.

---

## 1. System Packages

```bash
apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev \
    git gh build-essential wget curl \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgomp1 \
    libjpeg-dev libpng-dev libtiff-dev libwebp-dev \
    ffmpeg
```

**GPU note:** CUDA drivers must already be installed before continuing.
Verify with `nvidia-smi`. Most GPU VPS providers offer CUDA-preloaded images -
select one at provisioning time. For manual driver installation, use NVIDIA's
runfile installer at developer.nvidia.com/cuda-downloads (select "runfile (local)"
for your Debian version). Do not use the NVIDIA apt repository on Debian 12+;
the SHA1-signed keyring is rejected by current Debian policy.

---

## 2. GitHub Authentication

The repo is private. Use a Personal Access Token (PAT) to authenticate.

**Create a PAT:**
1. Go to: github.com -> Settings -> Developer settings -> Personal access tokens -> Tokens (classic)
2. Click "Generate new token (classic)"
3. Select the `repo` scope, set an expiry date
4. Copy the token immediately - it will not be shown again

**Authenticate gh CLI on the server:**
```bash
# Run as the user who will clone the repo (spaitra, after step 3):
echo "YOUR_PAT_HERE" | gh auth login --with-token
# Verify: gh auth status
```

---

## 3. Service User and Directory

```bash
# As root:
adduser --system --group --home /opt/spaitra spaitra
mkdir -p /opt/spaitra
chown spaitra:spaitra /opt/spaitra
```

---

## 4. Clone and Python Setup

```bash
su - spaitra -s /bin/bash
cd /opt/spaitra

# Authenticate (paste PAT from step 2):
echo "YOUR_PAT_HERE" | gh auth login --with-token

gh repo clone tsa-softwaredev-26/TSA-soft-dev-backend-2026 .

python3.11 -m venv venv
source venv/bin/activate
```

Install the package and gunicorn:
```bash
pip install -e .
pip install gunicorn
```

**GPU servers:** run the helper script to auto-detect your CUDA version and install
the matching PyTorch and PaddlePaddle GPU packages:
```bash
bash deploy/install_gpu_deps.sh
```

The script reads `nvidia-smi`, selects CUDA 11 or 12 indexes automatically, installs
both packages, and verifies both can see the GPU. It will print an error and exit if
drivers are missing. The CPU paddle installed by `pip install -e .` is replaced by
the GPU version.

**CPU-only servers:** install PyTorch CPU and skip PaddlePaddle GPU:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# PaddleOCR will run on CPU (3-18s per image crop) - no further action needed.
```

---

## 5. HuggingFace Auth

Two models require accepting a license on huggingface.co before they will download:

- `facebook/dinov3-vits16-pretrain-lvd1689m` (image embedder)
- `IDEA-Research/grounding-dino-base` (remember mode detector)

Accept both licenses, then authenticate:
```bash
huggingface-cli login
# Paste a HuggingFace token with read access when prompted
# Get one at: huggingface.co/settings/tokens
```

---

## 6. Download Model Weights

```bash
python setup_weights.py
```

Downloads `checkpoints/depth_pro.pt` (~2GB) and YOLOE weights (~80MB).
DINOv3 (~1.2GB) and GroundingDINO (~900MB) download from HuggingFace on first
startup. Total disk required: ~5GB. First cold start takes 10-30 min.

---

## 7. Environment Configuration

```bash
cp deploy/env.example .env
nano .env
```

Set `API_KEY` to a strong random secret. All routes except `/health` require the
`X-API-Key` header to match this value.

| Variable | Default | Notes |
|----------|---------|-------|
| `API_KEY` | (none) | Required in production. |
| `ENABLE_DEPTH` | `1` | Set `0` on servers with <4GB VRAM or CPU-only. |
| `ENABLE_OCR` | `1` | Set `0` to disable PaddleOCR (faster startup). |
| `ENABLE_LEARNING` | `1` | Set `0` to disable projection head. |

---

## 8. Systemd Service

```bash
# Exit back to root, then:
cp /opt/spaitra/deploy/spaitra.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable spaitra
systemctl start spaitra

# Watch startup logs (model downloads happen here on first run):
journalctl -u spaitra -f
```

Service is ready when logs show `{"event": "warm_all_complete", ...}`.

Test: `curl http://127.0.0.1:5000/health`
Expected: `{"status": "ok"}`

---

## 9. srv.us Tunnel

Install the srv.us client per their documentation, then create a second service
so the tunnel restarts automatically:

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

The public HTTPS URL printed by srv.us is the base URL for the iOS frontend.

---

## 10. Updating

```bash
su - spaitra -s /bin/bash
cd /opt/spaitra
source venv/bin/activate
git pull
pip install -e .
exit

systemctl restart spaitra
```

---

## 11. Troubleshooting

**Service fails to start:**
```bash
journalctl -u spaitra --no-pager -n 100
```

Common causes:
- Missing `.env` - run `cp deploy/env.example .env` and set `API_KEY`
- HuggingFace auth missing - run `huggingface-cli login` as the spaitra user
- PyTorch/CUDA version mismatch - reinstall torch with the correct index URL
- Disk full - model downloads need ~5GB free in `/opt/spaitra`

**HuggingFace 403 on model download:**
- Check token is valid: `huggingface-cli whoami`
- Confirm you accepted the license for each gated model on huggingface.co

**PaddlePaddle GPU not working after install:**
```bash
python -c "import paddle; print(paddle.device.get_device())"
# Should print: gpu:0
# If it prints cpu: the GPU package did not install; check CUDA version vs index URL
```

**gunicorn timeout / 503:**
- Increase `--timeout` in `deploy/spaitra.service` (default 120s; model warm-up can exceed this on slow hardware)
- After editing the service file: `systemctl daemon-reload && systemctl restart spaitra`

**Depth Pro OOM:**
- Set `ENABLE_DEPTH=0` in `.env`, then `systemctl restart spaitra`

---

## Directory Layout (after setup)

```
/opt/spaitra/
├── .env                        # secrets and feature flags
├── venv/                       # Python virtualenv
├── checkpoints/depth_pro.pt    # downloaded by setup_weights.py (~2GB)
├── data/memory.db              # SQLite database (created on first run)
├── models/                     # projection head weights (created on first /retrain)
├── logs/app.log                # JSON lines application log
└── src/visual_memory/...       # source code
```
