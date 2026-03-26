# Spaitra Backend - Deployment Guide

Target: Debian 12+ (Bookworm/Trixie), Python 3.11, optional NVIDIA GPU.
Public access via srv.us tunnel to gunicorn on localhost:5000.

This guide covers server deployment only. For local development, see README.md.

All steps are manual. Run them in order.

---

## 1. System Packages

```bash
apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev \
    git gh build-essential curl \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgomp1 \
    libjpeg-dev libpng-dev libtiff-dev libwebp-dev \
    ffmpeg
```

**GPU servers:** CUDA drivers must be installed before continuing - check with
`nvidia-smi`. Most GPU VPS providers offer CUDA-preloaded images; select one at
provisioning time. If you need to install drivers manually, use the NVIDIA runfile
installer (developer.nvidia.com/cuda-downloads, select "runfile"). Do not use the
NVIDIA apt repo on Debian 12+; its signing key is rejected by current Debian policy.

---

## 2. Service User

```bash
adduser --system --group --home /opt/spaitra spaitra
mkdir -p /opt/spaitra
chown spaitra:spaitra /opt/spaitra
```

---

## 3. GitHub Access

The repo is private. Use an SSH deploy key (recommended) or a PAT.

### Option A - SSH Deploy Key (recommended)

Scoped to one repo, read-only, no expiry.

```bash
su - spaitra -s /bin/bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
ssh-keygen -t ed25519 -C "spaitra-server" -f ~/.ssh/deploy_key -N ""
cat ~/.ssh/deploy_key.pub
```

Add the printed public key to GitHub:
- github.com/tsa-softwaredev-26/TSA-soft-dev-backend-2026 -> Settings -> Deploy keys -> Add deploy key
- Leave "Allow write access" unchecked

```bash
cat >> ~/.ssh/config <<'EOF'
Host github.com
    IdentityFile /opt/spaitra/.ssh/deploy_key
    StrictHostKeyChecking accept-new
EOF
chmod 600 ~/.ssh/config

# Verify before cloning:
ssh -T git@github.com
```

Clone:
```bash
cd /opt/spaitra
git clone git@github.com:tsa-softwaredev-26/TSA-soft-dev-backend-2026.git .
```

### Option B - Personal Access Token

1. github.com -> Settings -> Developer settings -> Personal access tokens -> Tokens (classic)
2. Generate new token, select `repo` scope, set an expiry
3. Copy the token - it will not be shown again

```bash
su - spaitra -s /bin/bash
echo "YOUR_PAT" | gh auth login --with-token --hostname github.com
gh auth status
cd /opt/spaitra
gh repo clone tsa-softwaredev-26/TSA-soft-dev-backend-2026 .
```

---

## 4. Python Environment

All `pip` commands below run inside the activated venv. Do not use a system-wide pip.

```bash
# As spaitra user:
cd /opt/spaitra
python3.11 -m venv venv
source venv/bin/activate

pip install -e .
pip install gunicorn
```

**GPU servers:** run the helper script - it reads `nvidia-smi`, picks the right
PyTorch and PaddlePaddle wheel indexes for your CUDA version, installs both, and
verifies both can see the GPU:

```bash
bash deploy/install_gpu_deps.sh
```

If your CUDA version is not handled, the script exits with an error and links to
the official package indexes to find the right command manually.

**CPU-only servers:** install the CPU PyTorch wheel (smaller download, no CUDA):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

PaddleOCR will run on CPU at 3-18s per image crop. No other action needed.

---

## 5. HuggingFace Auth

Two models require accepting a license on huggingface.co before they will download:

- facebook/dinov3-vits16-pretrain-lvd1689m (image embedder)
- IDEA-Research/grounding-dino-base (remember mode detector)

Accept both licenses, then log in:

```bash
hf auth login
```

Get a token at huggingface.co/settings/tokens (read access is sufficient).

---

## 6. Model Weights

```bash
python setup_weights.py
```

Downloads Depth Pro (~2GB) and YOLOE (~80MB). DINOv3 (~1.2GB) and GroundingDINO
(~900MB) download from HuggingFace on first startup. Total: ~5GB disk, 10-30 min.

---

## 7. Environment

```bash
cp deploy/env.example .env
nano .env  # set API_KEY at minimum
```

| Variable | Default | Notes |
|----------|---------|-------|
| `API_KEY` | (none) | All routes except /health require `X-API-Key` header. |
| `ENABLE_DEPTH` | `1` | Depth Pro needs ~2GB VRAM. All models together: ~4GB VRAM per worker. Set `0` on CPU-only or <4GB VRAM. |
| `ENABLE_OCR` | `1` | Set `0` to skip PaddleOCR entirely. |
| `ENABLE_LEARNING` | `1` | Set `0` to disable projection head. |

---

## 8. Systemd Service

```bash
# As root:
cp /opt/spaitra/deploy/spaitra.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable spaitra
systemctl start spaitra
journalctl -u spaitra -f
```

Ready when logs show `{"event": "warm_all_complete", ...}`.

```bash
curl http://127.0.0.1:5000/health
# {"status": "ok"}
```

The service runs 2 workers so health checks do not block during slow inference.
Each worker loads all models independently (~4GB VRAM each with `ENABLE_DEPTH=1`).
On servers with <8GB VRAM, edit `spaitra.service` to set `-w 1`, then
`systemctl daemon-reload && systemctl restart spaitra`.

---

## 9. srv.us Tunnel

Create a second systemd service so the tunnel restarts automatically:

```bash
cat > /etc/systemd/system/spaitra-tunnel.service <<'EOF'
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
EOF
systemctl enable --now spaitra-tunnel
```

The public HTTPS URL srv.us prints is the base URL for the iOS frontend.

---

## 10. Updating

```bash
su - spaitra -s /bin/bash
cd /opt/spaitra && source venv/bin/activate
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

- Missing `.env`: `cp deploy/env.example .env` and set `API_KEY`
- HuggingFace auth: run `hf auth login` as the spaitra user
- GPU packages wrong: rerun `bash deploy/install_gpu_deps.sh`
- Disk full: need ~5GB free in `/opt/spaitra`

**SSH key rejected:**
- Check the key is in GitHub repo deploy keys
- `ssh -vT git@github.com` shows which key is being offered
- Confirm `~/.ssh/config` has `IdentityFile /opt/spaitra/.ssh/deploy_key`

**HuggingFace 403:**
- `hf whoami` to verify the token is valid
- Confirm you accepted each gated model's license on huggingface.co

**PaddlePaddle not using GPU:**
```bash
python -c "import paddle; print(paddle.device.get_device())"
# Should print gpu:0 - if it prints cpu, rerun deploy/install_gpu_deps.sh
```

**VRAM OOM with 2 workers:**
- Set `-w 1` in `spaitra.service` or `ENABLE_DEPTH=0` in `.env`
- `systemctl daemon-reload && systemctl restart spaitra`

**gunicorn timeout:**
- Increase `--timeout` in `spaitra.service` (default 120s; warm-up can exceed this)
- `systemctl daemon-reload && systemctl restart spaitra`

---

## Directory Layout

```
/opt/spaitra/
├── .env
├── .ssh/deploy_key             # Option A only
├── venv/
├── checkpoints/depth_pro.pt   # ~2GB, from setup_weights.py
├── data/memory.db             # created on first run
├── models/                    # created on first /retrain
├── logs/app.log
└── src/visual_memory/
```
