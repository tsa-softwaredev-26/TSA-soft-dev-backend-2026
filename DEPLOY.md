# Spaitra Backend - Deployment Guide

Target: Debian 12 (Bookworm) or 13 (Trixie), Python 3.11, optional NVIDIA GPU.
Public access via srv.us tunnel pointing to gunicorn on localhost:5000.

This guide is for server deployment only. For local development and running tests,
see the Local Development section in README.md.

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

## 2. Service User and Directory

```bash
# As root:
adduser --system --group --home /opt/spaitra spaitra
mkdir -p /opt/spaitra
chown spaitra:spaitra /opt/spaitra
```

---

## 3. GitHub Access

The repo is private. Use an SSH deploy key (recommended) or a Personal Access Token.

### Option A - SSH Deploy Key (recommended)

A deploy key is scoped to one repo, read-only, has no expiry, and requires no
external tooling after setup.

```bash
# As spaitra user:
su - spaitra -s /bin/bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
ssh-keygen -t ed25519 -C "spaitra-server" -f ~/.ssh/deploy_key -N ""
cat ~/.ssh/deploy_key.pub
```

Copy the printed public key. Add it to the repo on GitHub:
- Go to: github.com/tsa-softwaredev-26/TSA-soft-dev-backend-2026 -> Settings -> Deploy keys
- Click "Add deploy key", paste the key, leave "Allow write access" unchecked

Configure SSH to use this key for GitHub:
```bash
cat >> ~/.ssh/config <<'EOF'
Host github.com
    IdentityFile /opt/spaitra/.ssh/deploy_key
    StrictHostKeyChecking accept-new
EOF
chmod 600 ~/.ssh/config
```

Test the connection:
```bash
ssh -T git@github.com
# Expected: "Hi tsa-softwaredev-26/TSA-soft-dev-backend-2026! You've successfully authenticated..."
```

Clone using SSH:
```bash
cd /opt/spaitra
git clone git@github.com:tsa-softwaredev-26/TSA-soft-dev-backend-2026.git .
```

---

### Option B - Personal Access Token (PAT)

Use this if you cannot add deploy keys to the repo (e.g. insufficient permissions).

**Create a PAT:**
1. Go to: github.com -> Settings -> Developer settings -> Personal access tokens -> Tokens (classic)
2. Click "Generate new token (classic)"
3. Select the `repo` scope, set an expiry date
4. Copy the token immediately - it will not be shown again

**Authenticate gh CLI as the spaitra user:**
```bash
su - spaitra -s /bin/bash
echo "YOUR_PAT_HERE" | gh auth login --with-token --hostname github.com
gh auth status
```

Clone:
```bash
cd /opt/spaitra
gh repo clone tsa-softwaredev-26/TSA-soft-dev-backend-2026 .
```

---

## 4. Python Setup

```bash
# As spaitra user (if not already):
su - spaitra -s /bin/bash
cd /opt/spaitra
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

The script reads `nvidia-smi`, selects the CUDA 11 or 12 index automatically,
installs both packages, and verifies both can see the GPU. It will exit with an
error if drivers are missing or the version is unsupported.

The CPU paddle installed by `pip install -e .` is replaced by the GPU version.

**CPU-only servers:** install PyTorch CPU and skip PaddlePaddle GPU:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# PaddleOCR runs on CPU (3-18s per image crop) - no further action needed.
```

---

## 5. HuggingFace Auth

Two models require accepting a license on huggingface.co before they will download:

- `facebook/dinov3-vits16-pretrain-lvd1689m` (image embedder)
- `IDEA-Research/grounding-dino-base` (remember mode detector)

Accept both licenses, then authenticate:
```bash
hf auth login
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
| `ENABLE_DEPTH` | `1` | Depth Pro requires ~2GB VRAM. All models together need ~4GB VRAM per worker. Set `0` on CPU-only or <4GB VRAM servers. |
| `ENABLE_OCR` | `1` | Set `0` to disable PaddleOCR (faster startup). |
| `ENABLE_LEARNING` | `1` | Set `0` to disable projection head. |

---

## 8. Systemd Service

```bash
# As root:
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

The service runs 2 gunicorn workers by default so that health checks and fast
requests do not block behind slow inference. Each worker loads all models
independently (~4GB VRAM each with `ENABLE_DEPTH=1`). On servers with <8GB VRAM
or CPU-only, edit `spaitra.service` and set `-w 1`, then
`systemctl daemon-reload && systemctl restart spaitra`.

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

# As root:
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
- HuggingFace auth missing - run `hf auth login` as the spaitra user
- PyTorch/CUDA version mismatch - rerun `bash deploy/install_gpu_deps.sh`
- Disk full - model downloads need ~5GB free in `/opt/spaitra`

**SSH deploy key: "Permission denied (publickey)":**
- Check the key is added to the repo deploy keys on GitHub
- Run `ssh -vT git@github.com` to see which key is being offered
- Confirm `/opt/spaitra/.ssh/config` has the `IdentityFile` pointing to `deploy_key`

**HuggingFace 403 on model download:**
- Check token is valid: `huggingface-cli whoami`
- Confirm you accepted the license for each gated model on huggingface.co

**PaddlePaddle GPU not working:**
```bash
python -c "import paddle; print(paddle.device.get_device())"
# Should print: gpu:0
# If it prints cpu: GPU package did not install; rerun deploy/install_gpu_deps.sh
```

**gunicorn timeout / 503:**
- Increase `--timeout` in `deploy/spaitra.service` (default 120s; model warm-up can exceed this)
- After editing: `systemctl daemon-reload && systemctl restart spaitra`

**Out of VRAM with 2 workers:**
- Set `-w 1` in `deploy/spaitra.service`, or set `ENABLE_DEPTH=0` in `.env`
- After editing service file: `systemctl daemon-reload && systemctl restart spaitra`

---

## Directory Layout (after setup)

```
/opt/spaitra/
├── .env                        # secrets and feature flags
├── .ssh/deploy_key             # SSH deploy key (Option A only)
├── venv/                       # Python virtualenv
├── checkpoints/depth_pro.pt    # downloaded by setup_weights.py (~2GB)
├── data/memory.db              # SQLite database (created on first run)
├── models/                     # projection head weights (created on first /retrain)
├── logs/app.log                # JSON lines application log
└── src/visual_memory/...       # source code
```
