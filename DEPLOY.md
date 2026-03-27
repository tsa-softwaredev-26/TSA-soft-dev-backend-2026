# Spaitra Deployment Guide

Target: Debian 12 or newer, Python 3.11, optional NVIDIA GPU for the core backend.

This deployment uses two isolated services on the same host:

- `spaitra-core`: Flask + gunicorn backend, Torch models, SQLite
- `spaitra-ocr`: FastAPI + uvicorn OCR microservice, PaddleOCR

The services communicate over localhost with `OCR_SERVICE_URL=http://127.0.0.1:8001/ocr`.

## 1. Install system packages

```bash
apt-get update
apt-get install -y \
    python3.11 python3.11-venv python3.11-dev \
    git gh build-essential curl \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgomp1 \
    libjpeg-dev libpng-dev libtiff-dev libwebp-dev \
    ffmpeg
```

For GPU hosts, make sure NVIDIA drivers are already installed and `nvidia-smi` works before continuing.

## 2. Create the service user

```bash
adduser --system --group --home /opt/spaitra spaitra
mkdir -p /opt/spaitra
chown spaitra:spaitra /opt/spaitra
```

## 3. Clone the repository

Use either an SSH deploy key or a GitHub PAT as the `spaitra` user.

SSH deploy key flow:

```bash
su - spaitra -s /bin/bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
ssh-keygen -t ed25519 -C "spaitra-server" -f ~/.ssh/deploy_key -N ""
cat ~/.ssh/deploy_key.pub
```

Add that public key to the repository deploy keys, then:

```bash
cat >> ~/.ssh/config <<'EOF'
Host github.com
    IdentityFile /opt/spaitra/.ssh/deploy_key
    StrictHostKeyChecking accept-new
EOF
chmod 600 ~/.ssh/config

cd /opt/spaitra
git clone git@github.com:tsa-softwaredev-26/TSA-soft-dev-backend-2026.git .
```

PAT flow:

```bash
su - spaitra -s /bin/bash
echo "YOUR_PAT" | gh auth login --with-token --hostname github.com
cd /opt/spaitra
gh repo clone tsa-softwaredev-26/TSA-soft-dev-backend-2026 .
```

> **Note — subdirectory layout:** Both clone commands above use a trailing `.` to place
> the repo directly in `/opt/spaitra/`. If you omit the `.` (or clone via a GUI), the
> repo lands at `/opt/spaitra/TSA-soft-dev-backend-2026/`. In that case all paths in the
> steps below shift by one level: create venvs inside the repo dir, set
> `WorkingDirectory` in the service files to that subdir, and update the `ExecStart`
> venv paths accordingly. The service files in `deploy/` assume the flat layout; the
> installed files under `/etc/systemd/system/` on this server use the subdirectory layout.

## 4. Create two isolated Python environments

All commands below run as the `spaitra` user inside `/opt/spaitra` (or the repo subdir
if you used the subdirectory layout).

### Core backend environment

```bash
python3.11 -m venv venv-core
source venv-core/bin/activate
pip install --upgrade pip
pip install -e ".[core]"
deactivate
```

GPU hosts should install the correct PyTorch wheel index inside `venv-core`:

```bash
source /opt/spaitra/venv-core/bin/activate
bash deploy/install_gpu_deps.sh
deactivate
```

CPU-only core hosts can replace that with:

```bash
source /opt/spaitra/venv-core/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
deactivate
```

### OCR environment

```bash
python3.11 -m venv venv-ocr
source venv-ocr/bin/activate
pip install --upgrade pip
pip install -e ".[ocr]"
deactivate
```

Do not install OCR packages into `venv-core`. Do not install Torch model dependencies into `venv-ocr` unless you intentionally want a combined debug environment.

## 5. Authenticate HuggingFace for the core service

The core backend needs access to gated model repositories.

```bash
source /opt/spaitra/venv-core/bin/activate
hf auth login
python setup_weights.py
deactivate
```

## 6. Create environment files

Core backend env:

```bash
cp deploy/env.example /opt/spaitra/.env
nano /opt/spaitra/.env
```

Recommended core values:

```dotenv
API_KEY=replace-me
ENABLE_DEPTH=1
ENABLE_OCR=1
OCR_SERVICE_URL=http://127.0.0.1:8001/ocr
ENABLE_LEARNING=1
API_HOST=127.0.0.1
API_PORT=5000
```

**GPU VRAM < 8 GB (e.g. GTX 1060 6 GB):** add this line:

```dotenv
SAVE_VRAM=1
```

With `SAVE_VRAM=1` the service swaps model weights between GPU and CPU RAM between
pipeline calls instead of keeping everything loaded simultaneously (~5.9 GB idle).
- Remember calls: GroundingDINO + DINOv3 + CLIP on GPU; YOLOE + Depth Pro on CPU.
- Scan calls: YOLOE + Depth Pro + DINOv3 + CLIP on GPU; GroundingDINO on CPU.
- Transfer cost per swap: ~1–2 s for GroundingDINO, ~3–5 s for Depth Pro (PCIe, not disk).
- Requires 16 GB system RAM to hold all models resident in CPU memory.

OCR service env:

```bash
cp deploy/ocr.env.example /opt/spaitra/.ocr.env
nano /opt/spaitra/.ocr.env
```

Recommended OCR values:

```dotenv
OCR_HOST=127.0.0.1
OCR_PORT=8001
OCR_LANG=en
OCR_MIN_CONFIDENCE=0.3
OCR_USE_ANGLE_CLS=0
```

## 7. Smoke test each service before systemd

### OCR service

```bash
cd /opt/spaitra
source venv-ocr/bin/activate
set -a
. /opt/spaitra/.ocr.env
set +a
python -m services.ocr.run
```

In another shell:

```bash
curl http://127.0.0.1:8001/health
curl -X POST -F image=@/opt/spaitra/src/visual_memory/tests/text_demo/typed.jpeg \
  http://127.0.0.1:8001/ocr
```

Stop the manual OCR process after verification.

### Core service

```bash
cd /opt/spaitra
source venv-core/bin/activate
set -a
. /opt/spaitra/.env
set +a
python -m services.core.run
```

In another shell:

```bash
curl http://127.0.0.1:5000/health
```

Stop the manual core process after verification.

## 8. Install systemd units

```bash
cp /opt/spaitra/deploy/spaitra-core.service /etc/systemd/system/
cp /opt/spaitra/deploy/spaitra-ocr.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now spaitra-ocr
systemctl enable --now spaitra-core
```

Check status:

```bash
systemctl status spaitra-ocr --no-pager
systemctl status spaitra-core --no-pager
journalctl -u spaitra-ocr -n 100 --no-pager
journalctl -u spaitra-core -n 100 --no-pager
```

## 9. Verify end-to-end behavior

Check both health endpoints:

```bash
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:5000/health
```

Confirm the core service can reach OCR through its configured URL:

```bash
set -a
. /opt/spaitra/.env
set +a
curl -X POST -F image=@/opt/spaitra/src/visual_memory/tests/text_demo/typed.jpeg \
  "$OCR_SERVICE_URL"
```

If you want the core backend to run without OCR, set `ENABLE_OCR=0` in `/opt/spaitra/.env` and restart only `spaitra-core`.

## 10. Public access with srv.us

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

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now spaitra-tunnel
```

## 11. Updating the deployment

```bash
su - spaitra -s /bin/bash
cd /opt/spaitra
git pull

source venv-core/bin/activate
pip install -e ".[core]"
deactivate

source venv-ocr/bin/activate
pip install -e ".[ocr]"
deactivate
exit

systemctl restart spaitra-ocr
systemctl restart spaitra-core
```

If the core model stack changes, also rerun:

```bash
su - spaitra -s /bin/bash
cd /opt/spaitra
source venv-core/bin/activate
python setup_weights.py
deactivate
```

## 12. Troubleshooting

### Core service fails to start

```bash
journalctl -u spaitra-core -n 100 --no-pager
```

Check:

- `/opt/spaitra/.env` exists and has valid values
- `hf auth login` was run in `venv-core`
- `python setup_weights.py` completed in `venv-core`
- `OCR_SERVICE_URL` points to a reachable endpoint if `ENABLE_OCR=1`

### OCR service fails to start

```bash
journalctl -u spaitra-ocr -n 100 --no-pager
```

Check:

- `/opt/spaitra/.ocr.env` exists
- `pip install -e ".[ocr]"` completed in `venv-ocr`
- the port in `.ocr.env` matches the port in `OCR_SERVICE_URL`

### OCR requests time out

Check the OCR service directly:

```bash
curl http://127.0.0.1:8001/health
curl -X POST -F image=@/opt/spaitra/src/visual_memory/tests/text_demo/typed.jpeg \
  http://127.0.0.1:8001/ocr
```

### GPU issues in the core backend

Inside `venv-core`:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If false on a GPU host, rerun `bash deploy/install_gpu_deps.sh`.

## 13. Final deployment layout

```text
/opt/spaitra/
  .env
  .ocr.env
  venv-core/
  venv-ocr/
  deploy/
  services/
  src/visual_memory/
  checkpoints/
  data/memory.db
  models/
  logs/app.log
```
