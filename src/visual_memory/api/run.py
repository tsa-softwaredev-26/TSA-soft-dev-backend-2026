# Local dev entry point. Starts Flask on 127.0.0.1:5000 (single process, single thread).
#
# --- Debian server setup (run once after git clone) ---
#
#   adduser --system --group spaitra
#   git clone git@github.com:tsa-softwaredev-26/TSA-soft-dev-backend-2026.git /opt/spaitra
#   cd /opt/spaitra && python3.11 -m venv venv && source venv/bin/activate
#
#   1. Install CUDA-enabled PyTorch (check CUDA version with: nvidia-smi):
#      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#      # CPU-only fallback (no GPU): pip install torch torchvision
#
#   2. Install GPU PyTorch (auto-detects CUDA version):
#      bash deploy/install_gpu_deps.sh
#      # CPU-only: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#
#   3. Install the package and download weights:
#      pip install -e . && pip install gunicorn
#      python setup_weights.py
#      hf auth login   # needed for gated DINOv3 + GDINO models
#
#   4. Configure environment:
#      cp deploy/env.example .env
#      # edit .env: set API_KEY, toggle ENABLE_DEPTH/ENABLE_OCR as needed
#
#   5. Install and start systemd service:
#      cp deploy/spaitra.service /etc/systemd/system/
#      systemctl daemon-reload && systemctl enable --now spaitra
#      journalctl -u spaitra -f   # tail logs
#
#   6. srv.us tunnel (run as a separate systemd service or screen session):
#      # srv.us forwards a public HTTPS URL to 127.0.0.1:5000
#      # Install srv.us client per their docs, then:
#      #   srv.us --port 5000
#      # The public URL is your Base URL for the frontend.
#
# --- Environment variables ---
#   API_KEY=<secret>   enforce X-API-Key header on all routes except /health
#   ENABLE_DEPTH=0     skip Depth Pro (saves ~2GB VRAM; recommended if no GPU)
#   ENABLE_OCR=0       skip OCR calls (faster startup, no text recognition)
#   OCR_SERVICE_URL    OCR microservice endpoint (default http://127.0.0.1:8001/ocr)
#   ENABLE_LEARNING=0  disable projection head (raw embeddings only)
#
# --- Direct gunicorn invocation (no systemd) ---
#   API_KEY=secret ENABLE_DEPTH=0 \
#   gunicorn -w 1 -b 127.0.0.1:5000 --timeout 120 wsgi:application
#
# --- Mac (local dev) ---
#   python src/visual_memory/api/run.py

from visual_memory.api.app import create_app
from visual_memory.config import Settings

if __name__ == "__main__":
    settings = Settings()
    app = create_app()
    app.run(host=settings.api_host, port=settings.api_port, threaded=False)
