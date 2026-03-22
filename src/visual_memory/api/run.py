# Local dev entry point. Starts Flask on 127.0.0.1:5000 (single process, single thread).
#
# On Ubuntu with NVIDIA GPU - run once after git clone:
#
#   1. Install CUDA-enabled PyTorch (check CUDA version with: nvidia-smi):
#      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#
#   2. Install paddlepaddle-gpu instead of CPU paddle (match your CUDA version):
#      pip install paddlepaddle-gpu==3.0.0
#
#   3. Install the rest:
#      pip install -e .
#      python setup_weights.py
#
#   4. Run with gunicorn (single worker required - model state is process-local):
#      pip install gunicorn
#      gunicorn -w 1 -b 0.0.0.0:5000 "visual_memory.api.app:create_app()"
#
#   Environment variables:
#      API_KEY=<secret>  - enforce X-API-Key header on all routes except /health (opt-in)
#      ENABLE_DEPTH=0    - skip Depth Pro (saves ~2GB VRAM, recommended if not needed)
#      ENABLE_OCR=0      - skip PaddleOCR
#      Example: API_KEY=secret ENABLE_DEPTH=0 gunicorn -w 1 -b 0.0.0.0:5000 "visual_memory.api.app:create_app()"
#
# On Mac (local dev):
#      python src/visual_memory/api/run.py

from visual_memory.api.app import create_app
from visual_memory.config import Settings

if __name__ == "__main__":
    settings = Settings()
    app = create_app()
    app.run(host=settings.api_host, port=settings.api_port, threaded=False)
