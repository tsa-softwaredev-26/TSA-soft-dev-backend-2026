#!/usr/bin/env bash
# Detect CUDA version and install matching PyTorch + PaddlePaddle GPU.
# Run inside the activated virtualenv: bash deploy/install_gpu_deps.sh
set -euo pipefail

if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. NVIDIA driver is not installed."
    echo "On GPU VPS instances, select a CUDA-preloaded image when provisioning."
    echo "For manual installation: developer.nvidia.com/cuda-downloads (use runfile, not apt)."
    exit 1
fi

# Parse "CUDA Version: 12.4" from nvidia-smi header
CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || true)
if [ -z "$CUDA_VER" ]; then
    echo "ERROR: Could not parse CUDA version from nvidia-smi output."
    nvidia-smi | head -5
    exit 1
fi

CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
echo "Detected CUDA $CUDA_VER (major $CUDA_MAJOR)"

case "$CUDA_MAJOR" in
    12)
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu123/"
        ;;
    11)
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu118/"
        ;;
    *)
        echo "ERROR: Unsupported CUDA major version $CUDA_MAJOR. Supported: 11, 12."
        exit 1
        ;;
esac

echo "Installing PyTorch (CUDA $CUDA_MAJOR index)..."
pip install torch torchvision --index-url "$TORCH_INDEX"

echo "Installing PaddlePaddle GPU (CUDA $CUDA_MAJOR index)..."
pip install paddlepaddle-gpu==3.0.0 -i "$PADDLE_INDEX"

echo ""
echo "Verifying PyTorch GPU..."
python -c "
import torch
ok = torch.cuda.is_available()
name = torch.cuda.get_device_name(0) if ok else 'N/A'
print(f'  torch.cuda.is_available(): {ok}')
print(f'  device: {name}')
if not ok:
    raise SystemExit('PyTorch cannot see the GPU. Check driver and CUDA version match.')
"

echo ""
echo "Verifying PaddlePaddle GPU..."
python -c "import paddle; paddle.utils.run_check()"

echo ""
echo "GPU dependencies installed successfully."
