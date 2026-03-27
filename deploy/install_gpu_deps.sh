#!/usr/bin/env bash
# Detect CUDA version and install matching PyTorch + PaddlePaddle GPU.
# Run inside the activated virtualenv: bash deploy/install_gpu_deps.sh
#
# If your CUDA version is not handled here, use the official selectors:
#   PyTorch:      https://pytorch.org/get-started/locally/
#   PaddlePaddle: https://www.paddlepaddle.org.cn/packages/stable/
set -euo pipefail

if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. NVIDIA driver is not installed."
    echo "Provision a CUDA-preloaded image or install drivers via the NVIDIA runfile:"
    echo "  https://developer.nvidia.com/cuda-downloads (select runfile, not apt)"
    exit 1
fi

CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || true)
if [ -z "$CUDA_VER" ]; then
    echo "ERROR: Could not parse CUDA version from nvidia-smi output:"
    nvidia-smi | head -6
    exit 1
fi

CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
echo "Detected CUDA $CUDA_VER"

# PyTorch wheel index - https://download.pytorch.org/whl/
# Full list of available cu* indexes: https://download.pytorch.org/whl/
PADDLE_VER="3.3.1"

if   [ "$CUDA_MAJOR" -ge 13 ]; then
    PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu130/"
elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 9 ]; then
    PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu129/"
elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 8 ]; then
    PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu128/"
elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 6 ]; then
    PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu126/"
elif [ "$CUDA_MAJOR" -eq 12 ]; then
    PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu126/"
elif [ "$CUDA_MAJOR" -eq 11 ]; then
    PADDLE_VER="3.1.0"
    PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu118/"
fi

# PaddlePaddle GPU index - https://www.paddlepaddle.org.cn/packages/stable/
# Each cuXXX slot carries its own latest version; see the index for exact versions.
PADDLE_VER="3.3.1"
if   [ "$CUDA_MAJOR" -ge 13 ]; then
    PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu130/"
elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 9 ]; then
    PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu129/"
elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 8 ]; then
    PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu128/"
elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 6 ]; then
    PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu126/"
elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 1 ]; then
    # cu123 tops at 3.1.0 - older than the rest but functional
    PADDLE_VER="3.1.0"
    PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu123/"
elif [ "$CUDA_MAJOR" -eq 11 ]; then
    PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu118/"
else
    echo "ERROR: No known PaddlePaddle wheel for CUDA $CUDA_VER."
    echo "Check available indexes at: https://www.paddlepaddle.org.cn/packages/stable/"
    exit 1
fi

echo "PyTorch index:    $TORCH_INDEX"
echo "PaddlePaddle:     paddlepaddle-gpu==$PADDLE_VER from $PADDLE_INDEX"
echo ""

echo "Installing PyTorch..."
pip install torch torchvision --index-url "$TORCH_INDEX"

echo "Installing PaddlePaddle GPU..."
pip install "paddlepaddle-gpu==$PADDLE_VER" -i "$PADDLE_INDEX"

echo ""
echo "Verifying PyTorch GPU..."
python -c "
import torch
ok = torch.cuda.is_available()
print(f'  cuda available: {ok}')
if ok:
    print(f'  device: {torch.cuda.get_device_name(0)}')
else:
    raise SystemExit('ERROR: PyTorch cannot see the GPU. Check driver and CUDA version match.')
"

echo ""
echo "Verifying PaddlePaddle GPU..."
python -c "import paddle; paddle.utils.run_check()"

echo ""
echo "GPU dependencies installed successfully."
