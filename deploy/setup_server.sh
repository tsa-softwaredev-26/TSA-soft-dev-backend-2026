#!/usr/bin/env bash
# Spaitra server setup - Debian 12 (bookworm)
# Run as root: bash deploy/setup_server.sh
# Installs system packages, creates service user, sets up venv, downloads weights,
# configures systemd, and optionally installs the srv.us tunnel service.
set -euo pipefail

REPO_URL="https://github.com/tsa-softwaredev-26/TSA-soft-dev-backend-2026.git"
INSTALL_DIR="/opt/spaitra"
SERVICE_USER="spaitra"
PYTHON="python3.11"

# CUDA variant: "cpu", "cu118", or "cu121"
TORCH_CUDA="${TORCH_CUDA:-cpu}"

# ---- helpers ----

info()  { echo "[INFO]  $*"; }
error() { echo "[ERROR] $*" >&2; exit 1; }

require_root() {
    [ "$(id -u)" -eq 0 ] || error "This script must be run as root."
}

# ---- system packages ----

install_system_packages() {
    info "Installing system packages..."
    apt-get update -q
    apt-get install -y -q \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        git build-essential wget curl \
        libglib2.0-0 libsm6 libxrender1 libxext6 \
        libgomp1 \
        libjpeg-dev libpng-dev libtiff-dev libwebp-dev \
        ffmpeg
}

# ---- service user ----

create_user() {
    if id "$SERVICE_USER" &>/dev/null; then
        info "User $SERVICE_USER already exists."
    else
        info "Creating service user $SERVICE_USER..."
        adduser --system --group --home "$INSTALL_DIR" "$SERVICE_USER"
    fi
    mkdir -p "$INSTALL_DIR"
    chown "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
}

# ---- clone ----

clone_repo() {
    if [ -d "$INSTALL_DIR/.git" ]; then
        info "Repository already cloned. Pulling latest..."
        sudo -u "$SERVICE_USER" git -C "$INSTALL_DIR" pull
    else
        info "Cloning repository..."
        sudo -u "$SERVICE_USER" git clone "$REPO_URL" "$INSTALL_DIR"
    fi
}

# ---- Python venv ----

setup_venv() {
    info "Creating Python venv..."
    sudo -u "$SERVICE_USER" "$PYTHON" -m venv "$INSTALL_DIR/venv"

    info "Installing PyTorch (TORCH_CUDA=$TORCH_CUDA)..."
    case "$TORCH_CUDA" in
        cu121)
            sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install \
                torch torchvision --index-url https://download.pytorch.org/whl/cu121
            ;;
        cu118)
            sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install \
                torch torchvision --index-url https://download.pytorch.org/whl/cu118
            ;;
        *)
            sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install \
                torch torchvision --index-url https://download.pytorch.org/whl/cpu
            ;;
    esac

    info "Installing package..."
    sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install -e "$INSTALL_DIR"
}

# ---- env file ----

setup_env() {
    if [ -f "$INSTALL_DIR/.env" ]; then
        info ".env already exists, skipping."
    else
        info "Creating .env from example..."
        sudo -u "$SERVICE_USER" cp "$INSTALL_DIR/deploy/env.example" "$INSTALL_DIR/.env"
        echo
        echo "IMPORTANT: Edit $INSTALL_DIR/.env and set API_KEY before starting the service."
        echo "  nano $INSTALL_DIR/.env"
        echo
    fi
}

# ---- HuggingFace login ----

hf_login() {
    info "HuggingFace login (required for gated models)..."
    echo
    echo "Two models require accepted access on huggingface.co:"
    echo "  - facebook/dinov3-vits16-pretrain-lvd1689m"
    echo "  - IDEA-Research/grounding-dino-base"
    echo
    echo "Accept both licenses, then paste your HuggingFace token when prompted."
    sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install -q huggingface_hub
    sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/huggingface-cli" login
}

# ---- download weights ----

download_weights() {
    info "Downloading model weights (this may take 10-30 minutes)..."
    sudo -u "$SERVICE_USER" bash -c \
        "cd $INSTALL_DIR && $INSTALL_DIR/venv/bin/python setup_weights.py"
}

# ---- systemd ----

install_service() {
    info "Installing systemd service..."
    cp "$INSTALL_DIR/deploy/spaitra.service" /etc/systemd/system/spaitra.service
    systemctl daemon-reload
    systemctl enable spaitra
    systemctl start spaitra
    echo
    info "Service started. Watch logs with:"
    echo "  journalctl -u spaitra -f"
}

# ---- srv.us tunnel (optional) ----

install_tunnel() {
    read -r -p "Install srv.us tunnel service? [y/N] " yn
    case "$yn" in
        [Yy]*)
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
            systemctl daemon-reload
            systemctl enable --now spaitra-tunnel
            info "Tunnel service installed."
            ;;
        *)
            info "Skipping tunnel setup."
            ;;
    esac
}

# ---- main ----

require_root
install_system_packages
create_user
clone_repo
setup_venv
setup_env
hf_login
download_weights
install_service
install_tunnel

echo
info "Setup complete."
echo "  Test: curl http://127.0.0.1:5000/health"
echo "  Logs: journalctl -u spaitra -f"
