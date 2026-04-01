#!/usr/bin/env bash
# Install systemd service files with the correct repo path substituted in.
# Usage: sudo bash deploy/install.sh [/path/to/repo]
#
# Defaults to the directory containing this script's parent (the repo root).
# Pass an explicit path if you cloned elsewhere.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${1:-$(dirname "$SCRIPT_DIR")}"
ENV_DIR="${2:-/opt/spaitra}"

echo "Installing services for repo: $REPO_ROOT"
echo "Env files expected at:        $ENV_DIR/.env and $ENV_DIR/.ocr.env"
echo ""

install_service() {
    local name="$1"
    local src="$SCRIPT_DIR/$name"
    local dst="/etc/systemd/system/$name"

    sed "s|__REPO_ROOT__|$REPO_ROOT|g" "$src" > "$dst"
    echo "  Wrote $dst"
}

install_service spaitra-core.service
install_service spaitra-ocr.service

systemctl daemon-reload
echo ""
echo "Done. Enable and start with:"
echo "  systemctl enable --now spaitra-ocr"
echo "  systemctl enable --now spaitra-core"
echo ""
echo "To disable core via config only (without removing service):"
echo "  set ENABLE_CORE_SERVICE=0 in /opt/spaitra/.env then:"
echo "  systemctl daemon-reload && systemctl restart spaitra-core"
