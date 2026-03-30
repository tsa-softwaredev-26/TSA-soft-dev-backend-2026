#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/opt/spaitra/TSA-soft-dev-backend-2026}"

chmod 600 /opt/spaitra/.env 2>/dev/null || true
chmod 600 /opt/spaitra/.ocr.env 2>/dev/null || true

chmod 700 "$REPO_ROOT/data" 2>/dev/null || true
chmod 600 "$REPO_ROOT"/data/*.db 2>/dev/null || true

chmod 700 "$REPO_ROOT/models" 2>/dev/null || true
chmod 640 "$REPO_ROOT"/models/*.pt 2>/dev/null || true

chmod 750 "$REPO_ROOT/logs" 2>/dev/null || true
chmod 640 "$REPO_ROOT"/logs/*.log 2>/dev/null || true

chmod 600 ~/.cache/huggingface/token 2>/dev/null || true

echo "Permissions hardened for $REPO_ROOT"
