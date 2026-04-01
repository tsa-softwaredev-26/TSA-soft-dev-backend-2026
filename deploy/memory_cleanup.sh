#!/bin/bash
set -e

REPO_ROOT="${1:-/opt/spaitra/TSA-soft-dev-backend-2026}"
cd "$REPO_ROOT"
source "$REPO_ROOT/venv-core/bin/activate"
python -m visual_memory.utils.memory_monitor --cleanup --max-age 2 --log-only
