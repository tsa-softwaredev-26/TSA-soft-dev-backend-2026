#!/bin/bash
set -e
cd /opt/spaitra/TSA-soft-dev-backend-2026
source venv-core/bin/activate
python -m visual_memory.utils.memory_monitor --cleanup --max-age 2 --log-only
