#!/usr/bin/env bash
set -euo pipefail

# Migrate core runtime to Python 3.12 so SQLCipher bindings can be enabled.
# This script does not touch venv-ocr and does not delete the previous venv-core.
#
# Usage:
#   sudo bash deploy/migrate_core_to_py312.sh /opt/spaitra/TSA-soft-dev-backend-2026
#
# Prereqs:
#   - python3.12, python3.12-venv, python3.12-dev installed on host
#   - libsqlcipher-dev installed

REPO_ROOT="${1:-/opt/spaitra/TSA-soft-dev-backend-2026}"
ENV_FILE="/opt/spaitra/.env"
VENV_CORE="$REPO_ROOT/venv-core"
TS="$(date +%Y%m%d%H%M%S)"
NEW_VENV="$REPO_ROOT/venv-core-py312-$TS"

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "Missing repo root: $REPO_ROOT" >&2
  exit 1
fi

if ! command -v python3.12 >/dev/null 2>&1; then
  echo "python3.12 not found. Install python3.12 packages first." >&2
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE" >&2
  exit 1
fi

echo "[1/7] Stop core service"
systemctl stop spaitra-core

echo "[2/7] Create new core venv with Python 3.12"
python3.12 -m venv "$NEW_VENV"

echo "[3/7] Install core deps"
source "$NEW_VENV/bin/activate"
pip install --upgrade pip
pip install -e "$REPO_ROOT[core]"

echo "[4/7] Validate SQLCipher module import"
python - <<'PY'
import importlib.util
ok = bool(importlib.util.find_spec("pysqlcipher3")) or bool(importlib.util.find_spec("sqlcipher3"))
print("sqlcipher_module_available", ok)
if not ok:
    raise SystemExit(2)
PY

echo "[5/7] Ensure DB_ENCRYPTION_KEY is set"
if ! grep -q '^DB_ENCRYPTION_KEY=' "$ENV_FILE"; then
  DBK="$(python - <<'PY'
import secrets
print(secrets.token_hex(32))
PY
)"
  echo "DB_ENCRYPTION_KEY=$DBK" >> "$ENV_FILE"
fi

if grep -q '^DB_ENCRYPTION_KEY=$' "$ENV_FILE"; then
  DBK="$(python - <<'PY'
import secrets
print(secrets.token_hex(32))
PY
)"
  sed -i "s/^DB_ENCRYPTION_KEY=$/DB_ENCRYPTION_KEY=$DBK/" "$ENV_FILE"
fi

chown spaitra:spaitra "$ENV_FILE"
chmod 600 "$ENV_FILE"

echo "[6/7] Swap venv-core atomically"
if [[ -d "$VENV_CORE" ]]; then
  mv "$VENV_CORE" "${VENV_CORE}.bak.${TS}"
fi
mv "$NEW_VENV" "$VENV_CORE"

echo "[7/7] Start core service"
systemctl restart spaitra-core
systemctl is-active spaitra-core

echo "Done. Verify:"
echo "  curl -s -o /dev/null -w '%{http_code}\\n' http://127.0.0.1:5000/health"
