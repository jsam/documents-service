#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

echo "[START] Patching POINTS model files..."
/code/.venv/bin/python /code/scripts/patch_points_model.py

echo "[START] Starting Celery worker..."
exec /code/.venv/bin/celery -A server worker -l INFO -Q points_ocr --concurrency=1 --prefetch-multiplier=1
