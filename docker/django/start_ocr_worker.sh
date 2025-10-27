#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

echo "[START] Starting OCR processing worker..."
exec /code/.venv/bin/celery -A server worker -l INFO -Q ocr_processing --concurrency=1 --prefetch-multiplier=1
