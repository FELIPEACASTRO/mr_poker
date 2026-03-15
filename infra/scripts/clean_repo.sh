#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
find . -type d -name '__pycache__' -prune -exec rm -rf {} +
rm -rf .pytest_cache
rm -f var/poker_ai_local.db
find var/reports -type f -delete 2>/dev/null || true
find var/batches -type f -delete 2>/dev/null || true
find var/regression -type f ! -name '.gitkeep' -delete 2>/dev/null || true

find var/models -type f ! -name '.gitkeep' -delete 2>/dev/null || true
find var/model_cards -type f ! -name '.gitkeep' -delete 2>/dev/null || true
find var/datasets -type f ! -name '.gitkeep' -delete 2>/dev/null || true
