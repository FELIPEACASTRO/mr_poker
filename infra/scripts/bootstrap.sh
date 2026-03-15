#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
python3 -m pip install -e .[dev]
echo "[mr_pocker] bootstrap concluído"
echo "Próximos passos:"
echo "  1) make unit"
echo "  2) make integration"
echo "  3) make docker-up"
