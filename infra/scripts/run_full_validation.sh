#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
mkdir -p var/reports
echo "[mr_pocker] compile check"
python -m compileall packages services apps >/dev/null
echo "[mr_pocker] pytest collect"
python -m pytest --collect-only -q > var/reports/test_collection.txt
echo "[mr_pocker] unit suite"
bash infra/scripts/run_unit_tests.sh
echo "[mr_pocker] integration suite"
bash infra/scripts/run_integration_tests.sh
echo "[mr_pocker] full validation finished"
