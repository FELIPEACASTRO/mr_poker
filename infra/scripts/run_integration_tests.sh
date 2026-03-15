#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
TEST_FILES=(
  tests/integration/test_health.py
  tests/integration/test_api_engine.py
  tests/integration/test_api_replay_and_auto.py
  tests/integration/test_api_benchmark.py
  tests/integration/test_api_experiments.py
  tests/integration/test_api_spot_packs_and_sessions.py
  tests/integration/test_api_taxonomy_export_analytics.py
  tests/integration/test_api_solver_models_coach.py
  tests/integration/test_api_sprint20_endpoints.py
  tests/integration/test_api_sprint30_endpoints.py
)
for test_file in "${TEST_FILES[@]}"; do
  echo "[mr_pocker] running ${test_file}"
  python -m pytest -q "$test_file"
done
echo "[mr_pocker] integration suite passed"
