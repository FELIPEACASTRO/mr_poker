#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
TEST_FILES=(
  tests/unit/test_adaptive_benchmark.py
  tests/unit/test_baseline_agent.py
  tests/unit/test_baseline_agent_sprint04.py
  tests/unit/test_dataset_eval_external.py
  tests/unit/test_deck.py
  tests/unit/test_engine_rounds.py
  tests/unit/test_engine_setup.py
  tests/unit/test_equity.py
  tests/unit/test_experiment_runner.py
  tests/unit/test_features.py
  tests/unit/test_golden_hands.py
  tests/unit/test_harness.py
  tests/unit/test_opponent_profile_and_coach.py
  tests/unit/test_persistence_replay.py
  tests/unit/test_session_runner.py
  tests/unit/test_showdown.py
  tests/unit/test_sidepots.py
  tests/unit/test_solver_like_and_models.py
  tests/unit/test_spot_packs.py
  tests/unit/test_sprint30_services.py
  tests/unit/test_store_session_traces.py
  tests/unit/test_taxonomy_and_export.py
  tests/unit/test_tournament_curriculum_readiness.py
)
for test_file in "${TEST_FILES[@]}"; do
  echo "[mr_pocker] running ${test_file}"
  python -m pytest -q "$test_file"
done
echo "[mr_pocker] suite passed"
