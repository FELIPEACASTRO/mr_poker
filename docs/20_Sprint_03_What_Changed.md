# Sprint 03 — What Changed

## Engine
- Added `pot_segments` to snapshots.
- Added `_refund_uncalled_if_needed` before showdown/runout.
- Added segment-based showdown payout logic.

## Benchmarking
- Added `MatchHarness` for local agent-vs-agent runs.
- Added `BenchmarkService` and `/v1/benchmark/smoke`.

## Regression
- Added golden hands fixture file under `var/golden/golden_hands.json`.
- Added unit/integration tests for settlement and benchmark behavior.
