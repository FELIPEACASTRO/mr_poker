# API Guide

Base URL: `http://localhost:8000`

## System

```bash
# Liveness check
curl http://localhost:8000/health/live

# Readiness check (DB, disk)
curl http://localhost:8000/health/ready

# Prometheus metrics
curl http://localhost:8000/metrics
```

## Hands

```bash
# Create new hand
curl -X POST http://localhost:8000/v1/hands/new \
  -H "Content-Type: application/json" \
  -d '{"stacks": [100, 100], "button_seat": 0}'

# Get hand state
curl http://localhost:8000/v1/hands/{hand_id}

# Apply action
curl -X POST http://localhost:8000/v1/hands/{hand_id}/actions \
  -H "Content-Type: application/json" \
  -d '{"action_type": "call", "amount": 2}'

# Auto-act (AI decides)
curl -X POST http://localhost:8000/v1/hands/{hand_id}/auto

# Get decision traces
curl http://localhost:8000/v1/hands/{hand_id}/traces

# Replay hand
curl http://localhost:8000/v1/hands/{hand_id}/replay

# Hand taxonomy
curl http://localhost:8000/v1/hands/{hand_id}/taxonomy

# Export PHH-like
curl http://localhost:8000/v1/hands/{hand_id}/export/phh-like

# Coach review
curl http://localhost:8000/v1/hands/{hand_id}/review
```

## Sessions

```bash
# Run H2H session
curl -X POST http://localhost:8000/v1/sessions/h2h \
  -H "Content-Type: application/json" \
  -d '{"num_hands": 50, "stacks": [100, 100]}'

# Get session
curl http://localhost:8000/v1/sessions/{session_id}

# Session analytics
curl http://localhost:8000/v1/sessions/{session_id}/analytics

# Session traces
curl http://localhost:8000/v1/sessions/{session_id}/traces

# Coach report
curl http://localhost:8000/v1/sessions/{session_id}/coach-report

# Opponent profile
curl http://localhost:8000/v1/sessions/{session_id}/opponent-profile

# Curriculum
curl http://localhost:8000/v1/sessions/{session_id}/curriculum

# Export session
curl http://localhost:8000/v1/sessions/{session_id}/export/phh-like
```

## Benchmarks

```bash
# Smoke test
curl -X POST http://localhost:8000/v1/benchmark/smoke \
  -H "Content-Type: application/json" \
  -d '{"num_hands": 50}'

# H2H benchmark
curl -X POST http://localhost:8000/v1/benchmark/h2h \
  -H "Content-Type: application/json" \
  -d '{"num_matches": 3, "hands_per_match": 50}'

# Adaptive benchmark
curl -X POST http://localhost:8000/v1/benchmark/adaptive \
  -H "Content-Type: application/json" \
  -d '{"num_hands": 40}'

# Round-robin tournament
curl -X POST http://localhost:8000/v1/tournaments/round-robin \
  -H "Content-Type: application/json" \
  -d '{"entrants": ["baseline", "adaptive"], "hands_per_match": 20}'
```

## ML

```bash
# Train model
curl -X POST http://localhost:8000/v1/ml/train

# Evaluate model
curl -X POST http://localhost:8000/v1/ml/evaluate/{model_id}/{dataset_id}

# Build dataset
curl -X POST http://localhost:8000/v1/ml/datasets/build

# Calibrate
curl -X POST http://localhost:8000/v1/ml/calibrate/{model_id}/{dataset_id}
```

## Governance

```bash
# Build model card
curl -X POST http://localhost:8000/v1/governance/model-card/{model_id}

# Evaluate release gate
curl -X POST http://localhost:8000/v1/governance/release-gate/{model_id}/{dataset_id}

# Build regression suite
curl -X POST http://localhost:8000/v1/governance/regression-suite
```

## Tasks (Async)

```bash
# Get task status
curl http://localhost:8000/v1/tasks/{task_id}

# List tasks
curl http://localhost:8000/v1/tasks
```
