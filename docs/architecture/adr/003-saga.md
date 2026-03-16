# ADR-003: Saga Pattern for Long-Running Operations

## Status
Accepted

## Context
Session simulation involves multiple steps (create session, simulate hands, persist results, generate report). If any step fails, we need to clean up partial state.

## Decision
Implement the Saga pattern with:
- `SagaStep` dataclass with action + compensate callbacks
- `InProcessSaga` executor with reverse-order compensation
- Optional persistent state via `SagaStore` (SQLite)
- Idempotency keys for exactly-once semantics
- Retry with exponential backoff per step

## Consequences
- **Positive**: Automatic cleanup on failure
- **Positive**: Persistent state enables recovery after crashes
- **Positive**: Idempotency prevents duplicate operations
- **Negative**: Compensation logic must be manually defined
