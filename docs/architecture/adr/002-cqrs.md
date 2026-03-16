# ADR-002: CQRS Pattern for API

## Status
Accepted

## Context
The API has both read-heavy (hand state, analytics, traces) and write-heavy (session simulation, training) operations. We need clear separation of concerns.

## Decision
Implement CQRS with:
- `CommandBus` for write operations (sessions, benchmarks, training)
- `QueryBus` for read operations (hand state, analytics, session data)

## Consequences
- **Positive**: Clear separation of read/write paths
- **Positive**: Easy to add new commands/queries without modifying existing code
- **Positive**: Write operations can be wrapped in sagas for compensation
- **Negative**: Additional indirection layer
- **Mitigation**: Commands and queries are simple dataclasses, minimal overhead
