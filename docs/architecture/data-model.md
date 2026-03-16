# Data Model

## SQLite Tables

### Core Tables

| Table | Description |
|-------|------------|
| `hands` | Hand metadata (id, session_id, button_seat, stacks, seed) |
| `hand_actions` | Action log per hand (actor_seat, action_type, amount) |
| `hand_snapshots` | State snapshots at key moments |
| `sessions` | Session metadata and config |
| `decision_traces` | AI decision trace records |

### Infrastructure Tables

| Table | Description |
|-------|------------|
| `audit_log` | Append-only audit trail |
| `saga_state` | Persistent saga state for long-running operations |
| `events` | Domain event store (event sourcing) |

## Relationships

```
sessions 1──N hands 1──N hand_actions
                   1──N hand_snapshots
                   1──N decision_traces
```

## Key Invariants

1. **Chip Conservation**: `sum(stacks) + pot == initial_total` at all times
2. **Action Ordering**: Actions within a hand are strictly ordered
3. **Terminal State**: A terminal hand has exactly one winner (or showdown for split)
4. **Audit Trail**: All auth events, data access, and admin actions are logged
