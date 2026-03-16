# Architecture Overview

## System Architecture

mr_poker is a local-first Texas Hold'em No-Limit Heads-Up AI training platform built with clean architecture principles.

```
┌─────────────────────────────────────────────────┐
│                    Adapters                       │
│  ┌──────────────┐  ┌─────────────────────────┐  │
│  │  FastAPI API  │  │  React Web UI (SPA)     │  │
│  │  apps/api/    │  │  apps/web_ui/           │  │
│  └──────┬───────┘  └─────────────────────────┘  │
├─────────┼───────────────────────────────────────┤
│         │         Application Services            │
│  ┌──────┴──────────────────────────────────────┐ │
│  │  CQRS (CommandBus + QueryBus)               │ │
│  │  services/cqrs/                             │ │
│  ├─────────────────────────────────────────────┤ │
│  │  28 Domain Services                         │ │
│  │  services/session_service/                  │ │
│  │  services/benchmark_service/                │ │
│  │  services/decision_service/                 │ │
│  │  services/game_orchestrator/                │ │
│  │  services/tournament_service/               │ │
│  │  ... (see full list below)                  │ │
│  └──────┬──────────────────────────────────────┘ │
├─────────┼───────────────────────────────────────┤
│         │         Domain Packages                 │
│  ┌──────┴──────────────────────────────────────┐ │
│  │  Engine     │  Equity    │  Features        │ │
│  │  Evaluator  │  Strategy  │  Training        │ │
│  │  Persistence│  Events    │  Auth            │ │
│  │  Metrics    │  Tracing   │  Audit           │ │
│  │  packages/                                  │ │
│  └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

## Key Patterns

### CQRS (Command Query Responsibility Segregation)
- **Commands** modify state: `SessionH2HCommand`, `SmokeBenchmarkCommand`, etc.
- **Queries** read state: `get_session`, `get_session_analytics`, etc.
- Implementation in `services/cqrs/bus.py`

### Saga Pattern
- Orchestrates multi-step operations with compensation
- Persistent state in SQLite (`saga_state` table)
- Idempotency key support for exactly-once semantics
- Implementation in `services/cqrs/saga.py`

### Repository Pattern
- Abstract interfaces: `HandRepository`, `SessionRepository`
- SQLite implementation: `SqliteHandStore`
- Unit of Work pattern for batch persistence

### Event Sourcing
- Domain events: `HandStarted`, `ActionApplied`, `HandCompleted`
- Append-only event store in SQLite
- Implementation in `packages/events/`

## Layer Dependencies

```
apps/ → services/ → packages/
  ↓        ↓           ↓
configs    cqrs     engine, persistence, features, equity
```

**Rule**: Inner layers never import from outer layers.

## Data Flow

1. **HTTP Request** → FastAPI Router → Auth Middleware → Container
2. **Command** → CommandBus → Service → Engine/Store
3. **Query** → QueryBus → Store → Response
4. **Saga** → Steps → Compensations on failure

## Configuration

- Environment variables override config files
- Config files: `configs/app.{env}.json`
- Settings loaded via `packages/config/settings.py`
- See `.env.example` for all variables

## Security

- JWT authentication (optional, configurable)
- Rate limiting per IP (token bucket)
- Security headers (HSTS, CSP, X-Frame-Options)
- Audit logging for auth, access, and admin events
- Input validation via Pydantic with strict limits

## Observability

- Structured JSON logging with correlation IDs
- Prometheus metrics at `/metrics`
- OpenTelemetry tracing (optional)
- Health checks: `/health/live`, `/health/ready`
