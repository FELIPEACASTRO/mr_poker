# Deployment Guide

## Docker (Recommended)

```bash
# Build
docker compose build

# Run
docker compose up -d

# Check health
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
```

## Bare Metal

```bash
# Install
pip install -e .

# Configure
cp .env.example .env
# Edit .env as needed

# Run
python -m uvicorn apps.api.main:create_app --factory --host 0.0.0.0 --port 8000
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `POKER_AI_ENV` | `local` | Environment (local/prod/test) |
| `POKER_AI_DB` | `var/poker_ai_local.db` | SQLite database path |
| `POKER_AUTH_ENABLED` | `false` | Enable JWT authentication |
| `POKER_JWT_SECRET` | `dev-secret-change-me` | JWT signing secret |
| `POKER_CORS_ORIGINS` | `*` | Allowed CORS origins |
| `POKER_RATE_LIMIT_RPM` | `120` | Rate limit (requests/minute) |
| `POKER_LOG_LEVEL` | `INFO` | Logging level |
| `POKER_OTEL_ENABLED` | `false` | Enable OpenTelemetry tracing |
| `POKER_OTEL_ENDPOINT` | `http://localhost:4317` | OTLP endpoint |

## Health Checks

- **Liveness**: `GET /health/live` - Returns 200 if process is running
- **Readiness**: `GET /health/ready` - Checks DB connectivity and disk space

## Docker Security

The Docker image runs as non-root user (`appuser`) with:
- Read-only filesystem (`read_only: true`)
- No new privileges (`no-new-privileges:true`)
- tmpfs for `/tmp`
