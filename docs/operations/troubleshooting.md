# Troubleshooting Guide

## Common Issues

### Database Locked
**Symptom**: `sqlite3.OperationalError: database is locked`
**Cause**: Concurrent write operations
**Fix**: Increase SQLite timeout or reduce concurrent writes. The app uses `timeout=30` by default.

### Import Errors
**Symptom**: `ModuleNotFoundError: No module named 'packages'`
**Fix**: Install in development mode: `pip install -e .`

### Auth Failures
**Symptom**: `401 Unauthorized` on all requests
**Fix**: Set `POKER_AUTH_ENABLED=false` or provide valid JWT token.

### Rate Limiting
**Symptom**: `429 Too Many Requests`
**Fix**: Increase `POKER_RATE_LIMIT_RPM` or wait for the token bucket to refill.

### Memory Usage
**Symptom**: High memory consumption during sessions
**Cause**: Large number of hand runtimes held in memory
**Fix**: Reduce `num_hands` per session or increase available RAM.

### Frontend Not Loading
**Symptom**: 404 on `/` or static assets
**Fix**: Build the frontend: `cd apps/web_ui && npm run build`

## Diagnostic Commands

```bash
# Check API health
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready

# Check database
sqlite3 var/poker_ai_local.db ".tables"
sqlite3 var/poker_ai_local.db "SELECT COUNT(*) FROM hands"

# Check logs
docker compose logs --tail=100 mr_poker_api

# Run tests
bash infra/scripts/run_unit_tests.sh
```
