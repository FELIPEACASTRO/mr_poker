# Monitoring Guide

## Metrics (Prometheus)

Metrics are available at `GET /metrics` when `prometheus-client` is installed.

### Available Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `http_requests_total` | Counter | Total HTTP requests by method, path, status |
| `http_request_duration_seconds` | Histogram | Request latency |
| `active_hands` | Gauge | Currently active hand runtimes |

### Grafana Dashboard Setup

1. Add Prometheus data source pointing to your Prometheus instance
2. Import dashboard or create panels for:
   - Request rate (rate of `http_requests_total`)
   - P95 latency (`histogram_quantile(0.95, ...)`)
   - Error rate (status >= 400)
   - Active hands gauge

## Structured Logging

All logs are JSON-formatted with fields:
- `timestamp`: ISO 8601
- `level`: DEBUG/INFO/WARNING/ERROR
- `correlation_id`: Request correlation ID
- `module`: Source module
- `message`: Log message

### Log Aggregation

Pipe stdout to your log aggregator:
```bash
docker compose logs -f mr_poker_api | your-log-shipper
```

## Tracing (OpenTelemetry)

Enable with:
```
POKER_OTEL_ENABLED=true
POKER_OTEL_ENDPOINT=http://your-collector:4317
```

Traces include spans for:
- HTTP requests (auto-instrumented)
- Engine operations
- Database queries

## Alerting Recommendations

| Alert | Condition | Severity |
|-------|-----------|----------|
| High error rate | 5xx rate > 1% for 5min | Critical |
| High latency | P95 > 5s for 5min | Warning |
| Health check failing | /health/ready returns non-200 | Critical |
