# API Contracts — Minimum

## Health
GET /health

Resposta:
```json
{"status": "ok"}
```

## Contratos que serão abertos na Sprint 01/02
- POST /session/start
- POST /hand/user-action
- GET /hand/{id}
- GET /hand/{id}/replay
- GET /hand/{id}/decision-trace

## Regra
Nenhuma rota de decisão pode embutir lógica de regra fora do engine.
