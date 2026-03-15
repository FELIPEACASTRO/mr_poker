# Local execution runbook

## Sem Docker
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
make unit
python -m uvicorn apps.api.main:create_app --factory --reload --port 8000
```

### Frontend React integrado em `/ui`
```bash
cd apps/web_ui
npm install
npm run dev
```

Em outro terminal para servir a versao buildada pelo FastAPI:
```bash
cd apps/web_ui
npm run build
cd ../..
python -m uvicorn apps.api.main:create_app --factory --reload --port 8000
```

Abra `http://localhost:8000/ui`.

## Com Docker
```bash
cp .env.example .env
docker compose up --build
```

## Verificações mínimas
- `GET /health`
- `GET /docs`
- `GET /ui`
- `POST /v1/hands/new`
- `POST /v1/benchmark/smoke`
- `GET /v1/system/readiness`

## Validação completa
```bash
make validate
```
