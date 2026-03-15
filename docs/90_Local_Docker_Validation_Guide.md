# Guia de validação local em Docker

## Subir o ambiente
```bash
docker compose up --build
```

## Verificações mínimas
1. `GET /health` responde `ok`.
2. `GET /docs` abre normalmente.
3. `POST /v1/hands/new` cria uma mão.
4. `POST /v1/benchmark/smoke` executa smoke benchmark.
5. `POST /v1/tournaments/round-robin` executa round robin local.
6. `GET /v1/system/readiness` responde com snapshot.

## Testes locais
```bash
make unit
make integration
```

## Observação honesta
O objetivo deste pacote é **rodar localmente com consistência**. Ele não substitui o hardening necessário antes de um Beta fechado ou Produção inicial.
