# mr_pocker

Plataforma **local-first** para treino, avaliação, replay, analytics e coaching de **Texas Hold'em No-Limit Heads-Up** em ambiente próprio.

> Estado atual: **implementação consolidada até a Sprint 30**, pronta para **teste local**, **Docker**, **versionamento em GitHub** e continuação da trilha **Alpha -> Beta -> Produção**.

## O que está implementado
- engine determinístico heads-up no-limit
- dealing por `seed` ou `deck_prefix`
- ações legais: fold, check, call, bet, raise, all-in
- showdown, refund de aposta não coberta, settlement auditável
- replay canônico e persistência em SQLite
- baseline agent, adaptive baseline e policy-table local
- spot packs, taxonomy, export PHH-like v1
- dataset builder, splits estáveis e evaluation service
- adapter para solver externo e comparação solver-like
- session analytics, curriculum, coach report e hand review
- tournament round robin, readiness e release gates
- packaging local com Docker, compose, scripts e workflow de CI

## O que **não** está concluído
- integração com solver real de produção
- distilação supervisionada mais forte
- endurecimento completo para multiusuário/produção
- observabilidade e segurança de nível enterprise

## Quick start local (sem Docker)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
make unit
python -m uvicorn apps.api.main:create_app --factory --reload --port 8000
```

Abra:
- `http://localhost:8000/health`
- `http://localhost:8000/docs`

## Quick start com Docker
```bash
cp .env.example .env
docker compose up --build
```

Depois valide:
```bash
curl http://localhost:8000/health
```

## Validação recomendada
```bash
make unit
make integration
make validate
```

## Estrutura principal
- `apps/api/` - API FastAPI
- `packages/` - engine, agentes, features, datasets e utilitários
- `services/` - orquestração, benchmark, avaliação, coach, readiness
- `tests/` - unitários e integração
- `docs/` - documentação técnica, funcional e roadmap
- `infra/scripts/` - bootstrap, testes, smoke e limpeza

## Arquitetura adotada (interna)
- **Clean Architecture** em monólito modular: `domain` (`packages/*`), `application` (`services/*`) e `adapters` (`apps/api/*`, persistência SQLite e integrações externas).
- **CQRS interno** sem quebra de contrato HTTP:
  - write-side: `services/cqrs/bus.py` (`CommandBus`) para sessões, benchmarks, experimentos e torneios.
  - read-side: `services/cqrs/bus.py` (`QueryBus`) para sessão, traces, analytics, readiness, modelos e datasets.
- **SAGA síncrona in-process** em `services/cqrs/saga.py` com compensação reversa em falhas de fluxos compostos.
- **ACL interno** para integrações externas via `packages/external_solver/*`, isolando mapeamentos e DTOs solver-like.
- **Strategy + Factory** no agente baseline:
  - `packages/baseline_agent/strategy.py` para política de decisão e estimador de equity.
  - `packages/baseline_agent/factory.py` para criação de componentes por perfil de execução.
- **Repository + Unit of Work** na persistência SQLite com `SqliteUnitOfWork` em `packages/persistence/sqlite_store.py`.

## Estratégia de performance e SLO
- SLO alvo de API: **p95 < 3s** para endpoints `/v1` críticos, mantendo execução síncrona e contratos existentes.
- Otimizações principais:
  - avaliação/equity com caminho otimizado (`eval7`, com fallback determinístico local);
  - cache LRU por estado canônico em `packages/equity/monte_carlo.py`;
  - persistência em lote (`executemany`) para ações/snapshots/traces;
  - redução de overhead por ação em sessão com escrita por mão e compensação atômica.

## Comandos de validação
```bash
python -m pytest -q
python infra/scripts/audit_consistency_check.py --repo-root . --strict
python infra/scripts/architecture_fitness_checks.py --repo-root .
python -m mypy
python -m ruff check \
  services/cqrs services/session_service/service.py \
  packages/equity/monte_carlo.py packages/evaluator/hands.py \
  packages/persistence/sqlite_store.py packages/baseline_agent \
  packages/external_solver/adapter.py services/external_solver_service/service.py \
  infra/scripts/perf_gate_v1.py apps/api/contracts.py
python -m black --check \
  services/cqrs services/session_service/service.py \
  packages/equity/monte_carlo.py packages/evaluator/hands.py \
  packages/persistence/sqlite_store.py packages/baseline_agent \
  packages/external_solver/adapter.py services/external_solver_service/service.py \
  infra/scripts/perf_gate_v1.py apps/api/contracts.py
python infra/scripts/perf_gate_v1.py
```

## Documentos recomendados para começar
- `docs/88_Rigorous_Audit_2026_03_15.md`
- `docs/89_Gap_Register_and_Fixes.md`
- `docs/90_Local_Docker_Validation_Guide.md`
- `docs/87_Alpha_Beta_Production_Plan.md`
- `FINAL_AUDIT_STATUS.md`

## Execução honesta
Este projeto está **pronto para rodar localmente** e **pronto para teste**. Ele **não** está production-ready. O gate correto daqui é **Alpha controlado**, não deploy direto em produção.
