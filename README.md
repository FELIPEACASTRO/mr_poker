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

## Documentos recomendados para começar
- `docs/88_Rigorous_Audit_2026_03_15.md`
- `docs/89_Gap_Register_and_Fixes.md`
- `docs/90_Local_Docker_Validation_Guide.md`
- `docs/87_Alpha_Beta_Production_Plan.md`
- `FINAL_AUDIT_STATUS.md`

## Execução honesta
Este projeto está **pronto para rodar localmente** e **pronto para teste**. Ele **não** está production-ready. O gate correto daqui é **Alpha controlado**, não deploy direto em produção.
