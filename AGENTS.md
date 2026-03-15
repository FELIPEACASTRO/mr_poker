# mr_poker Agent Guide

## Objetivo

Este repositorio contem o `mr_poker`: um laboratorio local-first, Python-first, focado em heads-up NLHE, com runtime de engine, baseline/adaptive agents, treino local de policy table, avaliacao offline, governanca basica e empacotamento para Alpha controlado.

O objetivo deste arquivo e acelerar navegacao, reduzir ambiguidades e fixar a ordem de verdade tecnica do sistema.

## Mapa do repositorio

- `apps/api/main.py`: composicao principal do runtime FastAPI e superficie HTTP canonica.
- `packages/*`: dominio, engine, modelos, features, evaluator, equity, solver-like, policy table, persistencia e export.
- `services/*`: orquestracao entre pacotes, persistencia, datasets, modelos, analytics, governance e release workflows.
- `tests/unit/*`: cobertura de engine, pacotes, services e regressao de comportamento local.
- `tests/integration/*`: cobertura do contrato HTTP exposto por `apps/api/main.py`.
- `docs/*`: historico do projeto, especificacoes, auditorias, ADRs, diagramas e runbooks.
- `doc/PACOTE_EXECUTIVO_ALPHA_BETA_PRODUCAO/*`: pacote executivo para promocao local -> Alpha -> Beta -> producao inicial.
- `infra/scripts/*`: scripts operacionais e de validacao.
- `configs/*`: configuracoes locais e template de producao.
- `var/*`: dados de runtime, golden hands, solver sample, datasets, modelos, relatorios, model cards e suites de regressao.

## Ordem de verdade

Quando houver conflito documental, use esta ordem:

1. Codigo executavel.
2. Testes automatizados.
3. Contratos expostos pela API e pelo schema SQLite.
4. README e auditorias recentes.
5. Documentos historicos de sprint.

## Verdades canonicas do sistema

- O ponto de composicao do sistema e `create_app()` em `apps/api/main.py`.
- O banco local padrao e `var/poker_ai_local.db`, gerido por `packages/persistence/sqlite_store.py`.
- O sistema e local-first; readiness documental existe, mas o estado operacional real ainda e Alpha/local lab.
- O solver externo atual nao e um solver real online; a integracao vigente e arquivo/mock local.
- O `Makefile` assume shell `bash`; em Windows/PowerShell prefira executar diretamente os comandos Python e Docker equivalentes.
- `ruff` esta configurado em `pyproject.toml`, mas nao deve ser assumido como disponivel no `PATH` do ambiente.

## Comandos canonicos

- API local: `python -m uvicorn apps.api.main:create_app --factory --reload --port 8000`
- Testes: `python -m pytest -q`
- Docker local: `docker compose up --build`
- Healthcheck: `python infra/scripts/healthcheck.py`

## Regras de navegacao

- Mantenha a API fina: rotas HTTP traduzem request/response, services orquestram, packages concentram logica.
- Ao documentar ou alterar algo, explicite produtor, consumidor, persistencia e artefatos em `var/`.
- Nao promova o sistema para "producao" documental sem reconciliar os gaps em `docs/99_Gap_and_Risk_Register.md`.
- Ao analisar comportamento, confirme se ele existe em testes unitarios, testes de integracao, ou apenas em documentacao historica.
- Trate `doc/` como camada executiva e `docs/` como camada tecnico-operacional.

