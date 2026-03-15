# System Knowledge Dossier

## Escopo e base usada

Este dossie consolida o conhecimento tecnico e funcional do `mr_poker` a partir do conteudo versionado do repositorio e da observacao do workspace local em `var/`.

- Base versionada autoritativa: `git ls-files` em `2026-03-14`.
- Contagem autoritativa atual: `367` arquivos versionados.
- Reconciliacao de baseline: referencias anteriores de contagem estavam stale e foram substituidas pela contagem tracked atual do workspace.
- Ordem de verdade aplicada: codigo > testes > contratos API/DB > README/auditorias atuais > docs historicos de sprint.

## 1. Resumo executivo tecnico

`mr_poker` e um laboratorio local-first para heads-up NLHE com quatro blocos principais:

1. Runtime de mao e simulacao.
2. Agentes e heuristicas locais de decisao.
3. Pipeline local de dataset -> modelo -> avaliacao -> calibracao -> governance.
4. Empacotamento, relatorios e readiness para Alpha controlado.

O sistema e operacional localmente, tem API FastAPI funcional, persistencia SQLite, geracao de datasets, treino de policy table, avaliacao offline, calibracao e model cards. O sistema nao esta tecnicamente hardenizado para producao real, apesar de existirem proxies locais de readiness.

## 2. Topologia canonica do repositorio

| Escopo | Arquivos versionados | Papel principal |
| --- | ---: | --- |
| `(root)` | 14 | bootstrap, packaging, status, README, metadados do projeto e governanca de escopo |
| `.github` | 2 | workflows CI |
| `apps` | 2 | composicao HTTP FastAPI e regras de escopo da camada API |
| `configs` | 2 | configuracao local e template de producao |
| `doc` | 26 | pacote executivo Alpha/Beta/Producao |
| `docs` | 116 | especificacoes, historico, auditorias, ADRs, diagramas e parecer frontier |
| `infra` | 8 | scripts operacionais locais e regras de escopo |
| `packages` | 53 | nucleo de dominio, IA, dados e persistencia |
| `services` | 56 | ownership funcional e orquestracao |
| `tests` | 34 | suites unitarias e de integracao com regras de escopo |
| `var` | 8 | placeholders versionados e artefatos canonicos minimos de model cards |

## 3. Composition root e runtime HTTP

O composition root esta em `apps/api/main.py`, com `create_app()` na linha `102`. Entre as linhas `103-132`, a aplicacao instancia:

- `GameEngine`
- `BaselineAgent`
- `SqliteHandStore`
- `ReplayService`
- `BenchmarkService`
- `ExperimentRunner`
- `SessionRunner`
- `SpotPackService`
- `TaxonomyService`
- `SessionAnalyticsService`
- `ExportService`
- `SolverLabelService`
- `ModelService`
- `OpponentProfileService`
- `CoachService`
- `DatasetService`
- `EvaluationService`
- `ExternalSolverService`
- `TournamentService`
- `CurriculumService`
- `ReadinessService`
- `GovernanceService`
- `ReleaseGateService`
- `BatchGenerationService`
- `RegressionSuiteService`
- `CalibrationService`
- `DeployService`
- `ReleaseNotesService`
- `AlphaCandidateService`

As rotas FastAPI ficam nas linhas `135-489` e cobrem:

- health e raiz
- ciclo de vida de hand
- auto-play baseline
- traces, replay, taxonomy e export
- spot packs
- benchmark e experimentos
- sessions e analytics
- solver-like e external solver
- train/eval/list de modelos
- datasets
- tournaments, curriculum e readiness
- opponent profile e coach
- governance, release gate, batch generation, regression, calibration
- deploy manifest, release notes e alpha candidate

Adicionalmente, a superficie web integrada expõe:
- `GET /ui` (entrypoint da SPA React)
- `GET /ui/{path:path}` (assets e fallback de roteamento client-side)

## 4. Nucleo funcional do poker

### 4.1 Engine e estado

- `packages/engine/engine.py`
  - `HandRuntime` na linha `13`
  - `GameEngine` na linha `19`
  - responsabilidade: iniciar mao, postar blinds, validar acoes, transicionar streets, resolver showdown, side pots e refund de all-in nao pago.
- `packages/engine/models.py`
  - `Card@13`, `PlayerState@34`, `ActionEvent@45`, `HandState@54`
  - responsabilidade: modelo estrutural da mao.
- `packages/engine/deck.py`
  - `Deck@9`
  - responsabilidade: baralho deterministico com suporte a `seed` e `deck_prefix`.

### 4.2 Avaliacao e equity

- `packages/evaluator/hands.py`
  - `evaluate_five@34`, `best_hand_rank@70`, `hand_label@76`
  - responsabilidade: ranking de maos e label legivel.
- `packages/equity/monte_carlo.py`
  - `estimate_equity@18`
  - responsabilidade: equity proxy por Monte Carlo para heuristicas locais.

### 4.3 Features e taxonomia

- `packages/features/minimum.py`
  - `MinimumDecisionFeatures@9`, `derive_minimum_features@20`
- `packages/features/poker.py`
  - `BaselineFeatures@12`, `canonical_hole@29`, `_board_texture@39`, `derive_baseline_features@53`
- `packages/taxonomy/spot_taxonomy.py`
  - `classify_trace@30`, `classify_hand@93`
- `packages/solver_like/buckets.py`
  - `bucketize_spot@27`

Esses modulos sustentam a taxonomia de spots, as explicacoes locais e o treinamento da policy table.

## 5. Pilha de IA e modelo local

### 5.1 Agentes

- `packages/baseline_agent/agent.py`
  - `BaselineAgent@13`
  - heuristica principal para auto-play, baseada em features + equity local + regras por street.
- `packages/adaptive_agent/agent.py`
  - `AdaptiveBaselineAgent@10`
  - ajusta thresholds do baseline com base no perfil de agressao observado.
- `packages/policy_agent/agent.py`
  - `PolicyTableAgent@9`
  - executa uma policy table treinada localmente, com fallback implicito para heuristica.

### 5.2 Contratos e traces

- `packages/baseline_agent/contracts.py`
  - `AgentDecision@11`
- `packages/logging_schema/decision_trace.py`
  - `DecisionTrace@11`

Os traces sao o elo entre decisao online, persistencia local, dataset builder, avaliacao e governanca.

### 5.3 Solver-like e treino local

- `packages/solver_like/normalize.py`
  - `normalize_runtime@8`, `normalize_snapshot_trace@37`
- `packages/solver_like/labeler.py`
  - `SolverLikeLabeler@9`
- `packages/policy_table/trainer.py`
  - `PolicyTableTrainer@12`
- `packages/policy_table/model.py`
  - `PolicyTableModel@8`
- `packages/policy_table/registry.py`
  - `PolicyRegistry@10`

Essa trilha normaliza spots, aplica rotulacao solver-like proxy, agrega buckets e serializa modelos JSON em `var/models`.

### 5.4 Dados, avaliacao e governance

- `packages/dataset_builder/builder.py`
  - `stable_split@14`, `DatasetBuilder@23`
- `packages/evaluation/taxonomy_eval.py`
  - `DatasetEvaluation@9`, `evaluate_policy_model@30`
- `services/evaluation_service/service.py`
  - `EvaluationService@9`
- `services/calibration_service/service.py`
  - `CalibrationService@6`
- `services/governance_service/service.py`
  - `GovernanceService@8`
- `services/release_gate_service/service.py`
  - `ReleaseGateService@7`
- `services/alpha_candidate_service/service.py`
  - `AlphaCandidateService@8`

O fluxo de governanca atual e totalmente local, persistido em arquivos JSON sob `var/`.

## 6. Services e ownership funcional

Os `services/*` sao a camada de ownership funcional do sistema:

- simulacao e runtime: `benchmark_service`, `experiment_runner`, `session_service`, `replay_service`
- spots e taxonomy: `spot_pack_service`, `solver_label_service`, `taxonomy_service`
- dados e modelos: `dataset_service`, `model_service`, `evaluation_service`, `calibration_service`
- analise e estudo: `analytics_service`, `coach_service`, `curriculum_service`, `opponent_profile_service`
- promocao e operacao: `governance_service`, `release_gate_service`, `deploy_service`, `release_notes_service`, `alpha_candidate_service`
- extensoes de suporte: `external_solver_service`, `tournament_service`, `readiness_service`, `batch_generation_service`, `regression_suite_service`

Tres pacotes de service estao presentes apenas como placeholder de namespace no estado versionado atual:

- `services/decision_service/__init__.py`
- `services/explanation_service/__init__.py`
- `services/game_orchestrator/__init__.py`

Eles nao devem ser tratados como features implementadas.

## 7. Persistencia e artefatos

### 7.1 SQLite

`packages/persistence/sqlite_store.py` define `SqliteHandStore@9`. A inicializacao (`_init_db@21`) aplica:

- `PRAGMA journal_mode=WAL`
- `PRAGMA synchronous=NORMAL`
- `PRAGMA foreign_keys=ON`

Tabelas canonicas:

- `hands`
- `action_log`
- `snapshots`
- `decision_traces`
- `sessions`

### 7.2 Artefatos em `var/`

Versionados:

- `var/golden/golden_hands.json`
- `var/external_solver/sample_solver_v1.json`
- placeholders `.gitkeep` para datasets, models, model_cards e regression

Observados no workspace local:

- `var/poker_ai_local.db`
- `var/datasets/<dataset_id>/*`
- `var/models/<model_id>.json`
- `var/model_cards/<model_id>.card.json`
- `var/reports/*.json`
- `var/regression/suite_api.json`
- `var/batches/*.json`

## 8. Testes, packaging e execucao

### 8.1 Baseline validado

- `pytest -q`: `40` testes passando no baseline observado.
- Integracao API: `tests/integration/*`.
- Unidade e regressao comportamental: `tests/unit/*`.

### 8.2 Packaging

- `Dockerfile` presente.
- `docker-compose.yml` presente.
- `pyproject.toml` configura dependencias e `ruff`.
- `Makefile` contem atalhos para lint, tests, API e Docker.

### 8.3 Caveats operacionais

- `Makefile` assume `bash`.
- `ruff` pode nao estar disponivel no `PATH` do ambiente.
- readiness e deploy manifest sao proxies documentais locais, nao hardening de producao real.

## 9. Estado real do sistema

### O que esta realmente implementado

- engine jogavel e deterministico
- persistencia SQLite
- baseline agent
- adaptive benchmark
- replay, taxonomy e export
- spot packs
- sessions, analytics e coach
- dataset builder
- policy table training
- evaluation por split e taxonomy
- calibracao proxy
- model cards, release gate e alpha candidate
- tournament local
- assets Docker/local

### O que ainda nao deve ser vendido como concluido

- solver externo real e integrado a runtime decisoria
- observabilidade produtiva
- segredos, controles de runtime e hardening operacional
- readiness honesta para Beta fechado ou producao inicial

## 10. Saidas permanentes desta implementacao

Este plano materializou a camada permanente de conhecimento em:

- `AGENTS.md`
- `apps/api/AGENTS.md`
- `packages/AGENTS.md`
- `services/AGENTS.md`
- `tests/AGENTS.md`
- `docs/AGENTS.md`
- `infra/AGENTS.md`
- `docs/97_File_By_File_Inventory.md`
- `docs/98_Data_AI_DB_Integration_Map.md`
- `docs/99_Gap_and_Risk_Register.md`
