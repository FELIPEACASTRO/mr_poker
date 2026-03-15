# Data, AI, DB and Integration Map

## Escopo

Este documento canoniza:

- schema SQLite local
- artefatos `var/*`
- fluxo de dados da trilha de IA
- ownership dos endpoints FastAPI
- contratos JSON observados no runtime local

## 1. Composition root e ownership por endpoint

`apps/api/main.py` instancia os owners entre as linhas `103-132` e expone as rotas entre `135-489`.

| Linha(s) | Endpoint / grupo | Owner principal | Persistencia / artefato |
| --- | --- | --- | --- |
| `135-141` | `/health`, `/` | API local + `SqliteHandStore` | leitura de `db_path`, sem escrita |
| `143-171` | `/v1/hands/new`, `/v1/hands/{hand_id}` | `GameEngine` + `SqliteHandStore` | cria `hands` e snapshot inicial |
| `173-223` | `/v1/hands/{hand_id}/actions`, `/auto` | `GameEngine` + `BaselineAgent` + `SqliteHandStore` | escreve `action_log`, `snapshots`, `decision_traces` |
| `225-248` | `/traces`, `/replay`, `/taxonomy`, `/export/phh-like` | `SqliteHandStore`, `ReplayService`, `TaxonomyService`, `ExportService` | leitura de DB e export dict |
| `250-266` | `/v1/spots/packs*`, `/v1/spots/taxonomy` | `SpotPackService`, `TaxonomyService` | runtime em memoria; sem escrita canonica |
| `268-289` | `/v1/benchmark/*`, `/v1/experiments/run` | `BenchmarkService`, `ExperimentRunner` | relatorios em `var/reports/experiment_*.json` |
| `291-329` | `/v1/sessions/*` | `SessionRunner`, `SqliteHandStore`, `SessionAnalyticsService`, `ExportService` | `sessions`, `hands`, `snapshots`, `decision_traces`, relatorios `session_*.json` |
| `330-344` | solver-like normalized / compare spot pack | `SolverLabelService` | compara spot e gera rows locais |
| `346-384` | solver catalog, train/list/eval model, datasets | `ModelService`, `DatasetService`, `EvaluationService` | `var/models`, `var/datasets`, `var/reports` |
| `382-390` | external solver catalog/compare | `ExternalSolverService` | leitura de `var/external_solver/sample_solver_v1.json` |
| `393-435` | tournaments, curriculum, readiness, opponent profile, adaptive benchmark, coach e hand review | `TournamentService`, `CurriculumService`, `ReadinessService`, `OpponentProfileService`, `CoachService` | relatorios locais e leitura de DB |
| `437-485` | governance, release gate, batch generation, regression suite, calibration, deploy manifest, release notes, alpha candidate | `GovernanceService`, `ReleaseGateService`, `BatchGenerationService`, `RegressionSuiteService`, `CalibrationService`, `DeployService`, `ReleaseNotesService`, `AlphaCandidateService` | `var/model_cards`, `var/batches`, `var/regression`, `var/reports` e leitura de `docs/` |

## 1A. Appendice canonico de endpoints (metodo + path + owner)

| Metodo + path | Owner principal |
| --- | --- |
| `GET /health` | API local (`create_app`) |
| `GET /` | API local (`create_app`) |
| `GET /ui` | UI static host (`apps/web_ui/dist`) |
| `GET /ui/{path:path}` | UI static host + SPA fallback (`apps/web_ui/dist`) |
| `POST /v1/hands/new` | `GameEngine` + `SqliteHandStore` |
| `GET /v1/hands/{hand_id}` | `GameEngine` + `SqliteHandStore` |
| `POST /v1/hands/{hand_id}/actions` | `GameEngine` + `SqliteHandStore` |
| `POST /v1/hands/{hand_id}/auto` | `BaselineAgent` + `GameEngine` + `SqliteHandStore` |
| `GET /v1/hands/{hand_id}/traces` | `SqliteHandStore` |
| `GET /v1/hands/{hand_id}/replay` | `ReplayService` |
| `GET /v1/hands/{hand_id}/taxonomy` | `TaxonomyService` |
| `GET /v1/hands/{hand_id}/export/phh-like` | `ExportService` |
| `GET /v1/spots/packs` | `SpotPackService` |
| `GET /v1/spots/taxonomy` | `TaxonomyService` |
| `POST /v1/spots/packs/{spot_id}/instantiate` | `SpotPackService` |
| `POST /v1/benchmark/smoke` | `BenchmarkService` |
| `POST /v1/benchmark/h2h` | `BenchmarkService` |
| `POST /v1/experiments/run` | `ExperimentRunner` |
| `POST /v1/sessions/h2h` | `SessionRunner` |
| `GET /v1/sessions/{session_id}` | `SqliteHandStore` |
| `GET /v1/sessions/{session_id}/traces` | `SqliteHandStore` |
| `GET /v1/sessions/{session_id}/analytics` | `SessionAnalyticsService` |
| `GET /v1/sessions/{session_id}/export/phh-like` | `ExportService` |
| `GET /v1/hands/{hand_id}/spots/normalized` | `SolverLabelService` |
| `POST /v1/spots/packs/{spot_id}/compare-solver-like` | `SolverLabelService` |
| `GET /v1/solver/catalog` | API local (`create_app`) |
| `POST /v1/models/train/policy-table` | `ModelService` |
| `GET /v1/models` | `ModelService` |
| `POST /v1/models/{model_id}/evaluate` | `ModelService` |
| `POST /v1/datasets/build` | `DatasetService` |
| `GET /v1/datasets` | `DatasetService` |
| `POST /v1/models/{model_id}/evaluate-dataset/{dataset_id}` | `EvaluationService` |
| `GET /v1/external-solver/catalog` | `ExternalSolverService` |
| `POST /v1/external-solver/{solver_name}/compare/{spot_id}` | `ExternalSolverService` |
| `POST /v1/tournaments/round-robin` | `TournamentService` |
| `GET /v1/sessions/{session_id}/curriculum` | `CurriculumService` |
| `GET /v1/system/readiness` | `ReadinessService` |
| `GET /v1/sessions/{session_id}/opponent-profile` | `OpponentProfileService` |
| `POST /v1/benchmark/adaptive` | `OpponentProfileService` |
| `GET /v1/sessions/{session_id}/coach-report` | `CoachService` |
| `GET /v1/hands/{hand_id}/review` | `CoachService` |
| `POST /v1/governance/model-cards/{model_id}` | `GovernanceService` |
| `GET /v1/governance/model-cards` | `GovernanceService` |
| `POST /v1/release/gate/{model_id}/{dataset_id}` | `ReleaseGateService` |
| `POST /v1/hands/batch-generate` | `BatchGenerationService` |
| `POST /v1/regression/suites/build` | `RegressionSuiteService` |
| `POST /v1/models/{model_id}/calibrate/{dataset_id}` | `CalibrationService` |
| `GET /v1/deploy/manifest` | `DeployService` |
| `GET /v1/release/notes` | `ReleaseNotesService` |
| `GET /v1/system/alpha-candidate` | `AlphaCandidateService` |

## 2. Schema SQLite local

Fonte: `packages/persistence/sqlite_store.py` linhas `21-78`.

| Tabela | Papel | Produtor principal | Consumidores principais |
| --- | --- | --- | --- |
| `hands` | registro mestre da mao, stacks, seed, snapshot inicial | `create_hand()` | API hand lookup, replay, export, taxonomy |
| `action_log` | acoes aplicadas por actor e valor | `append_action()` | replay, analytics, opponent profile |
| `snapshots` | snapshots ordenados do estado da mao | `append_snapshot()` | hand lookup, replay, export, hand review |
| `decision_traces` | racional e features da decisao automatica | `append_decision_trace()` | datasets, analytics, coach, model eval |
| `sessions` | metadata e resumo de sessoes H2H | `SessionRunner.run_h2h()` | session lookup, analytics, curriculum, coach |

PRAGMAs ativos:

- WAL
- synchronous NORMAL
- foreign_keys ON

## 3. Fluxo canonico de dados de IA

### 3.1 Runtime decisorio

1. `BaselineAgent` ou `AdaptiveBaselineAgent` decide a acao.
2. A decisao pode incluir `DecisionTrace`.
3. `apps/api/main.py` ou `SessionRunner` persistem o trace em `decision_traces`.
4. `TaxonomyService` e `SessionAnalyticsService` derivam classificacoes e agregacoes.

### 3.2 Normalizacao solver-like

1. `SolverLabelService` normaliza runtime/snapshot usando `packages/solver_like/normalize.py`.
2. `bucketize_spot()` gera a chave estrutural do spot.
3. `SolverLikeLabeler` aplica rotulo heuristico de acao/confidence.
4. `build_training_rows_from_spot_packs()` gera rows artificiais para treino local.

### 3.3 Dataset builder

1. `DatasetService.build_master_dataset()` instancia `DatasetBuilder`.
2. `DatasetBuilder.build_rows()` coleta traces persistidos e, opcionalmente, spot packs.
3. `stable_split()` define `train/validation/test` por hash estavel.
4. `build_manifest()` resume contagens, fontes e taxonomy.
5. O dataset e materializado em `var/datasets/<dataset_id>/`.

### 3.4 Treino de modelo

1. `ModelService.train_policy_table()` obtém rows do `SolverLabelService`.
2. `PolicyTableTrainer` agrega buckets e acao majoritaria.
3. `PolicyRegistry` salva o modelo em `var/models/<model_id>.json`.
4. `PolicyTableAgent` pode consumir o modelo salvo para inferencia local.

### 3.5 Avaliacao, calibracao e governance

1. `EvaluationService.evaluate_model()` compara dataset vs policy table.
2. O relatorio e salvo em `var/reports/evaluation_<model>_<dataset>_<split>.json`.
3. `CalibrationService.calibrate()` calcula `ece_proxy` a partir da avaliacao.
4. `GovernanceService.build_model_card()` gera `var/model_cards/<model_id>.card.json`.
5. `ReleaseGateService.evaluate_candidate()` combina readiness e metrica minima.
6. `AlphaCandidateService.snapshot()` agrega release gate, model card e deploy manifest.

## 4. Artefatos `var/*`

### 4.1 Versionados

| Caminho | Papel |
| --- | --- |
| `var/golden/golden_hands.json` | casos de regressao / golden hands |
| `var/external_solver/sample_solver_v1.json` | catalogo mock de solver externo |
| `var/datasets/.gitkeep` | placeholder de datasets |
| `var/models/.gitkeep` | placeholder de modelos |
| `var/model_cards/.gitkeep` | placeholder de model cards |
| `var/regression/.gitkeep` | placeholder de suites de regressao |

### 4.2 Observados no workspace local

| Caminho / padrao | Papel observado |
| --- | --- |
| `var/poker_ai_local.db` | banco SQLite ativo do laboratorio |
| `var/datasets/<dataset_id>/{rows,train,validation,test}.jsonl` | datasets materializados |
| `var/datasets/<dataset_id>/manifest.json` | manifesto resumido do dataset |
| `var/models/<model_id>.json` | policy tables treinadas |
| `var/model_cards/<model_id>.card.json` | governanca local do modelo |
| `var/reports/evaluation_*.json` | avaliacao de modelo vs dataset |
| `var/reports/session_*.json` | relatorios de sessao |
| `var/reports/experiment_*.json` | relatorios de benchmark/experimento |
| `var/regression/suite_api.json` | suite de regressao observada no workspace |
| `var/batches/*.json` | lotes gerados localmente |

## 5. Contratos JSON observados

### 5.1 Sample external solver

Fonte: `var/external_solver/sample_solver_v1.json`.

Chaves relevantes:

- `name`
- `labels[]`
  - `spot_id`
  - `action`
  - `confidence`

Interpretacao: adaptador arquivo/mock para comparacao offline de spot packs; nao ha chamada a solver real em tempo de runtime.

### 5.2 Modelo policy table

Fonte observada: `var/models/74162623-a510-40a0-930a-449e14052116.json`.

Chaves relevantes:

- `model_id`
- `metadata`
  - `model_name`
  - `row_count`
  - `bucket_count`
  - `trainer`
- `table`
  - bucket key string -> `{action, support, confidence, mean_equity, mean_pot_odds}`

### 5.3 Evaluation report

Fonte observada: `var/reports/evaluation_74162623-a510-40a0-930a-449e14052116_f198cba9-c371-46f5-a8d6-0bd5afc4461f_all.json`.

Chaves relevantes:

- `model_id`
- `dataset_id`
- `split`
- `rows`
- `accuracy`
- `split_accuracy`
- `taxonomy_accuracy`
- `confusion`
- `confidence_bins`

### 5.4 Dataset manifest

Fonte observada: `var/datasets/f198cba9-c371-46f5-a8d6-0bd5afc4461f/manifest.json`.

Chaves relevantes:

- `dataset_id`
- `dataset_name`
- `row_count`
- `split_counts`
- `source_counts`
- `taxonomy_counts`

### 5.5 Model card

Fonte observada: `var/model_cards/74162623-a510-40a0-930a-449e14052116.card.json`.

Chaves relevantes:

- `model_id`
- `metadata`
- `training_artifact_path`
- `latest_evaluation`
- `intended_use`
- `limitations`
- `safety_scope`

### 5.6 Regression suite manifest

Fonte observada: `var/regression/suite_api.json`.

Chaves relevantes:

- `suite_name`
- `golden_hand_count`
- `golden_hand_ids`
- `spot_pack_count`
- `spot_pack_ids`
- `recommended_gate`

## 6. Integracoes externas reais vs aparentes

### Integracoes reais implementadas

- FastAPI / Pydantic
- SQLite local
- Uvicorn local
- Docker / docker compose local
- pytest

### Integracoes locais simuladas ou proxies

- solver externo via arquivo JSON local
- release readiness via checks de artefatos locais
- deploy manifest via presenca de arquivos, nao deploy real
- release notes via leitura de `docs/`

### Integracoes ausentes no codigo versionado

- observabilidade cloud
- secrets manager
- deploy produtivo
- solver online
- registry/model serving externo

## 7. Ligacao entre dados e testes

| Area | Testes relevantes |
| --- | --- |
| engine / deck / showdown / sidepots | `tests/unit/test_engine_*`, `test_deck.py`, `test_showdown.py`, `test_sidepots.py` |
| baseline / adaptive / harness | `test_baseline_agent*.py`, `test_adaptive_benchmark.py`, `test_harness.py` |
| persistencia / replay / traces | `test_persistence_replay.py`, `test_store_session_traces.py`, `test_session_runner.py` |
| spot packs / solver-like / models | `test_spot_packs.py`, `test_solver_like_and_models.py`, `test_dataset_eval_external.py` |
| taxonomy / export / analytics / coach | `test_taxonomy_and_export.py`, `test_opponent_profile_and_coach.py` |
| tournaments / curriculum / readiness | `test_tournament_curriculum_readiness.py` |
| API end-to-end | `tests/integration/test_api_*.py`, `test_health.py` |
