# 101 - Exploratory Data Analysis Full I/O and Features

## Resumo

- Gerado em: `2026-03-15T03:11:38.158072+00:00`
- Repo root: `C:/Users/davis/Workspace/mr_poker`
- Rotas API mapeadas: `48/48`
- Request models mapeados: `10/10`
- Blocos de feature mapeados: `3/3`

## API Contracts

| Metodo | Path | Owner primario | Fonte |
| --- | --- | --- | --- |
| `GET` | `/` | `API Router` | `apps/api/routers/system.py:21` |
| `GET` | `/health` | `API Router` | `apps/api/routers/system.py:11` |
| `GET` | `/v1/datasets` | `DatasetService` | `apps/api/routers/ml.py:38` |
| `GET` | `/v1/deploy/manifest` | `DeployService` | `apps/api/routers/system.py:47` |
| `GET` | `/v1/external-solver/catalog` | `ExternalSolverService` | `apps/api/routers/ml.py:89` |
| `GET` | `/v1/governance/model-cards` | `GovernanceService` | `apps/api/routers/governance.py:21` |
| `GET` | `/v1/hands/{hand_id}` | `dict[str, HandRuntime]` | `apps/api/routers/hands.py:34` |
| `GET` | `/v1/hands/{hand_id}/export/phh-like` | `ExportService` | `apps/api/routers/hands.py:126` |
| `GET` | `/v1/hands/{hand_id}/replay` | `ReplayService` | `apps/api/routers/hands.py:108` |
| `GET` | `/v1/hands/{hand_id}/review` | `CoachService` | `apps/api/routers/hands.py:146` |
| `GET` | `/v1/hands/{hand_id}/spots/normalized` | `SolverLabelService` | `apps/api/routers/hands.py:135` |
| `GET` | `/v1/hands/{hand_id}/taxonomy` | `TaxonomyService` | `apps/api/routers/hands.py:117` |
| `GET` | `/v1/hands/{hand_id}/traces` | `SqliteHandStore` | `apps/api/routers/hands.py:102` |
| `GET` | `/v1/models` | `ModelService` | `apps/api/routers/ml.py:17` |
| `GET` | `/v1/release/notes` | `ReleaseNotesService` | `apps/api/routers/system.py:53` |
| `GET` | `/v1/sessions/{session_id}` | `SqliteHandStore` | `apps/api/routers/sessions.py:26` |
| `GET` | `/v1/sessions/{session_id}/analytics` | `SessionAnalyticsService` | `apps/api/routers/sessions.py:46` |
| `GET` | `/v1/sessions/{session_id}/coach-report` | `CoachService` | `apps/api/routers/sessions.py:82` |
| `GET` | `/v1/sessions/{session_id}/curriculum` | `CurriculumService` | `apps/api/routers/sessions.py:64` |
| `GET` | `/v1/sessions/{session_id}/export/phh-like` | `ExportService` | `apps/api/routers/sessions.py:55` |
| `GET` | `/v1/sessions/{session_id}/opponent-profile` | `OpponentProfileService` | `apps/api/routers/sessions.py:73` |
| `GET` | `/v1/sessions/{session_id}/traces` | `SqliteHandStore` | `apps/api/routers/sessions.py:35` |
| `GET` | `/v1/solver/catalog` | `API Router` | `apps/api/routers/ml.py:76` |
| `GET` | `/v1/spots/packs` | `SpotPackService` | `apps/api/routers/spots.py:11` |
| `GET` | `/v1/spots/taxonomy` | `TaxonomyService` | `apps/api/routers/spots.py:17` |
| `GET` | `/v1/system/alpha-candidate` | `AlphaCandidateService` | `apps/api/routers/system.py:36` |
| `GET` | `/v1/system/readiness` | `ReadinessService` | `apps/api/routers/system.py:30` |
| `POST` | `/v1/benchmark/adaptive` | `OpponentProfileService` | `apps/api/routers/benchmark.py:52` |
| `POST` | `/v1/benchmark/h2h` | `BenchmarkService` | `apps/api/routers/benchmark.py:28` |
| `POST` | `/v1/benchmark/smoke` | `BenchmarkService` | `apps/api/routers/benchmark.py:18` |
| `POST` | `/v1/datasets/build` | `DatasetService` | `apps/api/routers/ml.py:32` |
| `POST` | `/v1/experiments/run` | `ExperimentRunner` | `apps/api/routers/benchmark.py:39` |
| `POST` | `/v1/external-solver/{solver_name}/compare/{spot_id}` | `ExternalSolverService` | `apps/api/routers/ml.py:95` |
| `POST` | `/v1/governance/model-cards/{model_id}` | `GovernanceService` | `apps/api/routers/governance.py:12` |
| `POST` | `/v1/hands/batch-generate` | `BatchGenerationService` | `apps/api/routers/hands.py:155` |
| `POST` | `/v1/hands/new` | `GameEngine` | `apps/api/routers/hands.py:12` |
| `POST` | `/v1/hands/{hand_id}/actions` | `dict[str, HandRuntime]` | `apps/api/routers/hands.py:46` |
| `POST` | `/v1/hands/{hand_id}/auto` | `dict[str, HandRuntime]` | `apps/api/routers/hands.py:68` |
| `POST` | `/v1/models/train/policy-table` | `ModelService` | `apps/api/routers/ml.py:11` |
| `POST` | `/v1/models/{model_id}/calibrate/{dataset_id}` | `CalibrationService` | `apps/api/routers/ml.py:60` |
| `POST` | `/v1/models/{model_id}/evaluate` | `ModelService` | `apps/api/routers/ml.py:23` |
| `POST` | `/v1/models/{model_id}/evaluate-dataset/{dataset_id}` | `EvaluationService` | `apps/api/routers/ml.py:44` |
| `POST` | `/v1/regression/suites/build` | `RegressionSuiteService` | `apps/api/routers/governance.py:48` |
| `POST` | `/v1/release/gate/{model_id}/{dataset_id}` | `ReleaseGateService` | `apps/api/routers/governance.py:27` |
| `POST` | `/v1/sessions/h2h` | `SessionRunner` | `apps/api/routers/sessions.py:13` |
| `POST` | `/v1/spots/packs/{spot_id}/compare-solver-like` | `SolverLabelService` | `apps/api/routers/spots.py:35` |
| `POST` | `/v1/spots/packs/{spot_id}/instantiate` | `SpotPackService` | `apps/api/routers/spots.py:23` |
| `POST` | `/v1/tournaments/round-robin` | `TournamentService` | `apps/api/routers/benchmark.py:62` |

## Request Models (BaseModel)

### `ActionRequest` (apps/api/contracts.py:17)

| Campo | Tipo | Default |
| --- | --- | --- |
| `action_type` | `ActionType` | `-` |
| `amount` | `int` | `0` |

### `AdaptiveBenchmarkRequest` (apps/api/contracts.py:42)

| Campo | Tipo | Default |
| --- | --- | --- |
| `num_hands` | `int` | `40` |
| `stacks` | `tuple[int, int]` | `(100, 100)` |
| `seed_base` | `int` | `50000` |

### `BatchGenerateRequest` (apps/api/contracts.py:55)

| Campo | Tipo | Default |
| --- | --- | --- |
| `batch_name` | `str` | `'baseline_batch'` |
| `num_hands` | `int` | `50` |
| `seed_base` | `int` | `90000` |
| `stacks` | `tuple[int, int]` | `(100, 100)` |

### `H2HBenchmarkRequest` (apps/api/contracts.py:28)

| Campo | Tipo | Default |
| --- | --- | --- |
| `num_matches` | `int` | `3` |
| `hands_per_match` | `int` | `50` |
| `stacks` | `tuple[int, int]` | `(100, 100)` |
| `seed_base` | `int` | `20000` |

### `NewHandRequest` (apps/api/contracts.py:10)

| Campo | Tipo | Default |
| --- | --- | --- |
| `stacks` | `tuple[int, int]` | `(100, 100)` |
| `button_seat` | `int` | `0` |
| `seed` | `Optional[int]` | `None` |
| `deck_prefix` | `list[str]` | `Field(default_factory=list)` |

### `RegressionSuiteRequest` (apps/api/contracts.py:67)

| Campo | Tipo | Default |
| --- | --- | --- |
| `suite_name` | `str` | `'core_regression_v1'` |

### `ReleaseGateRequest` (apps/api/contracts.py:62)

| Campo | Tipo | Default |
| --- | --- | --- |
| `min_accuracy` | `float` | `0.55` |
| `min_rows` | `int` | `10` |

### `SessionRunRequest` (apps/api/contracts.py:35)

| Campo | Tipo | Default |
| --- | --- | --- |
| `num_hands` | `int` | `100` |
| `stacks` | `tuple[int, int]` | `(100, 100)` |
| `seed_base` | `int` | `30000` |
| `session_name` | `str` | `'local_h2h_session'` |

### `SmokeBenchmarkRequest` (apps/api/contracts.py:22)

| Campo | Tipo | Default |
| --- | --- | --- |
| `num_hands` | `int` | `50` |
| `stacks` | `tuple[int, int]` | `(100, 100)` |
| `seed_base` | `int` | `10000` |

### `TournamentRequest` (apps/api/contracts.py:48)

| Campo | Tipo | Default |
| --- | --- | --- |
| `entrants` | `list[str]` | `Field(default_factory=lambda: ['baseline', 'adaptive'])` |
| `hands_per_match` | `int` | `20` |
| `seed_base` | `int` | `70000` |
| `stacks` | `tuple[int, int]` | `(100, 100)` |

## Feature Catalog

### Decision Features - Minimum

- `street`: `str`
- `pot`: `int`
- `to_call`: `int`
- `player_stack`: `int`
- `board_size`: `int`
- `acting_seat`: `int | None`
- `is_button`: `bool`
- `current_bet`: `int`

### Decision Features - Baseline

- `street`: `str`
- `to_call`: `int`
- `pot`: `int`
- `stack`: `int`
- `hole_class`: `str`
- `is_pair`: `bool`
- `is_suited`: `bool`
- `is_connected`: `bool`
- `high_rank`: `int`
- `low_rank`: `int`
- `board_size`: `int`
- `pot_odds`: `float`
- `board_texture`: `str`
- `spr`: `float`

### Solver-like

- Bucket functions: `_spr_bucket, _equity_bucket, _price_bucket, bucketize_spot`
- Bucket tags: `equity_bucket, hole_bucket, position_bucket, price_bucket, spr_bucket, street_bucket, texture_bucket`
- normalize_runtime fields: `actor_seat, board, board_size, board_texture, button_seat, hand_id, high_rank, hole_class, is_connected, is_pair, is_suited, legal_actions, low_rank, pot, pot_odds, spr, stack, street, to_call`
- normalize_snapshot fields: `action_taken, actor_seat, board, board_size, board_texture, button_seat, estimated_equity, hand_id, hole_class, legal_actions, pot, pot_odds, rationale, spr, street, to_call`
- label output fields: `bucket_info, confidence, label_action, label_amount_rule, label_reason`

### Service Metrics

- `CalibrationService.calibrate` (`services/calibration_service/service.py:10`): `confidence_bins, dataset_id, ece_proxy, model_id, note, rows, split`
- `CurriculumService.build_for_session` (`services/curriculum_service/service.py:13`): `curriculum, session_id`
- `EvaluationService.evaluate_model` (`services/evaluation_service/service.py:27`): `accuracy, confidence_bins, confusion, dataset_id, model_id, rows, split, split_accuracy, taxonomy_accuracy`
- `GovernanceService.build_model_card` (`services/governance_service/service.py:24`): `intended_use, latest_evaluation, limitations, metadata, model_id, safety_scope, training_artifact_path`
- `ReadinessService.snapshot` (`services/readiness_service/service.py:14`): `asset_readiness, blockers, dataset_count, deprecations, docker_assets_ready, docs_ready, model_count, operational_blockers, operational_readiness, production_candidate, tests_ready`

## Data Artifacts (var/*)

- include_var_runtime: `True`
- EDA module path: `C:/Users/davis/.agents/skills/K-Dense-AI__claude-scientific-skills/scientific-skills/exploratory-data-analysis/scripts/eda_analyzer.py`
- Arquivos analisados: `233`

### Distribuicao por extensao

| Extensao | Count |
| --- | ---: |
| `.db` | `1` |
| `.json` | `152` |
| `.jsonl` | `80` |

### Distribuicao por categoria EDA

| Categoria | Count |
| --- | ---: |
| `general_scientific` | `152` |
| `unknown` | `81` |

## Gaps

- Nenhum gap P0 detectado para API/request models/feature blocks obrigatorios.
