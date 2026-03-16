# File By File Inventory

## Metodo

Inventario canonico baseado em `git ls-files` (`517` arquivos versionados) com apoio de leitura ancorada em linhas, headings Markdown e simbolos Python.

Leitura aplicada:

- arquivos de codigo: papel, classes/funcoes publicas, dependencias dominantes, efeitos colaterais e cobertura
- arquivos textuais nao-codigo: papel documental/configuracional e ancora principal
- artefatos binarios: tipo, funcao, consumidor e relevancia

Legenda:

- `autoritativo`: especificacao, ADR, runbook ou contrato atual
- `historico`: sprint, snapshot, roadmap, preview, backup
- `auditoria`: auditoria, checklist, validacao, gap register
- `executivo`: camada de negocio/promocao em `doc/`

## 1. Root, CI, configs e infra

| Arquivo | Tipo / ancora | Inventario tecnico |
| --- | --- | --- |
| `.dockerignore` | texto / L16 | exclusao de artefatos de build e runtime para imagens; sem cobertura automatica; afeta empacotamento Docker |
| `.env.example` | texto / L2 | exemplo minimo de ambiente; depende do bootstrap local; sem cobertura |
| `.github/workflows/ci.yml` | texto / L17 | workflow de CI principal; integra GitHub Actions com lint/tests; sem teste direto |
| `.github/workflows/python-tests.yml` | texto / L12 | workflow adicional de testes Python; redundancia operacional de CI |
| `.gitignore` | texto / L24 | define artefatos efemeros fora do Git, especialmente `var/*` gerado e caches |
| `AGENTS.md` | autoritativo operacional / `# mr_poker Agent Guide`@1 | ordem de verdade do repositorio, mapa por escopo e comandos canonicos |
| `CONTRIBUTING.md` | texto / `# Contributing`@1 | guia contributivo; papel organizacional; sem autoridade sobre runtime |
| `Dockerfile` | texto / L11 | imagem base da API local; integra packaging; sem teste direto, coberto indiretamente por docs/runbooks |
| `FINAL_AUDIT_STATUS.md` | auditoria / `# FINAL_AUDIT_STATUS`@1 | snapshot final de auditoria historica; usar apenas como contexto, nao como contrato vigente |
| `FINAL_STATUS.md` | historico / `# Final Status — mr_pocker`@1 | resumo final historico do projeto; sujeito a drift frente ao codigo atual |
| `LICENSE` | texto / L1 | licenca do repositorio |
| `Makefile` | texto / L35 | automacao local para lint/tests/api/docker; caveat: assume `bash`; integra `ruff`, `pytest`, `uvicorn`, `docker compose` |
| `README.md` | autoritativo operacional / `# mr_pocker`@1 | onboarding atual; anchors de quick start em `27`, `40`, `51`, `58`, `66`; integra API local, Docker e docs |
| `SYNC_STATUS.txt` | historico / L1 | nota curta de sync; valor operacional baixo |
| `docker-compose.yml` | texto / L17 | orquestracao local containerizada; integra `Dockerfile`, volumes/configs e porta da API |
| `pyproject.toml` | texto / L45 | metadados Python, dependencias (`fastapi`, `uvicorn`, `pytest`, `ruff`) e config do `ruff`; sem teste direto |
| `configs/app.local.json` | config / L6 | defaults locais de app; consumido por runbooks/packaging |
| `configs/app.prod.template.json` | config / L6 | template de producao; nao prova prontidao produtiva |
| `infra/scripts/bootstrap.sh` | script / L10 | bootstrap local Unix/bash; side effect em ambiente; sem cobertura direta |
| `infra/scripts/clean_repo.sh` | script / L14 | limpeza operacional do workspace; side effect destrutivo local controlado |
| `infra/scripts/healthcheck.py` | codigo / imports `urllib.request` | healthcheck HTTP simples da API local; sem teste direto do script, coberto indiretamente por `tests/integration/test_health.py` |
| `infra/scripts/repo_sync_manual.sh` | script / L11 | sync manual do repo; operacional historico |
| `infra/scripts/run_full_validation.sh` | script / L14 | pipeline local agregada de validacao; assume shell Unix |
| `infra/scripts/run_integration_tests.sh` | script / L21 | execucao dedicada de integration tests |
| `infra/scripts/run_unit_tests.sh` | script / L34 | execucao dedicada de unit tests; tambem assume bash |
| `apps/api/AGENTS.md` | autoritativo operacional / `# apps/api Scope Guide`@1 | regras de ownership da camada HTTP e validacao minima |
| `apps/api/main.py` | codigo / `create_app`@102 | composition root HTTP; depende de FastAPI, Pydantic, packages e services; side effects: instancia store e escreve DB/`var/*` via rotas; cobertura forte em `tests/integration/test_api_*.py` e `test_health.py` |
| `infra/AGENTS.md` | autoritativo operacional / `# infra Scope Guide`@1 | limita e explicita uso de scripts operacionais locais em shell Unix/Windows |

| `.claude/launch.json` | config / JSON | configuracao de launch do ambiente Claude |
| `=0.28.0` | artefato / versionamento | marcador de versao de dependencia |
| `infra/notebooks/mr_poker_kaggle_full_train.ipynb` | notebook / treino | notebook de treinamento para execucao em plataforma cloud |
| `infra/notebooks/mr_poker_sft_colab.ipynb` | notebook / treino | notebook de treinamento para execucao em plataforma cloud |
| `infra/scripts/colab_one_click.py` | script / treino-dados | script de treino ou upload de dados PokerBench |
| `infra/scripts/pokerbench_sft_train_hfjob.py` | script / treino-dados | script de treino ou upload de dados PokerBench |
| `infra/scripts/pokerbench_upload_hfjob.py` | script / treino-dados | script de treino ou upload de dados PokerBench |
| `infra/scripts/train_pokerbench_colab.py` | script / treino-dados | script de treino ou upload de dados PokerBench |

## 2. Pacote executivo em `doc/`

| Arquivo | Tipo / ancora | Inventario tecnico |
| --- | --- | --- |
| `doc/PACOTE_EXECUTIVO_ALPHA_BETA_PRODUCAO/README.md` | executivo / `# Pacote Executivo...`@1 | indice do pacote executivo; consumidor: lideranca e promocao de ambiente |
| `doc/PACOTE_EXECUTIVO_ALPHA_BETA_PRODUCAO/Plano_Executivo_Alpha_Beta_Producao.docx` | binario DOCX | versao editavel executiva do plano; consumidor humano; nao e fonte primaria tecnica |
| `doc/PACOTE_EXECUTIVO_ALPHA_BETA_PRODUCAO/Plano_Executivo_Alpha_Beta_Producao.pdf` | binario PDF | versao distribuivel do plano executivo |
| `doc/PACOTE_EXECUTIVO_ALPHA_BETA_PRODUCAO/csv/cronograma.csv` | texto / L19 | cronograma estruturado; consumidor: PMO / execucao |
| `doc/PACOTE_EXECUTIVO_ALPHA_BETA_PRODUCAO/csv/matriz_gates.csv` | texto / L11 | gates Alpha/Beta/Producao em formato tabular |
| `doc/PACOTE_EXECUTIVO_ALPHA_BETA_PRODUCAO/csv/raci.csv` | texto / L14 | ownership RACI tabular |
| `doc/PACOTE_EXECUTIVO_ALPHA_BETA_PRODUCAO/diagrams/01_fluxo_promocao.mmd` | texto / L14 | diagrama Mermaid do fluxo de promocao |
| `doc/PACOTE_EXECUTIVO_ALPHA_BETA_PRODUCAO/diagrams/02_workstreams.mmd` | texto / L16 | diagrama Mermaid dos workstreams |
| `doc/PACOTE_EXECUTIVO_ALPHA_BETA_PRODUCAO/diagrams/03_governanca_gates.mmd` | texto / L16 | diagrama Mermaid de governanca/gates |
| `doc/PACOTE_EXECUTIVO_ALPHA_BETA_PRODUCAO/diagrams/04_finops_ciclo.mmd` | texto / L4 | diagrama Mermaid do ciclo FinOps |
| `doc/PACOTE_EXECUTIVO_ALPHA_BETA_PRODUCAO/docs/01_resumo_executivo.md` | executivo / `# Resumo executivo`@1 | narrativa de alto nivel para promocao |
| `doc/PACOTE_EXECUTIVO_ALPHA_BETA_PRODUCAO/docs/02_matriz_gates_alpha_beta_producao.md` | executivo / `# Matriz de gates...`@1 | gates por fase |
| `doc/PACOTE_EXECUTIVO_ALPHA_BETA_PRODUCAO/docs/03_checklists_operacionais_por_fase.md` | executivo / `# Checklists operacionais...`@1 | checklists por fase |
| `doc/PACOTE_EXECUTIVO_ALPHA_BETA_PRODUCAO/docs/04_raci_responsaveis.md` | executivo / `# RACI dos responsaveis`@1 | ownership executivo |
| `doc/PACOTE_EXECUTIVO_ALPHA_BETA_PRODUCAO/docs/05_cronograma_detalhado.md` | executivo / `# Cronograma detalhado...`@1 | cronograma expandido |
| `doc/PACOTE_EXECUTIVO_ALPHA_BETA_PRODUCAO/docs/06_finops_local_alpha_beta_prod.md` | executivo / `# Plano FinOps...`@1 | custos por fase |
| `doc/PACOTE_EXECUTIVO_ALPHA_BETA_PRODUCAO/docs/07_arquitetura_de_promocao.md` | executivo / `# Arquitetura de promocao...`@1 | arquitetura de promocao entre estagios |
| `doc/PACOTE_EXECUTIVO_ALPHA_BETA_PRODUCAO/docs/08_riscos_e_controles.md` | executivo / `# Riscos e controles`@1 | matriz de risco executiva |
| `doc/PACOTE_EXECUTIVO_ALPHA_BETA_PRODUCAO/docs/09_backlog_macro.md` | executivo / `# Backlog macro...`@1 | backlog macro por trilha |
| `doc/PACOTE_EXECUTIVO_ALPHA_BETA_PRODUCAO/docs/10_recomendacao_final.md` | executivo / `# Recomendacao final`@1 | recomendacao final de promocao |
| `doc/PACOTE_EXECUTIVO_ALPHA_BETA_PRODUCAO/docs/11_premissas_e_decisoes_fechadas.md` | executivo / `# Premissas e decisoes fechadas`@1 | premissas congeladas |
| `doc/PACOTE_EXECUTIVO_ALPHA_BETA_PRODUCAO/rendered/Plano_Executivo_Alpha_Beta_Producao.pdf` | binario PDF | render final do pacote executivo |
| `doc/PACOTE_EXECUTIVO_ALPHA_BETA_PRODUCAO/rendered/page-1.png` | binario PNG | preview visual da pagina 1 |
| `doc/PACOTE_EXECUTIVO_ALPHA_BETA_PRODUCAO/rendered/page-2.png` | binario PNG | preview visual da pagina 2 |
| `doc/PACOTE_EXECUTIVO_ALPHA_BETA_PRODUCAO/rendered/page-3.png` | binario PNG | preview visual da pagina 3 |
| `doc/PACOTE_EXECUTIVO_ALPHA_BETA_PRODUCAO/rendered/page-4.png` | binario PNG | preview visual da pagina 4 |

## 3. `packages/*`

| Arquivo | Tipo / ancora | Inventario tecnico |
| --- | --- | --- |
| `packages/AGENTS.md` | autoritativo operacional / `# packages Scope Guide`@1 | regras de ownership da camada de dominio e runtime packages |
| `packages/adaptive_agent/__init__.py` | codigo / L1 | re-export do pacote adaptive; sem logica; cobertura indireta por imports |
| `packages/adaptive_agent/agent.py` | codigo / `AdaptiveBaselineAgent`@10 | agente adaptativo sobre baseline; deps: `collections`, baseline contracts, `ActionType`; side effect: nenhum IO; cobertura direta em `tests/unit/test_adaptive_benchmark.py`, indireta em tournament/API |
| `packages/baseline_agent/__init__.py` | codigo / L2 | re-export de `agent` e `contracts` |
| `packages/baseline_agent/agent.py` | codigo / `BaselineAgent`@13 | heuristica principal de decisao; deps: contracts, engine, equity, evaluator, features; side effect: nenhum IO; cobertura direta em `test_baseline_agent.py`, `test_baseline_agent_sprint04.py`, integracao auto-play |
| `packages/baseline_agent/contracts.py` | codigo / `AgentDecision`@11 | contrato dataclass de decisao + trace; deps: `ActionType`, `DecisionTrace`; cobertura indireta forte via baseline/session/model tests |
| `packages/common/__init__.py` | codigo / L1 | namespace comum minimo |
| `packages/common/types.py` | codigo / `Street`@6; `ActionType`@15 | enums de street e acoes; dependencia transversal do dominio; cobertura indireta ampla em engine/baseline/persistencia/tests |
| `packages/curriculum/__init__.py` | codigo / L3 | re-export de planner |
| `packages/curriculum/planner.py` | codigo / `build_curriculum`@7 | gera curriculum de estudo a partir de analytics + coach; deps: `collections`, `typing`; sem IO; cobertura indireta em `test_tournament_curriculum_readiness.py` e API sprint20 |
| `packages/dataset_builder/__init__.py` | codigo / L3 | re-export de builder |
| `packages/dataset_builder/builder.py` | codigo / `stable_split`@14; `DatasetBuilder`@23 | constroi rows, splits e manifestos; deps: `hashlib`, `json`, `pathlib`; side effect: grava `var/datasets`; cobertura direta em `test_dataset_eval_external.py`, indireta via API sprint20/30 |
| `packages/dataset_builder/pokerbench_adapter.py` | codigo / `PokerBenchAdapter`@273 | converte rows do HF PokerBench (574K decisoes solver) para formato JSONL do DatasetBuilder; deps: `bucketize_spot`, `classify_trace`, `stable_split`; cobertura direta em `test_pokerbench_adapter.py` |
| `packages/engine/__init__.py` | codigo / L2 | re-export de engine/models |
| `packages/engine/deck.py` | codigo / `Deck`@9 | baralho e draw deterministico; deps: `random`, `Card`; sem IO; cobertura direta em `test_deck.py` |
| `packages/engine/engine.py` | codigo / `HandRuntime`@13; `GameEngine`@19 | motor de mao completo; deps: deck, models, `ActionType`; side effect: mutacao em memoria do runtime; cobertura direta em `test_engine_setup.py`, `test_engine_rounds.py`, `test_showdown.py`, `test_sidepots.py`, `test_golden_hands.py` e integracao API |
| `packages/engine/models.py` | codigo / `Card`@13; `PlayerState`@34; `ActionEvent`@45; `HandState`@54 | modelos dataclass de estado; deps: `typing`, `Street`; sem IO; cobertura indireta ampla via engine/evaluator |
| `packages/equity/__init__.py` | codigo / L1 | re-export do estimador de equity |
| `packages/equity/monte_carlo.py` | codigo / `_full_deck`@14; `estimate_equity`@18 | equity proxy por Monte Carlo; deps: `random`, `itertools`, evaluator; sem IO; cobertura direta em `test_equity.py` e indireta via baseline |
| `packages/evaluation/__init__.py` | codigo / L3 | re-export de taxonomy_eval |
| `packages/evaluation/taxonomy_eval.py` | codigo / `DatasetEvaluation`@9; `_bin_for`@18; `evaluate_policy_model`@30 | avaliacao offline por accuracy/split/taxonomy/confidence; deps: `collections`, `dataclasses`; sem IO direto; cobertura direta em `test_dataset_eval_external.py`, indireta via services/API |
| `packages/evaluator/__init__.py` | codigo / L1 | re-export de hand evaluator |
| `packages/evaluator/hands.py` | codigo / `_straight_high`@23; `evaluate_five`@34; `best_hand_rank`@70; `hand_label`@76 | ranking e label de maos; deps: `collections`, `itertools`; sem IO; cobertura direta em `test_showdown.py`, `test_golden_hands.py` |
| `packages/exports/__init__.py` | codigo / L3 | re-export de export PHH-like |
| `packages/exports/phh_like.py` | codigo / `_card_list`@6; `export_hand_phh_like`@10; `export_session_manifest`@57 | materializa formato PHH-like local; deps: `typing`; sem IO; cobertura direta em `test_taxonomy_and_export.py`, integracao API export |
| `packages/external_solver/__init__.py` | codigo / L3 | re-export do adapter |
| `packages/external_solver/adapter.py` | codigo / `ExternalSolverAdapter`@8 | adaptador arquivo/mock para labels externos; deps: `json`, `pathlib`; side effect: leitura de `var/external_solver`; cobertura direta em `test_dataset_eval_external.py`, indireta via API |
| `packages/features/__init__.py` | codigo / L1 | namespace de features |
| `packages/features/minimum.py` | codigo / `MinimumDecisionFeatures`@9; `derive_minimum_features`@20 | feature set minimo do baseline inicial; deps: engine models; sem IO; cobertura direta em `test_features.py` |
| `packages/features/poker.py` | codigo / `BaselineFeatures`@12; `canonical_hole`@29; `_board_texture`@39; `derive_baseline_features`@53 | features de poker e textura de board; deps: engine models; sem IO; cobertura direta em `test_features.py`, indireta via solver-like/baseline |
| `packages/harness/__init__.py` | codigo / L1 | re-export do match harness |
| `packages/harness/match.py` | codigo / `MatchSummary`@11; `MatchHarness`@25 | H2H local entre agentes; deps: baseline agent, engine; side effect: nenhum IO; cobertura direta em `test_harness.py`, indireta em benchmarks/tournaments |
| `packages/logging_schema/__init__.py` | codigo / L1 | namespace de schemas de log |
| `packages/logging_schema/decision_trace.py` | codigo / `DecisionTrace`@11 | schema Pydantic do trace de decisao; deps: `pydantic`, `ActionType`; sem IO; cobertura indireta em persistence/session/model/governance |
| `packages/opponent_model/__init__.py` | codigo / L1 | re-export de profile |
| `packages/opponent_model/profile.py` | codigo / `build_opponent_profile`@11; `summarize_profile`@34 | perfil estatistico simples do oponente; deps: `collections`; sem IO; cobertura direta em `test_opponent_profile_and_coach.py`, indireta via API |
| `packages/persistence/__init__.py` | codigo / L1 | re-export de SQLite store |
| `packages/persistence/sqlite_store.py` | codigo / `SqliteHandStore`@9 | persistencia canonicamente versionada; deps: `sqlite3`, `json`, `pathlib`; side effects: cria schema e escreve DB; cobertura direta em `test_persistence_replay.py`, `test_store_session_traces.py`, `test_session_runner.py`, integracao API |
| `packages/llm_agent/__init__.py` | codigo / L3 | re-export de LLMPokerAgent |
| `packages/llm_agent/agent.py` | codigo / `LLMPokerAgent`@49 | agente poker baseado em LLM fine-tuned; deps: BaselineAgent, LLMInference, prompt builder; cobertura direta em `test_llm_agent.py` |
| `packages/llm_agent/inference.py` | codigo / `LLMInference`@30 | wrapper lazy-loading para inferencia LLM via transformers; deps: `transformers` (opcional), `threading`; cobertura direta em `test_llm_agent.py` |
| `packages/llm_agent/prompt.py` | codigo / `build_pokerbench_prompt`@30 | constroi prompts no formato PokerBench a partir de game state; sem deps externas; cobertura direta em `test_llm_agent.py` |
| `packages/policy_agent/__init__.py` | codigo / L3 | re-export do policy agent |
| `packages/policy_agent/agent.py` | codigo / `PolicyTableAgent`@9 | inferencia via tabela treinada; deps: baseline fallback, contracts, buckets solver-like; sem IO; cobertura indireta via `test_solver_like_and_models.py` e tournament/model stack |
| `packages/policy_table/__init__.py` | codigo / L3 | re-export de trainer/registry/model |
| `packages/policy_table/model.py` | codigo / `PolicyTableModel`@8 | dataclass da policy table; deps: `typing`; sem IO; cobertura indireta via trainer/registry/model service |
| `packages/policy_table/registry.py` | codigo / `PolicyRegistry`@10 | salva/carrega modelos JSON; deps: `json`, `pathlib`; side effect: `var/models`; cobertura direta em `test_solver_like_and_models.py`, indireta via API |
| `packages/policy_table/trainer.py` | codigo / `PolicyTableTrainer`@12 | agrega rows por bucket e calcula confidence/support; deps: `collections`, `statistics`, `uuid`, buckets; sem IO direto; cobertura direta em `test_solver_like_and_models.py` |
| `packages/solver_like/__init__.py` | codigo / L3 | re-export de labeler/normalize/buckets |
| `packages/solver_like/buckets.py` | codigo / `_spr_bucket`@6; `_equity_bucket`@13; `_price_bucket`@20; `bucketize_spot`@27 | discretizacao estrutural de spots; sem IO; cobertura direta em `test_solver_like_and_models.py` |
| `packages/solver_like/labeler.py` | codigo / `SolverLikeLabeler`@9 | labeler heuristico de acao/confidence; deps: `ActionType`, buckets; sem IO; cobertura direta em `test_solver_like_and_models.py`, indireta via API compare-solver-like |
| `packages/solver_like/normalize.py` | codigo / `normalize_runtime`@8; `normalize_snapshot_trace`@37 | normalizacao de runtime/snapshot para spot; deps: features.poker; sem IO; cobertura direta em `test_solver_like_and_models.py`, indireta via dataset/model stack |
| `packages/spot_packs/__init__.py` | codigo / L1 | re-export de library |
| `packages/spot_packs/library.py` | codigo / `ScriptedAction`@8; `SpotPackScenario`@14; `list_spot_packs`@97 | biblioteca de cenarios guiados; deps: `dataclasses`; sem IO; cobertura direta em `test_spot_packs.py`, `test_solver_like_and_models.py` |
| `packages/state_model/__init__.py` | codigo / L1 | namespace state_model |
| `packages/state_model/contracts.py` | codigo / `ActionLog`@8; `SnapshotLog`@15 | schemas Pydantic de logs de estado; pouco usado no runtime atual; cobertura direta nao identificada |
| `packages/taxonomy/__init__.py` | codigo / L3 | re-export de spot taxonomy |
| `packages/taxonomy/spot_taxonomy.py` | codigo / `classify_trace`@30; `classify_hand`@93 | taxonomia de trace e hand; deps: `collections`, `typing`; sem IO; cobertura direta em `test_taxonomy_and_export.py` e integracao API |

| `packages/cfr_agent/__init__.py` | codigo / L1 | namespace do pacote CFR agent |
| `packages/cfr_agent/agent.py` | codigo / modulo CFR | modulo do agente CFR (Counterfactual Regret Minimization) |
| `packages/cfr_agent/cfr_mix.py` | codigo / modulo CFR | modulo do agente CFR (Counterfactual Regret Minimization) |
| `packages/cfr_agent/compact_cfr.py` | codigo / modulo CFR | modulo do agente CFR (Counterfactual Regret Minimization) |
| `packages/cfr_agent/deep_cfr.py` | codigo / modulo CFR | modulo do agente CFR (Counterfactual Regret Minimization) |
| `packages/cfr_agent/embedding_cfr.py` | codigo / modulo CFR | modulo do agente CFR (Counterfactual Regret Minimization) |
| `packages/cfr_agent/gpu_cfr.py` | codigo / modulo CFR | modulo do agente CFR (Counterfactual Regret Minimization) |
| `packages/cfr_agent/hdcfr.py` | codigo / modulo CFR | modulo do agente CFR (Counterfactual Regret Minimization) |
| `packages/cfr_agent/info_set.py` | codigo / modulo CFR | modulo do agente CFR (Counterfactual Regret Minimization) |
| `packages/cfr_agent/kdb_d2cfr.py` | codigo / modulo CFR | modulo do agente CFR (Counterfactual Regret Minimization) |
| `packages/cfr_agent/lazy_cfr.py` | codigo / modulo CFR | modulo do agente CFR (Counterfactual Regret Minimization) |
| `packages/cfr_agent/mmd.py` | codigo / modulo CFR | modulo do agente CFR (Counterfactual Regret Minimization) |
| `packages/cfr_agent/odcfr.py` | codigo / modulo CFR | modulo do agente CFR (Counterfactual Regret Minimization) |
| `packages/cfr_agent/pruning.py` | codigo / modulo CFR | modulo do agente CFR (Counterfactual Regret Minimization) |
| `packages/cfr_agent/qre.py` | codigo / modulo CFR | modulo do agente CFR (Counterfactual Regret Minimization) |
| `packages/cfr_agent/robust_deep_mccfr.py` | codigo / modulo CFR | modulo do agente CFR (Counterfactual Regret Minimization) |
| `packages/cfr_agent/trainer.py` | codigo / modulo CFR | modulo do agente CFR (Counterfactual Regret Minimization) |
| `packages/cfr_agent/vr_deep_dcfr.py` | codigo / modulo CFR | modulo do agente CFR (Counterfactual Regret Minimization) |
| `packages/cfr_agent/warm_start.py` | codigo / modulo CFR | modulo do agente CFR (Counterfactual Regret Minimization) |
| `packages/dmc_agent/__init__.py` | codigo / L1 | namespace do pacote DMC agent |
| `packages/dmc_agent/agent.py` | codigo / modulo DMC | modulo do agente DMC (Deep Monte Carlo) |
| `packages/equity/neural_equity.py` | codigo / modulo equity | estimador de equity baseado em rede neural |
| `packages/evaluation/aivat.py` | codigo / modulo eval | estimador AIVAT de variancia reduzida |
| `packages/llm_agent/tool_poker.py` | codigo / modulo LLM | ferramentas de poker para agente LLM |
| `packages/mcts_agent/__init__.py` | codigo / L1 | namespace do pacote MCTS agent |
| `packages/mcts_agent/agent.py` | codigo / modulo MCTS | modulo do agente MCTS (Monte Carlo Tree Search) |
| `packages/opponent_model/amp3.py` | codigo / modulo opponent | modulo de modelagem de oponente |
| `packages/opponent_model/bayes_relational.py` | codigo / modulo opponent | modulo de modelagem de oponente |
| `packages/opponent_model/bayesian_range.py` | codigo / modulo opponent | modulo de modelagem de oponente |
| `packages/opponent_model/behavior_prediction.py` | codigo / modulo opponent | modulo de modelagem de oponente |
| `packages/opponent_model/classifier.py` | codigo / modulo opponent | modulo de modelagem de oponente |
| `packages/opponent_model/fatigue_model.py` | codigo / modulo opponent | modulo de modelagem de oponente |
| `packages/opponent_model/meta_game.py` | codigo / modulo opponent | modulo de modelagem de oponente |
| `packages/opponent_model/particle_filter.py` | codigo / modulo opponent | modulo de modelagem de oponente |
| `packages/opponent_model/positional_profile.py` | codigo / modulo opponent | modulo de modelagem de oponente |
| `packages/opponent_model/sad_profiler.py` | codigo / modulo opponent | modulo de modelagem de oponente |
| `packages/opponent_model/sizing_tells.py` | codigo / modulo opponent | modulo de modelagem de oponente |
| `packages/opponent_model/street_patterns.py` | codigo / modulo opponent | modulo de modelagem de oponente |
| `packages/opponent_model/tilt_detector.py` | codigo / modulo opponent | modulo de modelagem de oponente |
| `packages/opponent_model/timing_tells.py` | codigo / modulo opponent | modulo de modelagem de oponente |
| `packages/opponent_model/transformer_model.py` | codigo / modulo opponent | modulo de modelagem de oponente |
| `packages/ppo_agent/__init__.py` | codigo / L1 | namespace do pacote PPO agent |
| `packages/ppo_agent/agent.py` | codigo / modulo PPO | modulo do agente PPO (Proximal Policy Optimization) |
| `packages/solver/__init__.py` | codigo / L1 | namespace do pacote solver |
| `packages/solver/abd.py` | codigo / modulo solver | modulo de resolucao de equilibrio |
| `packages/solver/action_translation.py` | codigo / modulo solver | modulo de resolucao de equilibrio |
| `packages/solver/casper.py` | codigo / modulo solver | modulo de resolucao de equilibrio |
| `packages/solver/equilibrium_refinements.py` | codigo / modulo solver | modulo de resolucao de equilibrio |
| `packages/solver/lamir.py` | codigo / modulo solver | modulo de resolucao de equilibrio |
| `packages/solver/qp_nash.py` | codigo / modulo solver | modulo de resolucao de equilibrio |
| `packages/solver/safe_subgame.py` | codigo / modulo solver | modulo de resolucao de equilibrio |
| `packages/strategy/bias_exploiter.py` | codigo / modulo strategy | modulo de estrategia avancada |
| `packages/strategy/icm.py` | codigo / modulo strategy | modulo de estrategia avancada |
| `packages/strategy/kelly.py` | codigo / modulo strategy | modulo de estrategia avancada |
| `packages/strategy/pcpg.py` | codigo / modulo strategy | modulo de estrategia avancada |
| `packages/strategy/psro.py` | codigo / modulo strategy | modulo de estrategia avancada |
| `packages/training/distillation.py` | codigo / modulo training | modulo de destilacao de conhecimento |

## 4. `services/*`

| Arquivo | Tipo / ancora | Inventario tecnico |
| --- | --- | --- |
| `services/AGENTS.md` | autoritativo operacional / `# services Scope Guide`@1 | ownership funcional da camada de services e regras de composicao |
| `services/alpha_candidate_service/__init__.py` | codigo / L1 | re-export do service |
| `services/alpha_candidate_service/service.py` | codigo / `AlphaCandidateService`@8 | agrega release gate, governance e deploy manifest; sem IO proprio alem de delegacao; cobertura indireta em `test_api_sprint30_endpoints.py` |
| `services/analytics_service/__init__.py` | codigo / L3 | re-export do service |
| `services/analytics_service/service.py` | codigo / `SessionAnalyticsService`@10 | agrega traces, taxonomy e estatisticas de sessao; deps: persistence, taxonomy; sem escrita propria; cobertura direta em `test_taxonomy_and_export.py`, indireta via API |
| `services/batch_generation_service/__init__.py` | codigo / L1 | re-export do service |
| `services/batch_generation_service/service.py` | codigo / `BatchGenerationService`@10 | gera lotes via session runner; deps: `json`, `pathlib`, `uuid`; side effect: `var/batches`; cobertura indireta via `test_api_sprint30_endpoints.py` |
| `services/benchmark_service/__init__.py` | codigo / L1 | re-export do benchmark service |
| `services/benchmark_service/service.py` | codigo / `BenchmarkService`@10 | smoke e H2H benchmark; deps: baseline, engine, harness, experiment runner; side effect: delega relatorios; cobertura direta em `test_api_benchmark.py`, indireta em experiment tests |
| `services/calibration_service/__init__.py` | codigo / L1 | re-export do service |
| `services/calibration_service/service.py` | codigo / `CalibrationService`@6 | calcula `ece_proxy` sobre avaliacao; deps: evaluation service; sem IO proprio alem de retorno dict; cobertura indireta em API sprint30 |
| `services/coach_service/__init__.py` | codigo / L1 | re-export do service |
| `services/coach_service/service.py` | codigo / `CoachService`@11 | monta relatorio de sessao e review de hand; deps: persistence, analytics, opponent profile, taxonomy; sem escrita propria; cobertura direta em `test_opponent_profile_and_coach.py`, integracao API solver/models/coach |
| `services/cqrs/__init__.py` | codigo / L1 | namespace CQRS interno com exports de command/query bus |
| `services/cqrs/bus.py` | codigo / `CommandBus`@39; `QueryBus`@140 | camada de segregacao command/query in-process para rotas `/v1`; centraliza handlers sem alterar contratos HTTP; cobertura indireta via testes de integracao da API |
| `services/curriculum_service/__init__.py` | codigo / L3 | re-export do service |
| `services/curriculum_service/service.py` | codigo / `CurriculumService`@8 | encadeia curriculum planner com analytics e coach; cobertura direta/indireta em `test_tournament_curriculum_readiness.py` e API sprint20 |
| `services/dataset_service/__init__.py` | codigo / L3 | re-export do service |
| `services/dataset_service/service.py` | codigo / `DatasetService`@9 | constroi e lista datasets; deps: dataset builder, solver label service; side effect: `var/datasets`; cobertura direta em `test_dataset_eval_external.py`, indireta via API sprint20/30 |
| `services/decision_service/__init__.py` | codigo / L1 | placeholder de namespace; sem implementacao real; sem cobertura |
| `services/deploy_service/__init__.py` | codigo / L1 | re-export do service |
| `services/deploy_service/service.py` | codigo / `DeployService`@7 | gera deploy manifest baseado em arquivos presentes; deps: `json`, `pathlib`; sem deploy real; cobertura direta em `test_sprint30_services.py` |
| `services/evaluation_service/__init__.py` | codigo / L3 | re-export do service |
| `services/evaluation_service/service.py` | codigo / `EvaluationService`@9 | avalia modelo em dataset e salva relatorio; deps: `json`, `pathlib`, evaluation package; side effect: `var/reports`; cobertura direta em `test_dataset_eval_external.py`, indireta via API |
| `services/experiment_runner/__init__.py` | codigo / L1 | re-export do service |
| `services/experiment_runner/service.py` | codigo / `ExperimentConfig`@16; `ExperimentRunner`@23 | roda matriz H2H e salva relatorio; deps: `json`, `datetime`, `pathlib`, `statistics`; side effect: `var/reports/experiment_*`; cobertura direta em `test_experiment_runner.py`, integracao API experiments |
| `services/explanation_service/__init__.py` | codigo / L1 | placeholder de namespace; sem implementacao real |
| `services/export_service/__init__.py` | codigo / L3 | re-export do service |
| `services/export_service/service.py` | codigo / `ExportService`@7 | expoe export de hand e session em formato PHH-like; deps: exports package, persistence; sem escrita propria; cobertura direta em `test_taxonomy_and_export.py`, integracao API |
| `services/external_solver_service/__init__.py` | codigo / L3 | re-export do service |
| `services/external_solver_service/service.py` | codigo / `ExternalSolverService`@7 | lista catalogo e compara spot pack com solver sample; deps: external solver adapter, solver label service; side effect de leitura em disco; cobertura direta em `test_dataset_eval_external.py`, integracao API |
| `services/game_orchestrator/__init__.py` | codigo / L1 | placeholder de namespace; sem implementacao real |
| `services/governance_service/__init__.py` | codigo / L1 | re-export do service |
| `services/governance_service/service.py` | codigo / `GovernanceService`@8 | gera/lista model cards; deps: `json`, `pathlib`, `typing`; side effect: `var/model_cards`; cobertura indireta via API sprint30 e runtime observado |
| `services/model_service/__init__.py` | codigo / L1 | re-export do service |
| `services/model_service/service.py` | codigo / `ModelService`@10 | treino/list/eval de policy table; deps: policy table packages, solver label service; side effects: `var/models`; cobertura direta em `test_solver_like_and_models.py`, integracao API solver/models |
| `services/opponent_profile_service/__init__.py` | codigo / L1 | re-export do service |
| `services/opponent_profile_service/service.py` | codigo / `OpponentProfileService`@13 | extrai perfis de sessao e adaptive benchmark; deps: adaptive/baseline/harness/opponent_model; side effect: nenhum IO proprio, usa store/benchmark stack; cobertura direta em `test_opponent_profile_and_coach.py`, `test_adaptive_benchmark.py`, integracao API |
| `services/readiness_service/__init__.py` | codigo / L3 | re-export do service |
| `services/readiness_service/service.py` | codigo / `ReadinessService`@6 | snapshot local de readiness por assets/contagens; deps: `pathlib`; sem IO alem de leitura de filesystem; cobertura direta em `test_tournament_curriculum_readiness.py`, indireta via API |
| `services/regression_suite_service/__init__.py` | codigo / L1 | re-export do service |
| `services/regression_suite_service/service.py` | codigo / `RegressionSuiteService`@9 | gera suite de regressao com golden hands e spot packs; deps: `json`, `pathlib`, `spot_packs.library`; side effect: `var/regression`; cobertura direta em `test_sprint30_services.py`, indireta via API |
| `services/release_gate_service/__init__.py` | codigo / L1 | re-export do service |
| `services/release_gate_service/service.py` | codigo / `ReleaseGateService`@7 | combina readiness e metrica minima de avaliacao; deps: readiness/evaluation services; cobertura indireta via API sprint30 e alpha candidate |
| `services/release_notes_service/__init__.py` | codigo / L1 | re-export do service |
| `services/release_notes_service/service.py` | codigo / `ReleaseNotesService`@6 | monta release notes correntes a partir de `docs/`; deps: `pathlib`; sem escrita; cobertura direta em `test_sprint30_services.py` |
| `services/replay_service/__init__.py` | codigo / L1 | re-export do service |
| `services/replay_service/service.py` | codigo / `ReplayService`@9 | reconstrucao de mao persistida; deps: common types, engine, persistence; sem IO alem do DB; cobertura direta em `test_persistence_replay.py`, integracao API replay |
| `services/session_service/__init__.py` | codigo / L1 | re-export do service |
| `services/session_service/service.py` | codigo / `SessionConfig`@18; `SessionRunner`@25 | roda sessoes H2H, persiste hands/traces e salva relatorio; deps: `json`, `datetime`, `pathlib`, `statistics`; side effects: DB + `var/reports/session_*`; cobertura direta em `test_session_runner.py`, `test_store_session_traces.py`, integracao API |
| `services/solver_label_service/__init__.py` | codigo / L1 | re-export do service |
| `services/solver_label_service/service.py` | codigo / `SolverLabelService`@12 | normaliza runtime, compara solver-like, gera rows de treino; deps: baseline, engine, persistence, solver_like; side effect indireto via consumers; cobertura direta em `test_solver_like_and_models.py`, `test_dataset_eval_external.py`, integracao API |
| `services/spot_pack_service/__init__.py` | codigo / L1 | re-export do service |
| `services/spot_pack_service/service.py` | codigo / `SpotPackService`@10 | lista/instancia spot packs e opcionalmente persiste contexto; deps: engine, persistence, spot packs; cobertura direta em `test_spot_packs.py`, integracao API |
| `services/taxonomy_service/__init__.py` | codigo / L3 | re-export do service |
| `services/taxonomy_service/service.py` | codigo / `TaxonomyService`@7 | catalogo e classificacao de hands; deps: persistence, taxonomy package; cobertura direta em `test_taxonomy_and_export.py`, integracao API |
| `services/tournament_service/__init__.py` | codigo / L3 | re-export do service |
| `services/tournament_service/service.py` | codigo / `TournamentService`@13 | round robin entre baseline/adaptive/policy agents; deps: adaptive, baseline, engine, typing, collections; side effect: nenhum IO proprio; cobertura direta em `test_tournament_curriculum_readiness.py`, integracao API |

## 5. `tests/*`

| Arquivo | Tipo / ancora | Inventario tecnico |
| --- | --- | --- |
| `tests/AGENTS.md` | autoritativo operacional / `# tests Scope Guide`@1 | regras de cobertura unit/integration e gate minimo de testes |
| `tests/integration/test_api_benchmark.py` | teste / `test_smoke_benchmark_endpoint`@6 | valida endpoint de benchmark smoke |
| `tests/integration/test_api_engine.py` | teste / `test_create_hand_and_fetch_it`@9 | valida hand lifecycle minimo da API |
| `tests/integration/test_api_experiments.py` | teste / `test_api_h2h_and_experiment_run`@7 | valida benchmark H2H e experiment run |
| `tests/integration/test_api_replay_and_auto.py` | teste / `test_api_auto_and_replay`@7 | valida auto-play, traces e replay |
| `tests/integration/test_api_solver_models_coach.py` | teste / `test_solver_model_and_coach_endpoints`@6 | valida solver-like, model service e coach stack |
| `tests/integration/test_api_spot_packs_and_sessions.py` | teste / `test_api_spot_pack_and_session_endpoints`@7 | valida spot packs e sessions |
| `tests/integration/test_api_sprint20_endpoints.py` | teste / `test_api_sprint20_dataset_tournament_and_readiness`@6 | cobre datasets, tournaments e readiness |
| `tests/integration/test_api_sprint30_endpoints.py` | teste / `_bootstrap_model_and_dataset`@7; `test_sprint30_endpoints`@13 | cobre governance, release gate, batch, regression, calibration, deploy, release notes e alpha candidate |
| `tests/integration/test_api_taxonomy_export_analytics.py` | teste / `test_api_taxonomy_export_and_session_analytics`@6 | valida taxonomy, export e analytics |
| `tests/integration/test_health.py` | teste / `test_health`@6 | valida rota `/health` |
| `tests/unit/test_adaptive_benchmark.py` | teste / `test_adaptive_benchmark_runs`@6 | cobre adaptive benchmark |
| `tests/unit/test_baseline_agent.py` | teste / `test_baseline_agent_returns_legal_action`@6 | cobre legalidade do baseline |
| `tests/unit/test_baseline_agent_sprint04.py` | teste / `test_baseline_agent_is_aggressive_with_premium_preflop`@7 | cobre heuristica preflop do baseline |
| `tests/unit/test_dataset_eval_external.py` | teste / `test_dataset_build_and_eval_and_external_solver`@11 | cobre dataset builder, evaluation e external solver |
| `tests/unit/test_deck.py` | teste / `test_deck_starts_with_52_unique_cards`@4 | cobre deck |
| `tests/unit/test_engine_rounds.py` | teste / multiplos@5/22/38 | cobre transicoes de rounds e fold/call/check |
| `tests/unit/test_engine_setup.py` | teste / `test_start_new_hand_posts_blinds_and_deals_cards`@5 | cobre setup inicial da mao |
| `tests/unit/test_equity.py` | teste / `test_equity_estimator_returns_reasonable_range_for_aces_preflop`@6 | cobre equity Monte Carlo |
| `tests/unit/test_experiment_runner.py` | teste / `test_experiment_runner_generates_report`@5 | cobre relatorio de experimentos |
| `tests/unit/test_features.py` | teste / `test_minimum_features`@6 | cobre derivacao de features |
| `tests/unit/test_golden_hands.py` | teste / `test_golden_hands_regression`@8 | cobre golden hands versionadas |
| `tests/unit/test_harness.py` | teste / `test_agent_vs_agent_harness_is_zero_sum`@4 | cobre harness H2H |
| `tests/unit/test_opponent_profile_and_coach.py` | teste / `test_opponent_profile_and_coach_report`@8 | cobre opponent profile e coach |
| `tests/unit/test_persistence_replay.py` | teste / `test_persistence_roundtrip_replay`@8 | cobre store e replay |
| `tests/unit/test_session_runner.py` | teste / `test_session_runner_persists_session_and_traces`@7 | cobre sessions e traces |
| `tests/unit/test_showdown.py` | teste / `test_pair_of_aces_beats_high_card_after_checkdown`@7; `test_hand_label_for_full_house`@33 | cobre showdown e labels |
| `tests/unit/test_sidepots.py` | teste / `test_uncalled_all_in_is_returned_and_segmented`@5 | cobre side pots e refund |
| `tests/unit/test_solver_like_and_models.py` | teste / `test_solver_like_compare_and_policy_training`@7 | cobre solver-like, trainer, registry e model service |
| `tests/unit/test_spot_packs.py` | teste / `test_spot_pack_instantiation_reaches_expected_actor`@7 | cobre biblioteca de spot packs |
| `tests/unit/test_sprint30_services.py` | teste / tres testes@8/15/22 | cobre deploy manifest, regression suite e release notes |
| `tests/unit/test_train_pokerbench.py` | teste / 7 testes | cobre pipeline de treino PolicyTableModel com dados PokerBench |
| `tests/unit/test_llm_agent.py` | teste / 23 testes | cobre LLMPokerAgent, prompt builder, action parsing e integracao tournament |
| `tests/unit/test_store_session_traces.py` | teste / `test_store_sessions_and_traces`@5 | cobre persistencia de sessions/traces |
| `tests/unit/test_taxonomy_and_export.py` | teste / `test_hand_taxonomy_and_export`@9; `test_session_analytics_has_taxonomy_and_edges`@42 | cobre taxonomy, export e analytics |
| `tests/unit/test_tournament_curriculum_readiness.py` | teste / `test_tournament_curriculum_and_readiness`@11 | cobre tournament, curriculum e readiness |

| `tests/unit/test_advanced_cfr.py` | teste / unitario | suite de testes unitarios |
| `tests/unit/test_batch10_missing_roadmap.py` | teste / unitario | suite de testes unitarios |
| `tests/unit/test_batch11_behavior.py` | teste / unitario | suite de testes unitarios |
| `tests/unit/test_batch1_cfr_optimizations.py` | teste / unitario | suite de testes unitarios |
| `tests/unit/test_batch2_strategy_modules.py` | teste / unitario | suite de testes unitarios |
| `tests/unit/test_batch3_ppo_gpu_cfr.py` | teste / unitario | suite de testes unitarios |
| `tests/unit/test_batch4_deep_cfr_extensions.py` | teste / unitario | suite de testes unitarios |
| `tests/unit/test_batch5_opponent_modeling.py` | teste / unitario | suite de testes unitarios |
| `tests/unit/test_batch6_search_selfplay.py` | teste / unitario | suite de testes unitarios |
| `tests/unit/test_batch7_embedding_hdcfr_qre.py` | teste / unitario | suite de testes unitarios |
| `tests/unit/test_batch8_solvers.py` | teste / unitario | suite de testes unitarios |
| `tests/unit/test_batch9_advanced_systems.py` | teste / unitario | suite de testes unitarios |
| `tests/unit/test_cfr_agent.py` | teste / unitario | suite de testes unitarios |
| `tests/unit/test_pokerbench_adapter.py` | teste / unitario | suite de testes unitarios |

## 6. `var/*` versionado

| Arquivo | Tipo / ancora | Inventario tecnico |
| --- | --- | --- |
| `var/datasets/.gitkeep` | placeholder / L0 | reserva o diretorio de datasets gerados; sinaliza que o artefato canonicamente existe apenas em runtime |
| `var/external_solver/sample_solver_v1.json` | texto / `name`@2; `labels`@3 | sample/mock de solver externo local; consumidor: `ExternalSolverAdapter`; side effect: nenhum, apenas leitura; cobertura direta em `test_dataset_eval_external.py` |
| `var/golden/golden_hands.json` | texto / `hands`@2 | suite canonica de golden hands para regressao do engine/showdown; consumidor: `test_golden_hands.py` |
| `var/model_cards/.gitkeep` | placeholder / L0 | reserva o diretorio de model cards gerados em runtime |
| `var/model_cards/292392e4-4a09-4ba9-a74e-f036dc9b0bac.card.json` | artefato JSON / model card | model card versionado para rastreabilidade e validacao da trilha de governance |
| `var/model_cards/74162623-a510-40a0-930a-449e14052116.card.json` | artefato JSON / model card | model card versionado para rastreabilidade e validacao da trilha de governance |
| `var/models/.gitkeep` | placeholder / L0 | reserva o diretorio de modelos persistidos em runtime |
| `var/regression/.gitkeep` | placeholder / L0 | reserva o diretorio de suites de regressao geradas em runtime |

| `var/model_cards/040006ca-60b2-4263-9267-8efd52bb44e7.card.json` | artefato versionado / model card | metadados de governance para versao de modelo |

Observacao semantica: o workspace local observado contem ainda `var/poker_ai_local.db`, datasets, modelos, model cards, relatorios e batches nao versionados; esses artefatos sao mapeados em `docs/96_System_Knowledge_Dossier.md` e `docs/98_Data_AI_DB_Integration_Map.md`, mas nao entram no baseline tracked do Git.

## 7. `docs/*`, ADRs e diagramas

| Arquivo | Tipo / ancora | Inventario tecnico |
| --- | --- | --- |
| `docs/00_Project_Status.md` | doc / historico de sprint | snapshot inicial de status; usar apenas como contexto historico |
| `docs/01_Sprint_00_Execution_Guide.md` | doc / documentacao tecnica | guia operacional do sprint 00; referencia historica de execucao |
| `docs/02_Sprint_01_Readiness_Check.md` | doc / documentacao tecnica | checklist de readiness do sprint 01; suporte historico |
| `docs/03_Domain_Model_Minimum.md` | doc / autoritativo | define modelo minimo de dominio do poker; util para reconciliar linguagem e entidades |
| `docs/04_API_Contracts_Minimum.md` | doc / autoritativo | contrato minimo historico da API; deve ser lido contra `apps/api/main.py` |
| `docs/05_Logging_and_Trace_Spec.md` | doc / autoritativo | especifica trilha de logs e decision traces; consumidor: stack de persistencia/analytics |
| `docs/06_Test_Strategy.md` | doc / autoritativo | estrategia de testes do laboratorio local; apoio ao entendimento da cobertura |
| `docs/07_FinOps_Local_Budget.md` | doc / historico de sprint | snapshot de custo local inicial; sem impacto de runtime |
| `docs/08_Open_Questions_Remaining.md` | doc / documentacao tecnica | backlog de duvidas abertas em fase inicial |
| `docs/09_Roadmap_Snapshot.md` | doc / historico de sprint | snapshot inicial de roadmap |
| `docs/10_Sprint_01_Engine_POC.md` | doc / documentacao tecnica | entrega e escopo tecnico do engine POC |
| `docs/11_Sprint_01_What_Changed.md` | doc / historico de sprint | changelog do sprint 01 |
| `docs/12_Sprint_02_Preview.md` | doc / historico de sprint | preview de sprint subsequente |
| `docs/13_FinOps_Local_Sprint_01.md` | doc / historico de sprint | update FinOps do sprint 01 |
| `docs/14_Sprint_02_Persistence_Baseline.md` | doc / documentacao tecnica | documenta a introducao de persistencia SQLite e baseline |
| `docs/15_Sprint_02_What_Changed.md` | doc / historico de sprint | changelog do sprint 02 |
| `docs/16_Sprint_03_Preview.md` | doc / historico de sprint | preview do sprint 03 |
| `docs/17_FinOps_Local_Sprint_02.md` | doc / historico de sprint | update FinOps do sprint 02 |
| `docs/18_Status_Snapshot_Sprint_02.md` | doc / historico de sprint | snapshot consolidado do estado apos sprint 02 |
| `docs/19_Sprint_03_Sidepots_Harness.md` | doc / documentacao tecnica | detalha side pots e harness H2H |
| `docs/20_Sprint_03_What_Changed.md` | doc / historico de sprint | changelog do sprint 03 |
| `docs/21_Sprint_04_Preview.md` | doc / historico de sprint | preview do sprint 04 |
| `docs/22_Status_Snapshot_Sprint_03.md` | doc / historico de sprint | snapshot do estado apos sprint 03 |
| `docs/23_Sprint_04_Eval_Runner.md` | doc / documentacao tecnica | introduz runner de avaliacao e benchmark |
| `docs/24_Sprint_04_What_Changed.md` | doc / historico de sprint | changelog do sprint 04 |
| `docs/25_Sprint_05_Preview.md` | doc / historico de sprint | preview do sprint 05 |
| `docs/26_Status_Snapshot_Sprint_04.md` | doc / historico de sprint | snapshot do estado apos sprint 04 |
| `docs/27_FinOps_Local_Sprint_04.md` | doc / historico de sprint | update FinOps do sprint 04 |
| `docs/28_Master_Audit_Double_Check.md` | doc / auditoria / atual | auditoria intermediaria; apoio tecnico, nao substitui codigo/testes |
| `docs/29_Business_Requirements_Document.md` | doc / autoritativo | BRD do sistema; referencia de objetivo funcional e escopo |
| `docs/30_Functional_Specification.md` | doc / autoritativo | especificacao funcional consolidada |
| `docs/31_Technical_Specification.md` | doc / autoritativo | especificacao tecnica consolidada |
| `docs/32_AI_Data_Specification.md` | doc / autoritativo | especificacao de IA, dados e artefatos |
| `docs/33_FinOps_Master_Plan.md` | doc / historico de sprint | plano FinOps historico |
| `docs/34_Roadmap_Detailed.md` | doc / historico de sprint | roadmap detalhado historico |
| `docs/35_Project_Status_Where_We_Are.md` | doc / historico de sprint | snapshot de status intermediario |
| `docs/36_Prompt_Backup_Master.md` | doc / historico de sprint | backup operacional de prompt/contexto; valor tecnico baixo |
| `docs/37_Repository_Sync_Note.md` | doc / documentacao tecnica | nota de sincronizacao do repositorio |
| `docs/38_Sprint_05_Spot_Packs_Session_Eval.md` | doc / documentacao tecnica | documenta spot packs, sessions e eval |
| `docs/39_Sprint_05_What_Changed.md` | doc / historico de sprint | changelog do sprint 05 |
| `docs/40_Sprint_06_Preview.md` | doc / historico de sprint | preview do sprint 06 |
| `docs/41_Decision_Log_Master.md` | doc / documentacao tecnica | log mestre de decisoes do projeto |
| `docs/42_Sprint_06_Taxonomy_Analytics_Export.md` | doc / documentacao tecnica | documenta taxonomy, analytics e export |
| `docs/43_Sprint_06_What_Changed.md` | doc / historico de sprint | changelog do sprint 06 |
| `docs/44_Status_Snapshot_Sprint_06.md` | doc / historico de sprint | snapshot apos sprint 06 |
| `docs/45_Sprint_07_Preview.md` | doc / historico de sprint | preview do sprint 07 |
| `docs/46_Sprint_07_Solver_Like_Normalized_Spots.md` | doc / documentacao tecnica | documenta trilha solver-like e normalizacao de spots |
| `docs/47_Sprint_08_Policy_Table_Training.md` | doc / documentacao tecnica | documenta treino da policy table |
| `docs/48_Sprint_09_Opponent_Profile_Adaptive.md` | doc / documentacao tecnica | documenta adaptive agent e profiling |
| `docs/49_Sprint_10_Coach_Review_Readiness.md` | doc / documentacao tecnica | documenta coach/review/readiness local |
| `docs/50_Status_Snapshot_Sprint_10.md` | doc / historico de sprint | snapshot apos sprint 10 |
| `docs/51_Roadmap_Sprint_07_to_10_and_Beyond.md` | doc / historico de sprint | roadmap historico de medio prazo |
| `docs/52_Prompt_Backup_Update_Sprint_10.md` | doc / historico de sprint | backup operacional de prompt/contexto |
| `docs/53_FinOps_Local_Sprint_07_to_10.md` | doc / historico de sprint | update FinOps do bloco 07-10 |
| `docs/54_Testing_Gates_Sprint_10.md` | doc / documentacao tecnica | gates de teste do estado apos sprint 10 |
| `docs/55_Known_Limits_After_Sprint_10.md` | doc / historico de sprint | registro historico de limites conhecidos |
| `docs/56_Sprint_11_Preview.md` | doc / historico de sprint | preview do sprint 11 |
| `docs/57_Project_Status_What_Is_Done_vs_Not_Done.md` | doc / historico de sprint | quadro de concluido vs nao concluido; usar com leitura conservadora |
| `docs/58_Sprint_11_Dataset_Builder.md` | doc / documentacao tecnica | documenta dataset builder |
| `docs/59_Sprint_12_Evaluation_By_Taxonomy.md` | doc / documentacao tecnica | documenta avaliacao por taxonomy |
| `docs/60_Sprint_13_External_Solver_Adapter.md` | doc / documentacao tecnica | documenta adapter externo mock/local |
| `docs/61_Sprint_14_Distillation_Export.md` | doc / documentacao tecnica | documenta export/distillation local |
| `docs/62_Sprint_15_Tournament_Leaderboard.md` | doc / documentacao tecnica | documenta tournament e leaderboard |
| `docs/63_Sprint_16_Experiment_Matrix.md` | doc / documentacao tecnica | documenta experimento matricial H2H |
| `docs/64_Sprint_17_Curriculum_Study_Layer.md` | doc / documentacao tecnica | documenta curriculum/study layer |
| `docs/65_Sprint_18_Release_Readiness.md` | doc / documentacao tecnica | documenta readiness local |
| `docs/66_Sprint_19_Packaging_And_Deploy_Assets.md` | doc / documentacao tecnica | documenta packaging e artefatos de deploy local |
| `docs/67_Sprint_20_Production_Handoff_Preview.md` | doc / historico de sprint | preview/handoff historico para promocao |
| `docs/68_Status_Snapshot_Sprint_20.md` | doc / historico de sprint | snapshot apos sprint 20 |
| `docs/69_Roadmap_Sprint_11_to_20_and_Beyond.md` | doc / historico de sprint | roadmap historico do bloco 11-20 |
| `docs/70_Prompt_Backup_Update_Sprint_20.md` | doc / historico de sprint | backup operacional de prompt/contexto |
| `docs/71_Known_Limits_After_Sprint_20.md` | doc / historico de sprint | limites conhecidos apos sprint 20 |
| `docs/72_Sprint_21_Governance_Model_Cards.md` | doc / documentacao tecnica | documenta governance e model cards |
| `docs/73_Sprint_22_Release_Gate.md` | doc / documentacao tecnica | documenta release gate local |
| `docs/74_Sprint_23_Batch_Generation.md` | doc / documentacao tecnica | documenta batch generation |
| `docs/75_Sprint_24_Regression_Suite.md` | doc / documentacao tecnica | documenta regression suite |
| `docs/76_Sprint_25_Calibration.md` | doc / documentacao tecnica | documenta calibracao |
| `docs/77_Sprint_26_Deploy_Manifest.md` | doc / documentacao tecnica | documenta deploy manifest local |
| `docs/78_Sprint_27_Release_Notes.md` | doc / documentacao tecnica | documenta release notes locais |
| `docs/79_Sprint_28_Runbooks_And_Alpha.md` | doc / documentacao tecnica | documenta runbooks e Alpha local |
| `docs/80_Sprint_29_Roadmap_To_Last_Sprint.md` | doc / historico de sprint | roadmap historico do fechamento |
| `docs/81_Sprint_30_Final_Handoff.md` | doc / historico de sprint | handoff final historico do sprint 30 |
| `docs/82_Status_Snapshot_Sprint_30.md` | doc / historico de sprint | snapshot apos sprint 30 |
| `docs/83_Roadmap_Sprint_21_to_30_and_Closeout.md` | doc / historico de sprint | roadmap historico do bloco 21-30 |
| `docs/84_Prompt_Backup_Update_Sprint_30.md` | doc / historico de sprint | backup operacional de prompt/contexto |
| `docs/85_Final_Test_Report.md` | doc / historico de sprint | relatorio final historico de testes |
| `docs/86_GitHub_Ready_Checklist.md` | doc / historico de sprint | checklist historico de publicacao/preparacao |
| `docs/87_Alpha_Beta_Production_Plan.md` | doc / historico de sprint | plano historico de promocao |
| `docs/88_Rigorous_Audit_2026_03_15.md` | doc / auditoria / atual | auditoria recente do projeto; apoio tecnico complementar |
| `docs/89_Gap_Register_and_Fixes.md` | doc / auditoria / atual | registro anterior de gaps e correcoes |
| `docs/90_Local_Docker_Validation_Guide.md` | doc / documentacao tecnica | guia de validacao local via Docker |
| `docs/93_Audit_Validation_Report.md` | doc / auditoria / atual | validacao de auditoria previa |
| `docs/94_Local_Execution_Runbook.md` | doc / autoritativo | runbook local atual; referencia operacional principal |
| `docs/95_External_Design_Check.md` | doc / auditoria / atual | validacao externa de design/escopo |
| `docs/96_System_Knowledge_Dossier.md` | doc / auditoria / atual | dossie permanente de conhecimento do sistema |
| `docs/97_File_By_File_Inventory.md` | doc / auditoria / atual | inventario canonico file-by-file do repositorio |
| `docs/98_Data_AI_DB_Integration_Map.md` | doc / auditoria / atual | mapa canonico de dados, IA, DB e integracoes |
| `docs/99_Gap_and_Risk_Register.md` | doc / auditoria / atual | gap register tecnico e risco operacional vigente |
| `docs/100_Multi_Specialist_Audit_and_Frontier_Roadmap.md` | doc / auditoria / atual | parecer multi-especialista e roadmap frontier com status objetivo de maturidade |
| `docs/AGENTS.md` | doc / autoritativo operacional | regras locais de navegacao e manutencao da pasta `docs/` |
| `docs/adr/0001_local_first.md` | doc / ADR / autoritativo | registra a decisao arquitetural de local-first |
| `docs/adr/0002_mvp_scope_heads_up.md` | doc / ADR / autoritativo | registra a decisao de focar no escopo HUNL |
| `docs/adr/0003_decision_and_explanation_are_separate.md` | doc / ADR / autoritativo | registra a separacao entre decisao e explicacao |
| `docs/adr/0004_engine_before_solver.md` | doc / ADR / autoritativo | registra a priorizacao do engine antes do solver |
| `docs/diagrams/01_system_context.md` | doc / diagrama / suporte | diagrama de contexto do sistema |
| `docs/diagrams/02_container_view.md` | doc / diagrama / suporte | diagrama de containers/modulos |
| `docs/diagrams/03_runtime_decision_sequence.md` | doc / diagrama / suporte | sequencia de decisao em runtime |
| `docs/diagrams/04_training_and_eval_pipeline.md` | doc / diagrama / suporte | pipeline de treino e avaliacao |
| `docs/diagrams/05_roadmap_flow.md` | doc / diagrama / suporte | fluxo visual de roadmap |
| `docs/diagrams/06_session_eval_flow.md` | doc / diagrama / suporte | fluxo de sessions e avaliacao |
| `docs/diagrams/07_phh_export_and_analytics.md` | doc / diagrama / suporte | fluxo de export PHH-like e analytics |
| `docs/diagrams/08_solver_like_pipeline.md` | doc / diagrama / suporte | pipeline solver-like |
| `docs/diagrams/09_opponent_profile_and_adaptive.md` | doc / diagrama / suporte | fluxo de profiling e adaptatividade |
| `docs/diagrams/10_coach_report_flow.md` | doc / diagrama / suporte | fluxo do coach report |
| `docs/diagrams/11_dataset_builder_and_eval.md` | doc / diagrama / suporte | fluxo do dataset builder e avaliacao |
| `docs/diagrams/12_tournament_and_readiness.md` | doc / diagrama / suporte | fluxo de tournament e readiness |

| `docs/99_AI_Research_Roadmap.md` | doc / roadmap AI | roadmap de pesquisa em IA e fronteira tecnica |

## 8. Complementos do baseline atual

| Arquivo | Tipo / ancora | Inventario tecnico |
| --- | --- | --- |
| `apps/api/container.py` | codigo / modulo DI | composicao explicita de dependencias e wiring da camada API |
| `apps/api/contracts.py` | codigo / contratos HTTP | modelos de request/response auxiliares da API |
| `apps/api/routers/__init__.py` | codigo / pacote routers | inicializacao do namespace de rotas |
| `apps/api/routers/benchmark.py` | codigo / router benchmark | superficie HTTP de benchmark e torneios |
| `apps/api/routers/governance.py` | codigo / router governance | endpoints de governance, release e readiness |
| `apps/api/routers/hands.py` | codigo / router hands | ciclo de vida de maos e auto-play |
| `apps/api/routers/ml.py` | codigo / router ML | treino/avaliacao de modelos e datasets |
| `apps/api/routers/sessions.py` | codigo / router sessions | execucao H2H, analytics e traces |
| `apps/api/routers/spots.py` | codigo / router spots | endpoints de taxonomia e spots |
| `apps/api/routers/system.py` | codigo / router system | health, readiness, alpha-candidate e serving da SPA em `/ui` |
| `apps/api/routers/utils.py` | codigo / utilitario routers | normalizacao de erros e payloads auxiliares |
| `docs/101_Exploratory_Data_Analysis_Full_IO_Features.md` | doc / auditoria tecnica | relatorio de EDA e cobertura de I/O |
| `docs/102_Architecture_Conformance_Report.md` | doc / auditoria tecnica | status de conformidade arquitetural atual |
| `docs/103_Architecture_Remediation_Plan.md` | doc / plano tecnico | plano de remediacao arquitetural por fase |
| `docs/adr/0005_monolito_modular_harden.md` | doc / ADR / autoritativo | decisao de endurecimento do monolito modular |
| `infra/scripts/architecture_fitness_checks.py` | script / guardrail | checks arquiteturais de fronteira/modularidade |
| `infra/scripts/audit_consistency_check.py` | script / guardrail | valida consistencia inventario, endpoints e baseline docs |
| `infra/scripts/mr_poker_full_eda.py` | script / analise | extracao de inventario EDA completo do repositorio |
| `tests/integration/test_api_ui_routes.py` | teste / integracao | valida `/ui`, assets, fallback SPA e regressao API |
| `tests/unit/test_architecture_fitness_checks.py` | teste / unitario | valida script de fitness arquitetural |
| `tests/unit/test_full_eda_extractor.py` | teste / unitario | valida extrator de EDA full I/O |
| `var/model_cards/1f2ec7e0-416c-49f3-91f8-ed94bf4e3952.card.json` | artefato versionado / model card | metadados de governance para versao de modelo |

## 9. Arquivos adicionais do baseline atual

| Arquivo | Tipo / ancora | Inventario tecnico |
| --- | --- | --- |
| `apps/__init__.py` | codigo / L1 | namespace raiz do pacote apps |
| `apps/api/__init__.py` | codigo / L1 | namespace do pacote API |
| `apps/api/middleware/__init__.py` | codigo / L1 | namespace do pacote de middlewares |
| `apps/api/middleware/error_handler.py` | codigo / middleware | tratamento centralizado de erros HTTP |
| `apps/api/middleware/rate_limit.py` | codigo / middleware | limitacao de taxa de requisicoes |
| `apps/api/middleware/request_logging.py` | codigo / middleware | logging estruturado de requisicoes HTTP |
| `apps/api/middleware/security_headers.py` | codigo / middleware | headers de seguranca HTTP |
| `apps/api/pagination.py` | codigo / utilitario | paginacao generica para endpoints de listagem |
| `apps/api/response_models.py` | codigo / contratos | modelos de resposta padronizados da API |
| `apps/api/routers/tasks.py` | codigo / router tasks | endpoints de gerenciamento de tasks assincronas |
| `apps/api/routers/ws.py` | codigo / router websocket | endpoints de comunicacao WebSocket |
| `configs/app.prod.json` | config / JSON | configuracao de producao da aplicacao |
| `configs/app.test.json` | config / JSON | configuracao de teste da aplicacao |
| `docs/architecture/README.md` | doc / indice arquitetural | indice da documentacao de arquitetura |
| `docs/architecture/adr/001-sqlite.md` | doc / ADR | decisao de uso do SQLite como persistencia |
| `docs/architecture/adr/002-cqrs.md` | doc / ADR | decisao de adocao do padrao CQRS |
| `docs/architecture/adr/003-saga.md` | doc / ADR | decisao de uso do padrao Saga |
| `docs/architecture/adr/004-jwt-auth.md` | doc / ADR | decisao de autenticacao via JWT |
| `docs/architecture/data-model.md` | doc / autoritativo | modelo de dados arquitetural |
| `docs/development/api-guide.md` | doc / guia tecnico | guia de desenvolvimento da API |
| `docs/development/getting-started.md` | doc / guia tecnico | guia de inicio rapido para desenvolvedores |
| `docs/operations/deployment.md` | doc / operacional | guia de deployment |
| `docs/operations/monitoring.md` | doc / operacional | guia de monitoramento |
| `docs/operations/troubleshooting.md` | doc / operacional | guia de troubleshooting |
| `infra/scripts/perf_gate_v1.py` | script / guardrail | gate de performance v1 para validacao de baseline |
| `infra/scripts/train_pokerbench_policy.py` | script / treino | treina PolicyTableModel a partir de dados solver do PokerBench (HuggingFace) |
| `infra/scripts/upload_pokerbench_sft.py` | script / dados | converte PokerBench para formato chat SFT e faz upload para HuggingFace Hub |
| `packages/__init__.py` | codigo / L1 | namespace raiz do pacote packages |
| `packages/audit/__init__.py` | codigo / L1 | namespace do pacote de auditoria |
| `packages/audit/logger.py` | codigo / modulo | logger especializado para auditoria |
| `packages/auth/__init__.py` | codigo / L1 | namespace do pacote de autenticacao |
| `packages/auth/dependencies.py` | codigo / modulo | dependencias de autenticacao para injecao |
| `packages/auth/jwt_handler.py` | codigo / modulo | geracao e validacao de tokens JWT |
| `packages/auth/models.py` | codigo / modulo | modelos de dados de autenticacao |
| `packages/baseline_agent/factory.py` | codigo / modulo | factory para criacao de instancias do baseline agent |
| `packages/baseline_agent/strategy.py` | codigo / modulo | estrategias configuráveis do baseline agent |
| `packages/common/bounded_dict.py` | codigo / modulo | dicionario com tamanho limitado para cache |
| `packages/config/__init__.py` | codigo / L1 | namespace do pacote de configuracao |
| `packages/config/settings.py` | codigo / modulo | settings centralizados da aplicacao |
| `packages/engine/exceptions.py` | codigo / modulo | excecoes customizadas do engine |
| `packages/equity/range_equity.py` | codigo / modulo | calculo de equity por range de maos |
| `packages/evaluation/calibration.py` | codigo / modulo | calibracao de modelos de avaliacao |
| `packages/evaluation/metrics.py` | codigo / modulo | metricas de avaliacao de performance |
| `packages/events/__init__.py` | codigo / L1 | namespace do pacote de eventos |
| `packages/events/models.py` | codigo / modulo | modelos de eventos do dominio |
| `packages/events/store.py` | codigo / modulo | store de eventos para event sourcing |
| `packages/external_solver/api_client.py` | codigo / modulo | cliente HTTP para solver externo |
| `packages/features/board_texture.py` | codigo / modulo | analise de textura do board |
| `packages/features/pipeline.py` | codigo / modulo | pipeline de extracao de features |
| `packages/features/temporal.py` | codigo / modulo | features temporais de decisao |
| `packages/logging_config/__init__.py` | codigo / L1 | namespace do pacote de configuracao de logging |
| `packages/logging_config/setup.py` | codigo / modulo | setup centralizado de logging |
| `packages/metrics/__init__.py` | codigo / L1 | namespace do pacote de metricas |
| `packages/metrics/collector.py` | codigo / modulo | coletor de metricas de runtime |
| `packages/persistence/database.py` | codigo / modulo | camada de acesso ao banco de dados |
| `packages/persistence/interfaces.py` | codigo / modulo | interfaces abstratas de persistencia |
| `packages/strategy/__init__.py` | codigo / L1 | namespace do pacote de estrategia |
| `packages/strategy/mixed.py` | codigo / modulo | estrategias mistas de decisao |
| `packages/task_queue/__init__.py` | codigo / L1 | namespace do pacote de fila de tarefas |
| `packages/task_queue/worker.py` | codigo / modulo | worker de processamento de tarefas assincronas |
| `packages/tracing/__init__.py` | codigo / L1 | namespace do pacote de tracing |
| `packages/tracing/setup.py` | codigo / modulo | setup de tracing distribuido |
| `packages/training/__init__.py` | codigo / L1 | namespace do pacote de treinamento |
| `packages/training/cross_validation.py` | codigo / modulo | cross-validation para modelos de policy |
| `packages/training/trainer.py` | codigo / modulo | trainer generico para modelos |
| `services/__init__.py` | codigo / L1 | namespace raiz do pacote services |
| `services/cqrs/saga.py` | codigo / modulo | orquestracao de sagas CQRS |
| `services/cqrs/saga_store.py` | codigo / modulo | persistencia de estado de sagas |
| `services/decision_service/service.py` | codigo / modulo | servico de decisao; implementacao pendente |
| `services/explanation_service/service.py` | codigo / modulo | servico de explicacao; implementacao pendente |
| `services/game_orchestrator/service.py` | codigo / modulo | servico de orquestracao de jogo; implementacao pendente |
| `tests/conftest.py` | teste / config | fixtures compartilhadas do pytest |
| `tests/e2e/__init__.py` | teste / L1 | namespace do pacote de testes end-to-end |
| `tests/e2e/test_full_hand_lifecycle.py` | teste / e2e | valida ciclo de vida completo de uma mao |
| `tests/e2e/test_full_session_lifecycle.py` | teste / e2e | valida ciclo de vida completo de uma sessao |
| `tests/property/__init__.py` | teste / L1 | namespace do pacote de testes de propriedade |
| `tests/property/test_engine_properties.py` | teste / property-based | valida propriedades invariantes do engine |
| `tests/unit/test_cqrs_saga.py` | teste / unitario | valida orquestracao de sagas CQRS |
| `tests/unit/test_equity_monte_carlo_paths.py` | teste / unitario | valida caminhos do estimador de equity Monte Carlo |
| `tests/unit/test_evaluator_hands_paths.py` | teste / unitario | valida caminhos do avaliador de maos |
| `tests/unit/test_sqlite_store_paths.py` | teste / unitario | valida caminhos do store SQLite |
| `var/model_cards/013272dc-fad5-4b81-9e99-3f8d124163e6.card.json` | artefato versionado / model card | metadados de governance para versao de modelo |
| `var/model_cards/02a7d58f-345f-49da-9602-9be79805d623.card.json` | artefato versionado / model card | metadados de governance para versao de modelo |
| `var/model_cards/0fc335da-48e8-4f16-8453-cdba697a2964.card.json` | artefato versionado / model card | metadados de governance para versao de modelo |

## 10. Fechamento do inventario

- Este inventario cobre o baseline tracked atual do repositorio (`517` arquivos versionados).
- Artefatos runtime nao versionados foram auditados semanticamente, mas permanecem fora do inventario canonico do Git.
- Para leitura arquitetural e operacional, usar este arquivo em conjunto com `docs/96_System_Knowledge_Dossier.md`, `docs/98_Data_AI_DB_Integration_Map.md` e `docs/99_Gap_and_Risk_Register.md`.
