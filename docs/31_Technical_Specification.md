# Technical Specification

## Arquitetura lógica
- `packages/engine`: regras, state machine, settlement, snapshots
- `packages/evaluator`: ranking de mãos
- `packages/equity`: Monte Carlo leve
- `packages/features`: features mínimas e de poker
- `packages/baseline_agent`: decisão explicável
- `packages/persistence`: SQLite local
- `packages/harness`: partidas agent-vs-agent
- `packages/spot_packs`: biblioteca de spots
- `services/replay_service`: replay determinístico
- `services/benchmark_service`: smoke/H2H
- `services/experiment_runner`: relatórios de experimento
- `services/session_service`: sessões locais persistidas
- `services/spot_pack_service`: criação de cenários de treino
- `apps/api`: FastAPI local

## Contratos principais
- snapshot canônico da mão
- action log ordenado
- decision trace estruturado
- summary de benchmark
- summary de sessão

## Estratégia técnica atual
- local-first
- SQLite para persistência local
- JSON para relatórios
- FastAPI para interface local
- testes unitários e integração com pytest

## Estratégia técnica futura
- PHH como formato canônico de treino
- storage de objetos para relatórios e datasets
- registry de modelos
- pipelines de treino dedicados
- track solver offline + distilação
