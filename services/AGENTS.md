# services Scope Guide

## Escopo

`services/*` faz a costura entre packages, persistencia local e artefatos operacionais. Esta e a camada de ownership funcional do produto.

## Regras locais

- Cada service deve ter uma responsabilidade clara e um artefato de saida identificavel.
- Services nao devem depender de `FastAPI`; eles devem receber objetos do dominio e devolver `dict`, `list` ou modelos simples.
- Quando houver escrita em disco, use caminhos previsiveis sob `var/` e documente o contrato.
- Quando houver dependencia cruzada entre services, mantenha a direcao simples e sem ciclos.

## Ownership esperado

- Sessao/bench/experimentos: `session_service`, `benchmark_service`, `experiment_runner`.
- Dados/modelos: `dataset_service`, `model_service`, `evaluation_service`, `calibration_service`.
- Governanca/promocao: `governance_service`, `release_gate_service`, `deploy_service`, `alpha_candidate_service`.
- Analise/estudo: `analytics_service`, `coach_service`, `curriculum_service`, `taxonomy_service`, `opponent_profile_service`.

## Validacao

- Sempre revalide `python -m pytest -q`.
- Ao tocar services expostos na API, confira os endpoints correspondentes em `apps/api/main.py`.

