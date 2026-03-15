# packages Scope Guide

## Escopo

`packages/*` concentra o nucleo tecnico e funcional do sistema: engine, modelos de estado, avaliacao, features, agents, persistencia, solver-like e artefatos de treino local.

## Regras locais

- Prefira pureza funcional e determinismo quando possivel.
- Separar claramente dominio, feature derivation, policy heuristics e persistencia.
- Side effects aceitos aqui devem ser explicitos:
  - `packages/persistence/*`: SQLite.
  - `packages/exports/*`: materializacao de payloads.
  - `packages/dataset_builder/*`: escrita em `var/datasets`.
  - `packages/external_solver/*`: leitura de catalogo/arquivo mock.
- Evite importar `FastAPI` ou detalhes HTTP nesta camada.

## Convencoes

- `__init__.py` expone a superficie do subpacote; mantenha-os finos.
- Quando adicionar uma feature ou bucket novo, atualize tambem a documentacao de IA/dados.
- Se um modulo depende de `var/`, documente produtor, formato e consumidor.

## Testes

- Cobertura esperada em `tests/unit/*`.
- Quando o pacote for consumido pela API, tambem confirme exercicio indireto em `tests/integration/*`.

