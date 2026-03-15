# tests Scope Guide

## Escopo

Os testes versionados se dividem em `unit` e `integration`.

## Regras locais

- `tests/unit/*`: valida comportamento de packages e services sem depender de servidor HTTP real.
- `tests/integration/*`: valida o contrato de `apps/api/main.py` e o encadeamento minimo com persistencia e artefatos.
- Prefira nomes de teste orientados a comportamento observavel.
- Quando um comportamento estiver apenas coberto por integracao, documente esse fato no inventario.

## Gate minimo

- Baseline atual confirmado: `40` testes passando com `pytest -q`.
- Se uma mudanca tocar API, banco ou `var/*`, espere impacto tanto em unit quanto em integration.

