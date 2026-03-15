# ADR 0005 — Monolito Modular Harden

## Decisao
A arquitetura oficial do `mr_poker` passa a ser **monolito modular harden**.

## Fronteiras obrigatorias
- `apps` orquestra HTTP e composicao.
- `services` concentra casos de uso e contratos internos.
- `packages` concentra regras de dominio e blocos reutilizaveis.
- Nao e permitido:
  - `services` depender de `apps`;
  - `packages` depender de `services` ou `apps`;
  - `services` depender de framework HTTP (`fastapi`).

## Motivo
Essa forma reduz acoplamento acidental, melhora testabilidade e cria gates automatizaveis de conformidade arquitetural.

## Evidencia operacional
- Verificacao automatizada em `infra/scripts/architecture_fitness_checks.py`.
- Relatorio versionado em `docs/102_Architecture_Conformance_Report.md` e `var/reports/architecture_conformance_report.json`.
