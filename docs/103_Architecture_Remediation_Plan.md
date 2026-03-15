# 103 - Architecture Remediation Plan

## Tipo

- classificacao: auditoria
- escopo: remediacao decisao-completa (incremental)

## Contexto

Este plano consolida a auditoria arquitetural do `mr_poker` para o alvo **monolito modular harden** e define backlog executavel com gates objetivos.

## Decisoes Fechadas

1. Arquitetura oficial: `monolito_modular_harden` (ADR-0005).
2. Fronteiras obrigatorias:
   - `apps` nao recebe dependencia de `services` invertida.
   - `services` nao depende de `apps` nem de `fastapi`.
   - `packages` nao depende de `services` nem de `apps`.
3. Conformidade passa a ser validada por gate automatizado:
   - `infra/scripts/architecture_fitness_checks.py --strict`
4. Integridade de contratos API e cobertura de features passa por gate:
   - `infra/scripts/mr_poker_full_eda.py`
   - `infra/scripts/audit_consistency_check.py --strict`

## Matriz de Remediacao

| Tema | Decisao | Status | Acao |
| --- | --- | --- | --- |
| Composicao API | Router split + DI container sem quebrar contrato HTTP | concluido | manter snapshot/contract tests |
| Fronteiras em camadas | fitness checks automatizados | concluido | manter regra em CI |
| GameEngine | quebrar por responsabilidades (transicao, validacao, payout, snapshot) | pendente (P0) | extrair modulos com testes de invariantes |
| Solver capability | classificar `mock/heuristico/real` sem overclaim | pendente (P0) | adicionar contrato de capacidade e evidencias |
| Qualidade estrategica Poker AI | benchmark com exploitability proxy + BR-lite + reducao de variancia | pendente (P0) | criar suite de benchmark e thresholds |
| Cross-platform ops | consolidar execucao em PowerShell e bash | pendente (P1) | scripts equivalentes e smoke checks |

## Gates Obrigatorios

- `python -m pytest -q`
- `python infra/scripts/audit_consistency_check.py --strict`
- `python infra/scripts/architecture_fitness_checks.py --strict`
- `python infra/scripts/mr_poker_full_eda.py --include-var-runtime`

## Criterio de Garantia

Garantia tecnica e tratada como **compliance verificavel por evidencia**:
- sem violacoes de fronteira de arquitetura;
- sem gaps criticos em contratos API/features;
- sem regressao de testes;
- sem pendencias P0 abertas para declarar conformidade final.
