# FinOps local — Sprint 07 a 10

## Custos controlados no modo local
- labels solver-like são heurísticos e baratos
- policy table local evita treino pesado
- benchmark adaptativo roda em CPU
- artifacts são persistidos em JSON/SQLite

## Guardrails
- limite de mãos por benchmark local
- limite de relatórios/artefatos em `var/`
- evitar Monte Carlo excessivo fora de testes/benchmarks

## Próximo estágio FinOps
- quando o solver real entrar, separar orçamento por:
  - geração de labels
  - avaliação H2H
  - self-play
  - inferência local
