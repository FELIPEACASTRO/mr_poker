# Sprint 04 — Baseline+ Evaluation Runner

## Entregas
- baseline agent reforçado por equity + pot odds
- decision trace enriquecido
- benchmark H2H com agregação de bb/100
- experiment runner local com relatório JSON

## Gate
- testes verdes
- endpoint `/v1/benchmark/h2h` funcional
- endpoint `/v1/experiments/run` gravando relatório em `var/reports/`

## Limites
- ainda não há exploitability formal
- equity é Monte Carlo leve, não solver-grade
