# AI & Data Specification

## Tipos de dados do projeto
1. hand histories canônicas
2. snapshots de estado
3. action logs
4. decision traces
5. relatórios de benchmark
6. relatórios de sessão
7. spot packs
8. datasets externos futuros (PHH, solver labels, PokerBench)

## Features já usadas
- street
- hole_class
- board_texture
- pot
- to_call
- pot_odds
- estimated_equity
- legal_actions
- flags de par/suited/conectividade
- board size
- SPR

## Features planejadas
- range posterior
- nut advantage
- range advantage
- sizing buckets
- last aggressor history por street
- opponent profile longitudinal
- taxonomy formal de spot

## Modelos atuais
- baseline deterministic agent
- Monte Carlo light equity estimator

## Modelos planejados
- opponent model
- value/policy distillation
- solver-centric blueprint generation
- session analytics e leak detection

## Regras de governança de dados
- separar histórico, benchmark e labels
- nunca misturar HU e 6-max sem tag explícita
- versionar features
- manter source-of-truth reproduzível
