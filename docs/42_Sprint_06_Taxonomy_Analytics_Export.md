# Sprint 06 — Taxonomy de Spots, Session Analytics e Export PHH-like

## Objetivo
Transformar o projeto de **local-eval-ready** em **analytics-ready + export-ready**, preparando a base para a trilha solver-centric sem quebrar o MVP local-first.

## O que esta sprint entrega
- taxonomy de spots em nível de trace e mão
- analytics ricos por sessão
- vínculo `session_id -> hands`
- export canônico **PHH-like** por mão e por sessão
- novos endpoints de taxonomia, analytics e export

## Racional técnico
Até a Sprint 05 o projeto já era capaz de:
- jogar localmente
- persistir mãos e traces
- rodar benchmarks H2H
- gerar relatórios de experimento

Faltavam três blocos importantes:
1. **linguagem comum de spots** para auditoria e backlog solver-centric
2. **analytics de sessão mais ricos** para análise diagnóstica
3. **formato de export canônico** para versionamento, replay compartilhável e preparo de datasets

## Novos endpoints
- `GET /v1/spots/taxonomy`
- `GET /v1/hands/{hand_id}/taxonomy`
- `GET /v1/hands/{hand_id}/export/phh-like`
- `GET /v1/sessions/{session_id}/analytics`
- `GET /v1/sessions/{session_id}/export/phh-like`

## Saídas principais
### Taxonomy
- `open`
- `defend_blind`
- `3bet_or_jam`
- `cbet`
- `barrel`
- `bluffcatch`
- `thin_value`
- `single_raised_pot`
- `showdown_reached`
- `all_in_hand`

### Session analytics
- distribuição por street
- distribuição por action type
- board textures
- top hole classes
- taxonomy counts
- média de edge por street
- média de edge por seat
- taxonomia por mão

### Export PHH-like
- metadados da mão
- stacks iniciais
- blinds
- board final
- ações ordenadas
- traces de decisão
- winner seat

## Critérios de aceite da sprint
- todos os testes existentes continuam verdes
- novos testes cobrem taxonomia, analytics e export
- hands de sessão ficam vinculadas ao `session_id`
- export retorna texto canônico consistente

## Resultado
Status do projeto após esta sprint:
**analytics-ready / export-ready**
