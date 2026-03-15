# Test Strategy

## Camadas
- unit: deck, modelos, features mínimas
- integration: API local e orquestração básica
- regression: mãos conhecidas e edge cases
- replay: reconstrução idêntica da mão

## Prioridade
1. engine
2. replay
3. contracts
4. baseline agent
5. UI

## Métrica mínima nesta fase
- testes de smoke sempre verdes
- cobertura crescente do engine a partir da Sprint 01
