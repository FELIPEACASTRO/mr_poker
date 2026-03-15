# Functional Specification

## Módulos
1. mesa local
2. engine de mão
3. baseline agent
4. replay
5. benchmark
6. experiment runner
7. spot packs
8. session runner
9. persistence
10. explanation/traces

## Casos de uso atuais
### UC-01 Criar uma mão
Usuário ou sistema cria uma nova mão heads-up com stacks, botão e seed.

### UC-02 Jogar manualmente
Usuário executa ações legais até fim da mão.

### UC-03 Deixar a IA agir
Sistema chama o baseline agent e devolve a ação, racional e trace.

### UC-04 Reproduzir uma mão
Sistema reconstroi a mão a partir de seed/log/snapshots persistidos.

### UC-05 Rodar benchmark smoke
Sistema roda agent-vs-agent e retorna resultado agregado.

### UC-06 Rodar benchmark H2H
Sistema roda múltiplos matches e gera relatório.

### UC-07 Instanciar spot pack
Sistema cria uma mão em um ponto específico da árvore de decisão para treino.

### UC-08 Rodar sessão local
Sistema roda várias mãos, persiste traces e retorna métricas de sessão.

## Regras funcionais
- somente heads-up no escopo atual
- blinds heads-up: botão posta small blind
- state machine deve bloquear ações ilegais
- replay deve ser determinístico
- toda decisão automática deve registrar trace quando disponível
- sessão deve persistir summary e traces

## Critérios de aceite Sprint 05
- listar packs
- instanciar spot packs válidos
- rodar sessão local persistida
- recuperar sessão e traces por API
- manter regressão anterior intacta
