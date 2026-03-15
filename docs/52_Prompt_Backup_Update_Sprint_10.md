# Prompt backup — atualização até Sprint 10

## Decisões congeladas
- local-first
- Texas Hold'em NL Heads-Up
- engine formal separado da linguagem
- baseline explicável antes do solver real
- PHH-like interno antes de PHH estrito

## Regras de implementação
- qualquer melhoria deve preservar replay determinístico
- novos agentes devem emitir traces auditáveis
- datasets de treino devem separar função e origem
- produção não depende de "100% de precisão"; depende de métricas corretas

## Norte da solução
- sistema multi-camada: engine + crenças + estratégia + profiling + coach
