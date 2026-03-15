# Prompt Backup Master

Use este documento para restaurar o contexto do projeto em qualquer ferramenta ou agente de IA.

## Missão do projeto
Construir uma solução local-first de Texas Hold'em No-Limit Heads-Up que atue como engine, agente, sparring e coach em ambiente próprio, com separação clara entre:
- motor de regras
- motor estratégico
- camada de explicação
- dados e analytics

## Restrições principais
- escopo atual: heads-up
- estrutura: no-limit
- ambiente: próprio/local-first
- nada de depender de LLM puro como policy final
- dados e benchmarks devem ser segregados por função
- produção depende de estabilidade e métricas estratégicas, não de “100% de precisão”

## Princípios técnicos
1. engine antes de solver
2. replay antes de escala
3. dados limpos antes de modelos pesados
4. decisão separada da explicação
5. benchmark e regressão em toda sprint
6. FinOps desde o local-first

## Arquitetura-alvo
- engine determinístico
- persistence canônica
- baseline agent explicável
- spot packs
- session analytics
- trilha futura solver-centric
- coaching layer separada

## Roadmap resumido
- Sprint 00: bootstrap
- Sprint 01: engine POC
- Sprint 02: persistência e replay
- Sprint 03: settlement e harness
- Sprint 04: baseline forte e eval runner
- Sprint 05: spot packs, sessões e traces
- Sprint 06+: taxonomy, opponent modeling, solver pipeline

## Regras da solução
- todo estado da mão deve ser reproduzível
- toda decisão automática deve ser auditável
- todo experimento relevante deve gerar relatório
- session evaluation deve distinguir realized result de EV proxy
- spot packs devem ser usados como ativos de treino/regressão
