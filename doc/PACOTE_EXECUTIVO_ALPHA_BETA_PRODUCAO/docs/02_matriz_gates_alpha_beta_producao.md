# Matriz de gates - Alpha / Beta / Produção

## Leitura da matriz
Cada gate abaixo deve ser tratado como uma condição de promoção. A passagem de fase exige:
- todos os gates críticos atendidos
- nenhum blocker vermelho aberto
- owner designado para cada desvio residual

## Matriz executiva

| Dimensão | Gate Alpha | Gate Beta | Gate Produção |
|---|---|---|---|
| Escopo | Heads-up, local-first, fluxo fim a fim funcional | Piloto fechado com usuários controlados | Produto operacional com ambientes formais |
| Regras/Engine | Regras principais e replay determinístico aprovados | Sem regressão de engine em piloto | Hardening operacional da engine |
| Dados | Dataset oficial v1 congelado | Coleta incremental confiável | Registry/governança reforçados |
| Modelo | Candidate alpha selecionado e avaliado | Candidate beta estável e superior ao alpha | Candidate production aprovado |
| Avaliação | Benchmark oficial e score por taxonomy | Monitoramento contínuo e comparativos por versão | Política formal de promoção/rebaixamento |
| Coach | Explicações coerentes com a decisão | Utilidade percebida por usuários piloto | Camada estável com governança e rastreabilidade |
| Operação | Packaging e release gate funcionais | Deploy repetível e rollback básico | Observabilidade, segurança e operação plenas |
| FinOps | Budget local e retenção controlada | Custo por ambiente e por sessão monitorados | FinOps contínuo com custo unitário e capacidade |
| Segurança | Mínimo suficiente para ambiente controlado | Auth básica e segredos por ambiente | Hardening, secret management e trilha de auditoria |
| Produto | Fluxo interno utilizável | UX aprovada em piloto fechado | UX e suporte adequados para produção |

## Critérios formais por fase

### Gate Alpha
A fase Alpha só deve ser aprovada quando:
- engine e replay estiverem estáveis
- dataset oficial v1 existir
- tournament round-robin oficial existir
- candidate alpha superar baseline de forma consistente
- regressão dourada estiver limpa
- model card existir
- release gate alpha estiver operacional
- budgets locais estiverem definidos

### Gate Beta
A fase Beta só deve ser aprovada quando:
- piloto fechado rodar sem suporte intensivo
- candidate beta superar candidate alpha de forma consistente
- UX mínima estiver aprovada
- observabilidade básica estiver operacional
- rollback estiver validado
- custo por ambiente estiver visível
- incidentes críticos estiverem controlados

### Gate Produção
A produção só deve ser aprovada quando:
- candidate production passar em todos os gates críticos
- operação multiambiente estiver pronta
- segurança mínima estiver implementada
- monitoramento e alertas estiverem ativos
- governança de dados/modelos estiver formalizada
- runbooks estiverem testados
- FinOps contínuo estiver operacional