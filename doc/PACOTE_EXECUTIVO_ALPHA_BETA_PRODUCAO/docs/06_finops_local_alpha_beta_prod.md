# Plano FinOps - Local -> Alpha -> Beta -> Produção

## 1. Objetivo
Garantir que a evolução do projeto preserve disciplina de custo desde o modo local-first até a operação em produção.

## 2. Princípios
1. **Todo experimento precisa de hipótese e budget.**
2. **Treino, benchmark e storage devem ter owner.**
3. **Artefatos temporários precisam de política de retenção.**
4. **Custo unitário deve ficar visível antes de produção.**
5. **Capacidade deve ser planejada, não presumida.**

## 3. Fase Local / Laboratório
### Controles
- limite de execuções paralelas
- budgets de experimento e benchmark
- retenção curta para relatórios temporários
- limpeza automática de diretórios de trabalho
- compressão de datasets oficiais
- uso preferencial de CPU antes de GPU quando a hipótese permitir

### KPIs locais
- custo/tempo por benchmark
- custo/tempo por treino
- tamanho de artefatos gerados por sprint
- consumo de storage de relatórios e modelos
- volume de runs descartadas

## 4. Fase Alpha
### Controles
- budgets fixos por semana
- limite de runs por owner
- classificação de artefatos em oficial vs temporário
- snapshots oficiais somente para candidatos relevantes
- política de retenção por diretório e por tipo

### KPIs Alpha
- custo por round-robin
- custo por candidate alpha
- custo por dataset oficial
- uso médio de CPU/RAM por job
- storage oficial vs temporário

## 5. Fase Beta
### Controles
- custo por ambiente
- custo por sessão
- custo por benchmark
- custo por release
- retenção por telemetria/log
- storage com lifecycle

### KPIs Beta
- custo médio por usuário piloto
- custo médio por sessão
- custo médio por relatório de coach
- custo por modelo promovido
- custo total do piloto fechado

## 6. Fase Produção
### Controles
- budget mensal por ambiente
- orçamento por capacidade
- custo por observabilidade
- custo por storage quente/frio
- custo por rollback/release
- planejamento de capacidade trimestral

### KPIs Produção
- custo por usuário ativo
- custo por sessão
- custo por benchmark contínuo
- custo por release
- custo por incidente
- custo total da plataforma

## 7. Regras executivas
- nenhum treino grande roda sem ticket/owner/hypótese
- nenhum artefato pesado sem política de expiração
- nenhum ambiente sem budget
- nenhum release sem leitura de custo prevista