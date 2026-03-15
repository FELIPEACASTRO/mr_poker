# FinOps Local Budget

## Princípio
No início, o custo principal é tempo de máquina, armazenamento e retrabalho.

## Orçamentos locais sugeridos
### Desenvolvimento diário
- CPU: moderado
- RAM: sem treino pesado simultâneo
- Disco: retenção curta de artefatos locais

### Experimentos
- toda run deve registrar objetivo
- toda run deve registrar orçamento de tempo
- toda run deve registrar limite de armazenamento
- checkpoints temporários devem ser podados

## Trigger para rever arquitetura local
- máquina degradando durante desenvolvimento normal
- storage de runs crescendo sem controle
- benchmark demorando a ponto de bloquear iteração
