# FinOps Master Plan — Local First até Produção

## Premissa
O projeto começa **na máquina local**. Produção só entra quando a solução estiver madura o suficiente em correção, estabilidade e qualidade estratégica.

## Objetivos FinOps
- limitar desperdício local
- tornar custo de experimento observável
- evitar escalar cedo demais
- preparar transição controlada para cloud/produção

## Fase atual: local-only
### Custos dominantes
- tempo de desenvolvimento
- CPU local
- RAM local
- armazenamento de relatórios/logs
- eventual uso de GPU local

### Controles
- budgets por sprint
- limites de retenção de logs
- relatórios compactos por experimento
- evitar checkpoints gigantes sem necessidade
- separar runs exploratórios de runs “oficiais”

## Métricas FinOps locais
- tempo médio por benchmark
- tempo médio por sessão
- tamanho médio por relatório
- crescimento do SQLite
- tempo de teste por sprint
- custo de oportunidade por experimento inválido

## Gate para cloud
Ir para produção apenas quando:
- engine estiver estabilizado
- baseline/sessão/traces estiverem maduros
- trilha solver-centric estiver validada em laboratório
- observabilidade mínima estiver pronta
- custo por sessão estiver modelado

## Estratégia de custo futuro
- separar treino de inferência
- usar storage barato para relatórios frios
- manter inferência com orçamento de latência
- só subir componentes que precisam escalar
