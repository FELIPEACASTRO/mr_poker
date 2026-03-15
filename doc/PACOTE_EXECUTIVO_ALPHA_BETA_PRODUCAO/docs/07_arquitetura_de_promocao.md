# Arquitetura de promoção - do laboratório até produção

## 1. Estado de origem
O laboratório já entrega:
- engine
- replay
- analytics
- datasets
- avaliação
- packaging inicial
- governança básica

## 2. Camadas que precisam evoluir para Alpha/Beta/Produção

### Camada de produto
Laboratório: uso interno e técnico  
Alpha: uso controlado e fim a fim  
Beta: piloto com usuários reais selecionados  
Produção: produto operacional

### Camada de dados
Laboratório: datasets locais e builder funcional  
Alpha: dataset oficial v1  
Beta: ingestão incremental e avaliação contínua  
Produção: registry formal e governança reforçada

### Camada de modelo
Laboratório: baseline/adaptive/policy local  
Alpha: candidate alpha e comparativos oficiais  
Beta: candidate beta com promoção controlada  
Produção: candidate production com política de rollback

### Camada de operação
Laboratório: packaging e scripts  
Alpha: packaging aprovado e release gate  
Beta: CI/CD, rollback e observabilidade básica  
Produção: multiambiente, runbooks, operação formal

### Camada de custo
Laboratório: budget local  
Alpha: budget por benchmark/candidato  
Beta: custo por ambiente/sessão  
Produção: custo unitário e capacidade

## 3. Linha de promoção correta
Laboratório concluído -> Alpha controlado -> Beta fechado -> Produção inicial

## 4. Regra de ouro
**Nenhuma fase deve ser promovida por pressão de calendário.**
Promoção deve ocorrer apenas por:
- gates aprovados
- blockers resolvidos
- ownership definido
- leitura de custo aceitável