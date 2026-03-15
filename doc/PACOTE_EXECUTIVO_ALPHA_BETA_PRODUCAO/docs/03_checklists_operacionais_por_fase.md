# Checklists operacionais por fase

## 1. Checklist - Alpha controlado

### Produto
- [ ] Fluxo de sessão fim a fim validado
- [ ] Replay navegável e auditável
- [ ] Coach report disponível por sessão
- [ ] Hand review coerente com a decisão do core
- [ ] Export canônico operacional

### Engenharia
- [ ] Build local reproduzível
- [ ] Testes principais automatizados
- [ ] Regression suite executando
- [ ] Packaging local validado
- [ ] Artefatos versionados

### Dados
- [ ] Dataset oficial v1 congelado
- [ ] Splits oficiais publicados
- [ ] Manifesto do dataset pronto
- [ ] Taxonomy oficial v1 aprovada
- [ ] Linhagem de dados documentada

### IA / Avaliação
- [ ] Candidate alpha definido
- [ ] Baseline oficial definido
- [ ] Adaptive baseline oficial definido
- [ ] Benchmark round-robin rodado
- [ ] Score por taxonomy consolidado
- [ ] Model card emitido
- [ ] Release gate alpha emitido

### FinOps
- [ ] Budget por experimento definido
- [ ] Budget por benchmark definido
- [ ] Política de retenção local definida
- [ ] Política de limpeza de artefatos aplicada
- [ ] Uso de CPU/RAM controlado por job

## 2. Checklist - Beta fechado

### Produto
- [ ] Onboarding mínimo aprovado
- [ ] Modos de uso claros
- [ ] UX de replay aprovada
- [ ] UX de analytics aprovada
- [ ] UX do coach aprovada

### Engenharia
- [ ] Auth básica funcional
- [ ] Configuração por ambiente funcional
- [ ] CI/CD básico operacional
- [ ] Backups testados
- [ ] Rollback validado

### Dados / IA
- [ ] Coleta de uso do beta funcional
- [ ] Candidate beta aprovado
- [ ] Comparativo alpha vs beta documentado
- [ ] Avaliação contínua funcional
- [ ] Drift e falhas por taxonomy rastreados

### Operação
- [ ] Alertas básicos ativos
- [ ] Logs estruturados disponíveis
- [ ] Tratamento de incidentes definido
- [ ] Changelog operacionalizado
- [ ] Release notes por versão disponíveis

### FinOps
- [ ] Custo por sessão disponível
- [ ] Custo por benchmark disponível
- [ ] Custo por ambiente disponível
- [ ] Storage com política de retenção
- [ ] Budget mensal do beta publicado

## 3. Checklist - Produção inicial

### Plataforma
- [ ] Staging e produção existentes
- [ ] Secret management implementado
- [ ] Dashboards ativos
- [ ] Alertas e health checks ativos
- [ ] Runbooks testados

### IA / MLOps
- [ ] Model registry formal
- [ ] Política de promoção de modelo
- [ ] Dataset registry formal
- [ ] Gate de produção automatizado/parcial
- [ ] Avaliação contínua em produção
- [ ] Rebaixamento/rollback de modelo documentado

### Segurança
- [ ] Auth robusta
- [ ] Autorização mínima adequada
- [ ] Logging seguro
- [ ] Segredos protegidos
- [ ] Auditoria básica disponível

### FinOps
- [ ] Custo por usuário
- [ ] Custo por sessão
- [ ] Custo por release
- [ ] Custo por storage
- [ ] Custo por observabilidade
- [ ] Política de arquivamento
- [ ] Planejamento de capacidade

### Produto / Operação
- [ ] Suporte mínimo definido
- [ ] SLA/SLO internos definidos
- [ ] Plano de incidentes aprovado
- [ ] Política de release aprovada
- [ ] Produção inicial autorizada