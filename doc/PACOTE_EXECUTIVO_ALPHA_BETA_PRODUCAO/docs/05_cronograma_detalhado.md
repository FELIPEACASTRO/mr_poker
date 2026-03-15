# Cronograma detalhado de execução

## Premissas
- Times paralelos podem reduzir duração.
- O cronograma abaixo assume execução disciplinada, sem expansão indevida de escopo.
- Janela total sugerida: **18 semanas**, partindo do fim do laboratório até produção inicial controlada.

## Fase 1 - Alpha Hardening (Semanas 1 a 6)

### Semana 1
- congelar escopo Alpha
- aprovar gates Alpha
- nomear owners por workstream
- congelar premissas de dados e taxonomy

### Semana 2
- congelar dataset oficial v1
- emitir manifesto do dataset
- revisar qualidade das labels
- publicar taxonomy oficial v1

### Semana 3
- integrar solver externo real (primeira versão)
- comparar solver real vs solver-like
- identificar lacunas de labels por taxonomy

### Semana 4
- treinar/selecionar candidate alpha
- rodar round-robin oficial
- consolidar score por taxonomy
- revisar hand review e coach report

### Semana 5
- emitir model card do candidate alpha
- rodar release gate alpha
- validar regression suite
- ajustar packaging

### Semana 6
- aprovar candidate alpha
- publicar readiness snapshot alpha
- congelar baseline, adaptive e candidate
- encerrar fase Alpha

## Fase 2 - Beta Enablement (Semanas 7 a 12)

### Semana 7
- refinar UX de replay, analytics e coach
- definir fluxo de onboarding
- planejar piloto fechado

### Semana 8
- implementar auth básica
- separar ambientes
- ativar observabilidade básica
- validar backups

### Semana 9
- preparar pipeline de release
- validar rollback
- reforçar coleta de uso
- ativar changelog/release notes operacionais

### Semana 10
- iniciar piloto fechado
- coletar feedback qualitativo
- medir custo por sessão e por benchmark

### Semana 11
- corrigir fricções de UX
- ajustar candidate beta
- consolidar avaliação contínua
- revisar incidentes do piloto

### Semana 12
- emitir gate beta
- aprovar ou reprovar promoção para Beta Hardening
- congelar plano de produção

## Fase 3 - Beta Hardening (Semanas 13 a 16)

### Semana 13
- estabilizar operação
- revisar incidentes e gargalos
- reforçar governança de dados/modelos

### Semana 14
- fortalecer CI/CD
- fortalecer rollback
- elevar cobertura de observabilidade
- revisar alertas

### Semana 15
- aprovar candidate production
- emitir release gate de produção
- validar runbooks

### Semana 16
- congelar janela de lançamento
- checklist final de segurança
- checklist final FinOps
- aprovação executiva para produção inicial

## Fase 4 - Produção Inicial (Semanas 17 a 18)

### Semana 17
- deploy controlado em staging final
- smoke tests operacionais
- liberar produção inicial para público controlado

### Semana 18
- acompanhar estabilidade
- monitorar custo, uso e incidentes
- consolidar relatório pós-lançamento
- definir backlog da versão seguinte