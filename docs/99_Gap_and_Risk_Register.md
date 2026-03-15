# Gap and Risk Register

## Escopo

Este registro reconcilia o que o codigo realmente faz, o que os testes realmente garantem e o que a documentacao historica ou proxies locais podem sugerir de forma otimista.

## 1. Regra de leitura

- Conclusao tecnica so e considerada fechada quando codigo e testes convergem.
- Quando um indicador local contradiz auditorias ou runbooks, prevalece a interpretacao mais conservadora.
- Este documento nao altera comportamento do sistema; ele corrige a interpretacao do sistema.

## 2. Gaps confirmados

| ID | Gap | Evidencia | Impacto | Status |
| --- | --- | --- | --- | --- |
| `G-01` | Baseline documental estava stale frente ao tracked real | `git ls-files` atual retorna `321` e os dossies centrais foram reconciliados para o mesmo baseline | risco de inventario inconsistente | fechado |
| `G-02` | Ambiguidade de readiness entre ativos locais e operacao real | `ReadinessService` agora separa `asset_readiness` e `operational_readiness`, com `operational_blockers` objetivos e `production_candidate` deprecado | superestimacao de prontidao | fechado |
| `G-03` | `Makefile` assume `bash`, mas o ambiente do usuario e PowerShell/Windows | targets de validacao usam shell Unix | quebra de reproducao local para quem seguir somente `make` | parcialmente mitigado via AGENTS e dossier |
| `G-04` | `ruff` esta configurado, mas nao reproduzivel neste ambiente | ferramenta nao encontrada no `PATH` local | lint gate incompleto | aberto |
| `G-05` | Solver externo e apenas mock de arquivo | `var/external_solver/sample_solver_v1.json` + adapter local | risco de leitura excessivamente otimista da capacidade de solver | aberto |
| `G-06` | `decision_service`, `explanation_service` e `game_orchestrator` existem apenas como namespaces | apenas `__init__.py`, sem implementacao real | pode haver falsa impressao de features prontas | mitigado documentalmente |
| `G-07` | Workspace runtime produz artefatos nao versionados essenciais para parte da narrativa de IA | datasets, modelos, model cards, relatorios e DB ficam fora do Git | clean clone nao reproduz completamente o estado observado | aberto |
| `G-08` | Packaging existe, deploy real nao | `Dockerfile`, `docker-compose.yml` e manifest local presentes, sem stack de secrets/observability/runtime controls | transicao local -> Alpha/Prod ainda incompleta | aberto |
| `G-09` | Historico de sprint pode conflitar com estado atual | `docs/00-95` misturam especificacao, preview, snapshot e auditoria | risco de conclusao usando documento historico errado | mitigado com ordem de verdade |
| `G-10` | Readiness de testes e metrica de cobertura nao sao a mesma coisa | `40` testes passam, mas nem todo modulo tem cobertura direta identificavel | falsa sensacao de completude | aberto |

## 3. Riscos operacionais priorizados

### R-01. Promocao prematura para producao

- Causa: proxies locais de readiness, packaging e release note podem parecer equivalentes a maturidade operacional.
- Evidencia: `ReadinessService` e `DeployService` trabalham por presenca de arquivos e contagens locais.
- Consequencia: expectativa incorreta de SLA, seguranca, suporte e observabilidade.
- Mitigacao recomendada: manter o termo oficial como `Alpha/local lab` ate existir stack real de secrets, observability, incident response e rollout control.

### R-02. Reproducao desigual entre ambientes

- Causa: `Makefile` assume bash e `ruff` nao esta presente no ambiente observado.
- Consequencia: um desenvolvedor em Windows pode acreditar que a automacao oficial esta quebrada quando, na pratica, o problema e de shell/toolchain.
- Mitigacao recomendada:
  - documentar o caminho Python/Docker sem `make`
  - adicionar script cross-platform ou notas explicitas por shell
  - incluir check de dependencia para `ruff`

### R-03. Dependencia de artefatos fora do Git

- Causa: datasets, modelos, model cards, relatorios e DB local sao produzidos no workspace e nao fazem parte do baseline versionado.
- Consequencia: um clone limpo nao contem o mesmo estado de IA/dados visto durante a auditoria.
- Mitigacao recomendada:
  - manter exemplos canonicos minimos versionados
  - preservar manifests de referencia
  - diferenciar sempre "versionado" de "observado no workspace"

### R-04. Overclaim de integracao com solver

- Causa: endpoints e services de external solver existem, mas usam arquivo sample/mock.
- Consequencia: stakeholders podem inferir capacidade GTO ou comparacao com solver real.
- Mitigacao recomendada:
  - manter o nome oficial como `external solver adapter mock/local`
  - exigir catalogo real, timeout, validacao de schema e trilha de erro antes de promover o recurso

### R-05. Lacuna entre score local e governanca real

- Causa: model card, calibracao e release gate ja geram artefatos, mas sem policy externa, sign-off humano formal ou trilha de aprovacao.
- Consequencia: governanca pode ser interpretada como concluida quando ainda e apenas local.
- Mitigacao recomendada:
  - adicionar criterios de aprovacao humana e versionamento de decisao
  - ligar gate a suite de regressao e baseline de observabilidade real

## 4. Riscos documentais

| Risco | Onde aparece | Controle aplicado |
| --- | --- | --- |
| Snapshot historico tratado como especificacao vigente | sprint docs, roadmaps, previews | ordem de verdade e dossie `96` |
| Documento executivo tratado como contrato tecnico | `doc/PACOTE_EXECUTIVO_*` | separacao entre `doc/` e `docs/` nos `AGENTS.md` |
| Audit pass lido como prova de prod readiness | `88`, `93`, `FINAL_*` | gap register explicito e leitura conservadora |

## 5. O que foi mitigado por esta implementacao

- Ordem de verdade agora esta explicita em `AGENTS.md` e neste registro.
- O repositorio passou a ter camada permanente de onboarding tecnico por escopo.
- O mapa de ownership de services e endpoints esta documentado.
- O contraste entre baseline versionado e artefatos runtime observados foi registrado.
- O contrato de readiness foi separado entre sinal de ativos e sinal operacional.
- O gap entre readiness local e maturidade produtiva agora esta explicitado com blockers objetivos.

## 6. O que continua aberto apos esta implementacao

- Cross-platform automation real para Windows/PowerShell.
- Reproducao de lint com `ruff` garantida no ambiente.
- Integracao de solver externo real.
- Hardening operacional para Alpha/Beta/Producao.
- Estrategia de versionamento de datasets/modelos de referencia para clean clone.
- Criterios de cobertura mais fortes para services e trilhas de governanca.

## 7. Recomendacoes imediatas

1. Tratar o estado oficial do produto como `local-first / Alpha candidate only`.
2. Priorizar um script de validacao cross-platform ou documentacao shell-especifica mais forte.
3. Versionar exemplos minimos de dataset/modelo/model card ou gerar fixtures deterministicas.
4. Manter qualquer narrativa de solver sob o rotulo `mock/local adapter`.
5. Evoluir de blockers operacionais para controles operacionais reais (observabilidade, secrets, rollout, incident response).
