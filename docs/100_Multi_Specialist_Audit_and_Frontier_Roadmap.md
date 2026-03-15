# Auditoria Multi-Specialista e Frontier Roadmap

## Escopo

Este documento responde a uma pergunta objetiva: o `mr_poker` atual esta no caminho para virar uma solucao dominante de poker ou ainda e um laboratorio Alpha/local com boa base, mas longe da fronteira competitiva?

A resposta curta e:

- como laboratorio de engenharia local-first: sim, a base e boa
- como stack de IA competitiva contra estado da arte de poker: nao
- como candidato real a "TOP 1 do poker": ainda nao, e por margem larga

## Base de auditoria

Foram conciliados:

- codigo versionado (`git ls-files`)
- runtime local observado em `var/`
- suite atual de testes passando com `pytest -q`
- schema SQLite, artefatos JSON e contratos FastAPI
- documentacao do repositorio
- literatura e frameworks externos de referencia para poker e jogos de informacao imperfeita

Fontes externas-chave usadas para orientar o roadmap frontier:

- ReBeL: <https://arxiv.org/abs/2007.13544>
- OpenSpiel: <https://arxiv.org/abs/1908.09453>
- RLlib multi-agent envs: <https://docs.ray.io/en/master/rllib/multi-agent-envs.html>
- PokerBench: <https://arxiv.org/abs/2501.08328>
- AIVAT: <https://arxiv.org/abs/1612.06915>
- RL-CFR: <https://raw.githubusercontent.com/mlresearch/v235/main/assets/li24t/li24t.pdf>
- IIG-RL-Benchmark: <https://openreview.net/pdf/ac4c418dd8da83accdc310fea73f182fb8c48445.pdf>
- cfrx (CFR acelerado em JAX): <https://github.com/Egiob/cfrx>
- PokerRL: <https://github.com/EricSteinberger/PokerRL>
- RLCard: <https://arxiv.org/abs/1910.04376>
- Pluribus / CMU: <https://www.cmu.edu/news/stories/archives/2019/july/cmu-facebook-ai-beats-poker-pros.html>

## Parecer executivo

### Veredito geral

O `mr_poker` atual e uma boa fundacao de laboratorio local para:

- engine deterministico e reproduzivel
- experimento H2H local
- persistencia de hands/traces
- dataset builder
- treino de policy table
- avaliacao offline
- calibracao e governanca basica

Mas ele nao e, hoje, uma pilha competitiva de poker de fronteira.

O motor de decisao atual ainda esta centrado em:

- heuristica baseline
- adaptacao estatistica simples
- rotulacao solver-like proxy
- policy table discreta

Isso basta para um Alpha tecnico e para uma plataforma de pesquisa local. Nao basta para competir com stacks da classe DeepStack, Libratus, Pluribus, ReBeL, Deep CFR, DCFR/PDCFR, ligas auto-jogadas modernas ou sistemas com medicao formal de exploitability.

### Scorecard de maturidade

| Especialista | Nota | Parecer |
| --- | ---: | --- |
| Engenheiro de engine/rules | `8/10` | base forte e confiavel para laboratorio HUNL |
| Arquiteto de sistema | `7/10` | modular monolith coerente para fase atual |
| Cientista de poker/game theory | `3/10` | falta o nucleo solver/search/exploitability de fronteira |
| Cientista de RL/ML | `4/10` | trilha de treino ainda e shallow e majoritariamente tabular |
| MLOps/Data platform | `4/10` | manifestos existem, mas lineage e reproducao ainda sao locais/manualizados |
| SRE/Platform/DevOps | `4/10` | packaging local bom; readiness real ainda insuficiente |
| Security/Integrity/Anti-cheat | `3/10` | ausente para qualquer contexto competitivo serio |
| Product/Coach/Study UX | `6/10` | boa oportunidade diferencial, mas ainda inicial |

## Triple Check v2 (status de fechamento)

| Item | Tipo | Status | Evidencia objetiva |
| --- | --- | --- | --- |
| Baseline canonico e inventario reconciliados | P0 | fechado | dossies `96/97/99/100` sincronizados para o baseline tracked atual e `docs/97` cobrindo os caminhos faltantes |
| Cobertura de endpoints no mapa de integracao | P0 | fechado | `docs/98` inclui o endpoint `GET /v1/hands/{hand_id}/review` e appendice canonico metodo+path+owner |
| Readiness sem ambiguidade | P0 | fechado | `ReadinessService.snapshot()` agora separa `asset_readiness` e `operational_readiness`, com `operational_blockers` e `deprecations` |
| Guardrail automatizado de auditoria | P1 | fechado | script `infra/scripts/audit_consistency_check.py` + teste unitario de execucao strict |
| Validacao automatizada de regressao | P1 | fechado | `pytest -q` completo apos as mudancas: `42 passed` |
| Refresh do parecer frontier com estado objetivo | P1 | fechado | este documento agora reporta fechamento P0/P1 e mantem roadmap frontier com leitura conservadora |

## Parecer por especialista

### 1. Principal Architect

#### O que esta bom

- `apps/api/main.py` funciona como composition root claro.
- O repo esta organizado como modular monolith: `packages/*` para logica, `services/*` para ownership funcional, `apps/*` para exposicao.
- A escolha atual de nao fragmentar cedo em microservices foi correta.

#### Gaps

- ha services placeholder sem implementacao real (`decision_service`, `explanation_service`, `game_orchestrator`)
- `ReadinessService` e `DeployService` ainda operam mais como proxies documentais do que como operacao real
- a trilha de treino, avaliacao e batch ainda esta acoplada a filesystem local, sem fila, sem workers dedicados e sem isolamento forte

#### Parecer

Arquiteturalmente, a decisao certa agora nao e migrar para microservices. A decisao certa e endurecer o monolito modular e separar os loops pesados de pesquisa/treino em workers dedicados.

#### Recomendacoes

1. Manter a API + services como monolito modular.
2. Extrair `training/eval/batch` para workers assicronos com fila.
3. Transformar `readiness` em dois produtos distintos:
   - `asset_readiness`
   - `operational_readiness`
4. Eliminar placeholders ou promovelos para implementacao real.

### 2. Poker Scientist / Game Theorist

#### O que esta bom

- engine deterministico com `seed` e `deck_prefix`
- side pots, showdown, refund e replay estao presentes
- spot normalization, taxonomy e benchmark local criam base util para pesquisa

#### O que falta e e critico

- exploitability / approximate best response
- CFR+, DCFR, MCCFR, PDCFR ou equivalente forte
- subgame solving / re-solving
- public belief state search
- action abstraction research de nivel competitivo
- avaliacao estatistica de alta qualidade para poucas hands

#### Parecer

Sem medir exploitability e sem uma familia forte de algoritmos de regret minimization/search, o projeto nao consegue afirmar qualidade teorica do jogo. Hoje ele mede utilidade local, nao forca GTO de fronteira.

#### Recomendacoes

1. Integrar OpenSpiel como baseline formal de jogos e algoritmos.
2. Implementar uma trilha solver interna com:
   - CFR+
   - DCFR/PDCFR
   - MCCFR
3. Adotar AIVAT para reduzir variancia de avaliacao.
4. Criar harness de exploitability e best response.
5. Construir uma camada de subgame solving para cenarios HUNL reduzidos.

### 3. Reinforcement Learning Scientist

#### O que esta bom

- baseline e adaptive sao suficientes como professores fracos e para gerar dados bootstrap
- policy table funciona como baseline barata e reproduzivel
- dataset builder + evaluation + calibration formam um loop util

#### O que esta limitando a evolucao

- o sistema ainda nao treina politicas profundas fortes
- nao existe self-play league real
- nao existe treinamento neural de regret/value/policy em escala
- nao existe modelo de crenca publica/privada
- nao existe distillation teacher -> student em varios niveis

#### Parecer

O projeto esta no degrau "bootstrapped local policy stack". A fronteira esta em "self-play + search + regret minimization + distillation + belief-state modeling".

#### Recomendacoes

1. Fasear a evolucao assim:
   - Fase A: solver baselines classicos
   - Fase B: Deep CFR / SD-CFR / RL-CFR
   - Fase C: ReBeL-like public belief search
   - Fase D: populacao de oponentes + exploiter
2. Usar RLlib apenas como orquestrador de multi-agent training, nao como substituto de teoria de poker.
3. Adotar JAX/PyTorch 2 compile onde fizer sentido:
   - JAX/cfrx para solver/regret loops acelerados
   - PyTorch para student models, sequence models e serving

### 4. Data and MLOps Specialist

#### O que esta bom

- manifests de datasets existem
- modelos sao serializados com metadata
- relatorios de avaliacao e model cards existem

#### Gaps

- sem versionamento formal de datasets
- sem lineage forte entre source rows, split, model, calibration e gate
- sem registry real de experimentos
- JSONL/JSON dominam, mas falta um backbone analitico mais eficiente

#### Parecer

O projeto tem rastreabilidade local suficiente para Alpha. Nao tem ainda stack robusta de reproducao cientifica.

#### Recomendacoes

1. Introduzir DVC ou LakeFS para versionamento de datasets e modelos.
2. Introduzir MLflow ou W&B para experiment tracking.
3. Mover datasets para Parquet/Arrow e indexacao analitica com DuckDB/Polars.
4. Criar IDs canonicos para:
   - hand
   - trace
   - dataset row
   - experiment run
   - model artifact
   - release gate decision

### 5. Platform / SRE Specialist

#### O que esta bom

- Dockerfile e docker compose existem
- healthcheck existe
- testes passam

#### Gaps

- `ruff` nao esta reproduzivel no ambiente
- `Makefile` assume bash em ambiente PowerShell
- readiness local esta otimista demais
- nao ha observabilidade real
- nao ha stack de secrets, rollout, incident response, nem controle operacional de longa duracao

#### Parecer

O produto esta pronto para laboratorio local disciplinado. Nao esta pronto para operacao competitiva seria.

#### Recomendacoes

1. Corrigir imediatamente a automacao cross-platform.
2. Introduzir CI matrix Windows/Linux.
3. Adicionar telemetry real:
   - logs estruturados
   - traces
   - metricas de treino/inferencia
   - dashboards
4. Separar imagens CPU e GPU.
5. Criar pipeline reproducivel de build de artefato de pesquisa.

### 6. Security / Integrity / Anti-Cheat Specialist

#### Parecer duro

Se a ambicao inclui poker competitivo serio, estudo em escala, coach remoto ou qualquer forma de confronto online, a camada atual de integridade e insuficiente.

#### Gaps

- sem hardening de secrets
- sem protecao forte de RNG/seed em contexto remoto
- sem trilha anti-collusion
- sem analise de timing patterns
- sem deteccao de bot-farms ou policy cloning
- sem attestation ou assinatura de hand histories

#### Recomendacoes

1. Para qualquer modo online, assinar hand histories e seeds.
2. Criar trilha de deteccao de anomalias de decisao/timing.
3. Instrumentar embeddings comportamentais por jogador.
4. Separar estritamente stack de estudo vs stack competitiva.

### 7. Product / UX / Coach Specialist

#### O que esta bom

- o repositorio ja aponta para um diferencial forte em `coach_service`, `curriculum_service` e `opponent_profile_service`

#### Oportunidade

LLMs nao devem ser o cerebro central da jogada. Eles podem ser uma interface excepcional para:

- coaching personalizado
- explicacao de leaks
- consulta em linguagem natural sobre o proprio banco de hands
- geracao de planos de estudo
- simulacao de linhas e contra-linhas

#### Parecer

O melhor caminho de inovacao produto nao e "um LLM que joga poker sozinho". E "um stack hibrido com engine/solver forte + coach inteligente + estudo personalizado + exploração adaptativa controlada".

### 8. Tech Strategy Specialist

#### Erro estrategico a evitar

Tentar pular de baseline heuristico + tabela para "top 1" usando apenas modelos gerais ou fine-tunes leves.

Poker de fronteira continua exigindo:

- teoria de jogos
- search
- regret minimization
- avaliacao rigorosa
- treinamento populacional

LLMs puros nao substituem isso. O proprio PokerBench e um sinal claro de que LLM reasoning generalista ainda nao resolve poker competitivo de forma confiavel.

## Gaps prioritarios

### P0

1. Exploitability nao existe.
2. Nao existe solver real competitivo.
3. Nao existe benchmark frontier padronizado.
4. Readiness operacional esta superestimada.
5. Automacao local ainda nao e plenamente reproduzivel.

### P1

1. Data lineage e experiment tracking insuficientes.
2. Ausencia de self-play league real.
3. Ausencia de distillation multi-estagio.
4. Ausencia de variancia reduzida na avaliacao.

### P2

1. Sem anti-cheat / integrity stack.
2. Sem serving de baixa latencia para politicas profundas.
3. Sem pesquisa de multiplayer frontier.

## Status objetivo das recomendacoes frontier

| Recomendacao frontier | Status atual |
| --- | --- |
| benchmark formal + exploitability | aberto |
| solver/search/regret minimization competitivo | aberto |
| self-play league e distillation multi-estagio | aberto |
| data lineage + experiment tracking de nivel de pesquisa | aberto |
| stack de integridade/anti-cheat para contexto competitivo | aberto |

## Proposta de arquitetura frontier

### Camada 1. Core game platform

- engine atual endurecido
- validadores formais de estados
- replay deterministico
- exports versionados
- harness formal de benchmark

### Camada 2. Teacher stack

- CFR+ / DCFR / MCCFR / RL-CFR
- subgame solving
- AIVAT
- exploitability + best response
- OpenSpiel interoperability

### Camada 3. Student stack

- modelos sequenciais por infoset/history
- heads separados para:
  - action policy
  - value
  - uncertainty/calibration
  - opponent embedding
- distillation teacher -> student

### Camada 4. Exploiter stack

- opponent clustering
- bandit/meta-policy para mix GTO vs exploit
- detection de regime de oponente
- adaptation com guardrails de robustez

### Camada 5. Coach and study stack

- leak detection
- coach em linguagem natural
- curriculum automatico
- query over hand DB

## Roadmap realista para chegar perto da fronteira

### Horizonte 0-60 dias

- corrigir reproducao cross-platform
- travar stack de lint/test/build
- introduzir experiment tracking
- introduzir Parquet + DuckDB/Polars
- medir exploitability em jogos menores e construir harness formal

### Horizonte 60-120 dias

- integrar OpenSpiel / PokerRL / RLCard como ambientes auxiliares
- implementar CFR+/DCFR/MCCFR baselines
- introduzir AIVAT no pipeline de avaliacao
- criar benchmark interno fixo com populacoes e seeds

### Horizonte 120-240 dias

- treinar student neural forte por distillation
- criar self-play league
- introduzir opponent embeddings
- construir serving de politica de baixa latencia

### Horizonte 240-360 dias

- experimentar RL-CFR e variantes profundas
- prototipar public-belief search inspirado em ReBeL
- introduzir exploiter robusto com fallback GTO-like
- construir benchmark publico/reprodutivel do proprio projeto

## A melhor aposta de diferenciacao

Se o objetivo e "ser o TOP 1", ha tres rotas. O projeto deve escolher uma principal:

### Rota A. Top 1 research engine

Foco em HUNL forte, exploitability minima, solver/search de fronteira.

### Rota B. Top 1 exploitative training platform

Foco em estudar populacoes reais, perfilacao, leaks, adaptation e EV pratico contra field.

### Rota C. Top 1 coach autonomo de poker

Foco em aprendizado, explicacao, analise de historico e planos personalizados.

Minha recomendacao:

- construir A como nucleo cientifico
- monetizar ou diferenciar com B + C

## Conclusao final

O `mr_poker` atual nao esta perto do topo do poker computacional. Mas ele tem algo valioso: uma base local coerente, reproduzivel e com bom particionamento entre engine, services, dados e artefatos.

O caminho certo nao e tentar "turbinhar" a policy table atual com mais heuristica ou com um LLM generalista.

O caminho certo e:

1. formalizar benchmark e exploitability
2. adicionar solver/search/regret minimization de verdade
3. criar liga de self-play e distillation neural
4. usar LLMs apenas onde eles realmente agregam: coaching, explicacao, planejamento de estudo e interface

Se essa sequencia for seguida com disciplina, o projeto pode deixar de ser apenas um Alpha local bem documentado e virar uma plataforma realmente seria de poker AI.
