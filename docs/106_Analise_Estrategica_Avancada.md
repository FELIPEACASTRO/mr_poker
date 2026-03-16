# ANALISE ESTRATEGICA AVANCADA: Como os Avancos em IA para Poker Impulsionam o MR_POKER

**Data:** 2026-03-16
**Versao:** 1.0
**Autor:** Analista Senior de Estrategia de IA — Game Theory, RL & Imperfect Information Domains
**Base:** 385+ recursos verificados em 12 paises, 76 papers poker-especificos, 47 datasets, 5 documentos de pesquisa

---

## NOSSA SOLUCAO: MR_POKER

O MR_POKER e uma plataforma local-first de IA para Texas Hold'em No-Limit Heads-Up (HUNL) que combina treinamento, avaliacao e analytics avancados. Escrita em Python 3.13 sobre Clean Architecture, a plataforma opera como laboratorio de pesquisa e ferramenta de treinamento para jogadores.

### Stack Tecnica Atual

| Componente | Tecnologia | Status |
|-----------|-----------|--------|
| **Engine** | Deterministico, seedable, showdown completo | Producao (8/10) |
| **CFR Suite** | 17+ variantes: Vanilla, DCFR, MCCFR, Deep CFR, VAD-CFR, NeuPL, NFSP, QRE, HDCFR, Compact, Lazy, GPU, Embedding CFR, ODCFR, Robust Deep MCCFR | Alpha |
| **Opponent Modeling** | 15+ modulos: tilt, timing tells, sizing tells, fatigue, particle filter, Bayesian range, MBOM, DDM, style embeddings, AMP3 | Alpha |
| **Behavioral Pipeline** | Orquestracao de 8+ modulos comportamentais integrados | Alpha |
| **Skill Estimation** | CNN-LSTM (Conv1D → BiLSTM → FC) com estimacao move-by-move | Alpha |
| **Synthetic Players** | PGM hierarquico, Big Five traits, 100+ atributos por persona | Alpha |
| **Cognitive Bias Exploiter** | Prospect theory, loss aversion, framing effects, tilt exploitation | Alpha |
| **RL Agents** | PPO, DMC (Distributional RL), MCTS | Alpha |
| **Decision-Time Planning** | Mirror Descent (Update-Equivalence), NfgTransformer | Alpha |
| **Evaluation** | PokerBench integrado, round-robin tournaments, exploitability calculator | Alpha |
| **Persistencia** | SQLite, PHH-like export, decision traces com Pydantic | Producao |
| **API** | FastAPI + Uvicorn, CQRS/SAGA | Alpha |
| **Testes** | 1450+ testes passando, pytest + mypy strict | Solido |
| **Infra** | Docker + Docker Compose, GitHub Actions CI | Alpha |

### Publico-Alvo
Jogadores de poker (low a high stakes) que desejam evoluir atraves de analise GTO com feedback personalizado baseado em leaks identificados, sem depender de coaches caros. Secundariamente: pesquisadores de IA em jogos de informacao imperfeita.

### Diferenciais Atuais
1. **17+ variantes CFR** em uma unica plataforma (nenhum concorrente oferece essa amplitude)
2. **Pipeline comportamental integrado** (tilt + timing + sizing + fatigue + meta-game em cadeia)
3. **Local-first** (privacidade total, sem dependencia de cloud)
4. **1450+ testes** (rigor de engenharia superior)
5. **Arquitetura modular** (Clean Architecture com CQRS/SAGA)

---

## 1. MAPEAMENTO DE OPORTUNIDADES

### 1.1 Datasets e Dados de Treinamento

#### 1.1.1 Datasets Mais Valiosos para o MR_POKER

| Prioridade | Dataset | Valor | Justificativa |
|-----------|---------|-------|---------------|
| **CRITICO** | **PHH Dataset** (21.6M maos, uoftcprg) | ★★★★★ | O maior corpus de maos de poker em formato padronizado. Essencial para: (a) treinar o BehaviorPredictor com dados reais em escala, (b) calibrar o SkillEstimator contra distribuicoes reais de niveis, (c) validar convergencia do CFR em cenarios realisticos. O MR_POKER ja exporta em formato PHH-like, facilitando integracao bidirecional. |
| **CRITICO** | **PokerBench** (RZ412, 100K+ cenarios) | ★★★★★ | Ja integrado via `pokerbench_eval.py`. Valor adicional: (a) usar como ground truth para medir desvio GTO de oponentes modelados, (b) fine-tunar modelos de linguagem para o LLM Agent, (c) criar "progressive difficulty" no curriculum. Licenca Apache-2.0 permite uso irrestrito. |
| **ALTO** | **Nemotron-Personas-Singapore** (888K personas) | ★★★★☆ | PGM hierarquico real com 22+ campos por persona. Upgrade direto do `SyntheticPlayerGenerator` — ja parcialmente implementado com PGM e Big Five. O dataset fornece distribuicoes realistas de personalidade que podem alimentar a geracao de oponentes sinteticos com fidelidade demografica. CC-BY-4.0. |
| **ALTO** | **gb6077/pokerstars.de hand histories** | ★★★★☆ | Historicos reais de jogadores em ambiente competitivo. Valor: (a) treinamento do opponent model com padroes reais de jogo, (b) validacao cruzada das previsoes do BehaviorPredictor, (c) calibracao de sizing tells e timing tells contra dados reais. |
| **MEDIO** | **SoelMgd/Poker_Dataset** | ★★★☆☆ | Features de decisao pre-computadas. Util para bootstrap rapido do treinamento do CNN-LSTM SkillEstimator antes de dados proprietarios estarem disponiveis. |
| **MEDIO** | **Kuhn Poker datasets** (the-acorn-ai) | ★★★☆☆ | Poker simplificado para validacao formal de convergencia CFR. Permite testes unitarios contra solucoes analiticas conhecidas — fortalece a suite de 1450+ testes. |

#### 1.1.2 Oportunidade de Datasets Proprietarios

**Estrategia de Dados Hibrida:**
1. **Camada 1 — Fundacao publica:** PHH + PokerBench + PokerStars histories como base
2. **Camada 2 — Sintetica enriquecida:** Gerar 1M+ maos via engine deterministico do MR_POKER com oponentes sinteticos PGM, rotuladas com decisoes GTO do solver
3. **Camada 3 — Proprietaria:** Dados de sessoes reais dos usuarios (com consentimento), criando flywheel de dados → melhor modelo → mais usuarios → mais dados
4. **Camada 4 — Augmentacao:** Aplicar tecnicas do PersonaHub (1B personas) para gerar variantes comportamentais infinitas a partir de dados reais limitados

**Vantagem competitiva:** Nenhum concorrente (PokerX, PokerAlpha, GTO Preflop Wizard) tem acesso simultaneo a 21.6M maos historicas + 888K personas sinteticas + engine deterministico para geracao controlada.

#### 1.1.3 Restricoes de Licenciamento do PokerBench

O PokerBench utiliza **Apache-2.0** para o dataset no HuggingFace, porem o paper original menciona **Commons Clause** para o software de avaliacao. Implicacoes:

| Uso | Permitido? | Detalhe |
|-----|-----------|---------|
| Treinar modelos internos com os dados | ✅ Sim | Apache-2.0 no dataset |
| Usar como benchmark interno | ✅ Sim | Uso de pesquisa/avaliacao |
| Redistribuir como parte do produto | ⚠️ Verificar | Commons Clause no software proibe "vender" |
| Criar produto comercial treinado nos dados | ✅ Sim | Apache-2.0 permite |
| Modificar e redistribuir o software de eval | ⚠️ Restrito | Commons Clause limita uso comercial do software |

**Recomendacao:** Usar os dados livremente. Reescrever o avaliador (ja feito em `pokerbench_eval.py`) ao inves de usar o codigo original.

---

### 1.2 Algoritmos e Tecnicas

#### 1.2.1 Tecnicas Mais Promissoras por Modulo

**A. CFR Solver — Convergencia e Escalabilidade**

| Tecnica | Paper | Impacto | Viabilidade | Prioridade |
|---------|-------|---------|-------------|-----------|
| **PDCFR+** (Optimistic Mirror Descent) | arXiv:2404.13891, IJCAI 2024 | Convergencia 2-5x mais rapida que DCFR+ em jogos grandes | ALTA — codigo aberto em github.com/rpSebastian/PDCFRPlus | **IMEDIATA** |
| **Deep Predictive DCFR** | arXiv:2511.08174 | Neural CFR model-free com reducao de variancia via bootstrapping | MEDIA — requer adaptacao do Deep CFR existente | CURTO PRAZO |
| **Embedding CFR** (CASIA) | arXiv:2511.12083 | Substituir clustering discreto por embeddings pre-treinados para abstracoes | MEDIA — ja temos `embedding_cfr.py` como base | CURTO PRAZO |
| **Robust Deep MCCFR** | arXiv:2509.00923 | Diagnosticos para distribution shift e action collapse em neural MCCFR | ALTA — ja temos `robust_deep_mccfr.py` | INCREMENTAL |
| **AlphaEvolve VAD-CFR** | arXiv:2602.16928 | Auto-descoberta de variantes CFR otimizadas | BAIXA — requer infra de LLM evolutionary | LONGO PRAZO |

**Insight critico:** O MR_POKER ja possui a suite CFR mais abrangente entre plataformas abertas (17+ variantes). A proxima fronteira nao e adicionar mais variantes, mas **escalar Deep CFR com neural approximation** para games de tamanho real. O PDCFR+ e a via mais rapida para isso — codigo aberto, convergencia comprovada, integracao direta com nosso `trainer.py`.

**B. Opponent Modeling — Precisao e Adaptabilidade**

| Tecnica | Paper | Impacto | Viabilidade | Prioridade |
|---------|-------|---------|-------------|-----------|
| **Consistent Opponent Modeling** | arXiv:2508.17671 | Garantias formais de convergencia para opponent model | ALTA — projected gradient descent no sequence-form | **IMEDIATA** |
| **Suspicion-Agent (Theory of Mind)** | arXiv:2309.17277 | Prever processo de pensamento do oponente, nao apenas acoes | MEDIA — requer LLM backbone (Qwen3-8B ou similar) | CURTO PRAZO |
| **Playing the Player (Patrick)** | arXiv:2512.04714 | Framework exploitativo maximal com perfis estatisticos | ALTA — alinhado com nossa filosofia GTO+exploit | **IMEDIATA** |
| **K-Level Reasoning** | arXiv:2402.01521 | Profundidade recursiva de raciocinio sobre oponente | MEDIA — complementa MBOM existente | MEDIO PRAZO |
| **General Social Agents** | arXiv:2508.17407 | Prever comportamento humano sem teoria especifica | BAIXA — requer dados de treinamento extensivos | LONGO PRAZO |

**Insight critico:** O pipeline comportamental do MR_POKER (tilt + timing + sizing + fatigue + meta-game) ja e mais sofisticado que qualquer sistema publicado. O gap principal e **garantias formais de convergencia** — o paper de Consistent Opponent Modeling (arXiv:2508.17671) preenche exatamente isso. A integracao com Playing the Player valida nossa abordagem de exploitacao personalizada.

**C. Behavioral Prediction — Fidelidade e Escala**

| Tecnica | Paper | Impacto | Viabilidade | Prioridade |
|---------|-------|---------|-------------|-----------|
| **Centaur Foundation Model** | arXiv:2410.20268 | 10M+ decisoes humanas para calibracao de vieses | ALTA — modelo disponivel no HF (Llama-3.1-Centaur-70B) | CURTO PRAZO |
| **DeepPersona** | arXiv:2511.07338 | 100+ atributos hierarquicos por persona | ALTA — complementa PGM do SyntheticPlayerGenerator | CURTO PRAZO |
| **PANDA (Big Five)** | arXiv:2504.06868 | Personalidade → comportamento em jogos | ALTA — ja temos Big Five no synthetic_players.py | INCREMENTAL |
| **TwinMarket (Cognitive Biases)** | arXiv:2502.01506 | Vieses cognitivos em simulacao comportamental financeira | MEDIA — transferivel para poker com adaptacao | MEDIO PRAZO |
| **PersonaEvolve** | arXiv:2509.16457 | Personas que evoluem durante sessao | MEDIA — alinhado com nosso fatigue_model | MEDIO PRAZO |

**Insight critico:** O paper SCOPE (arXiv:2601.07110) confirma que **demographics = apenas 1.5% da variancia comportamental**, validando a decisao do MR_POKER de focar em personality traits e comportamento observavel ao inves de dados demograficos. Isso e uma vantagem arquitetural permanente.

#### 1.2.2 Adaptacao de Algoritmos Poker → Dominio MR_POKER

O MR_POKER ja opera nativamente no dominio poker, entao a questao e: **como escalar os algoritmos para cenarios reais?**

**Desafio principal:** O HUNL tem ~10^161 information sets. Nossos 17+ variantes CFR funcionam em abstracoes menores. A ponte para o jogo completo requer:

1. **Abstraction + Solver:** Embedding CFR (arXiv:2511.12083) oferece a melhor via — embeddings pre-treinados substituem bucketing discreto, reduzindo perda de informacao. Ja temos `embedding_cfr.py` como base.

2. **Depth-Limited Subgame Solving:** DecisionHoldem (arXiv:2201.11580) demonstra 700+ mbb/h com subgame solving seguro. Adaptavel ao nosso `external_solver` adapter.

3. **Real-Time Re-planning:** Nosso `update_equiv.py` (Mirror Descent) ja faz decision-time planning. Combinar com `nfg_transformer.py` para re-planejamento context-aware.

#### 1.2.3 Estrategias Exploratorias (PokerAlpha-style)

Sim, o conceito de "Beyond GTO" do PokerAlpha e diretamente aplicavel. O paper "Playing the Player" (arXiv:2512.04714) formaliza isso:

**Ciclo GTO → Exploit:**
```
1. Convergir para baseline GTO via self-play (nosso CFR Suite)
2. Observar desvios do oponente (nosso BehavioralPipeline)
3. Calcular exploit maximo contra desvios (nosso BiasExploiter)
4. Aplicar exploit com safety margin (nosso exploitability.py garante bound)
5. Monitorar adaptacao do oponente (nosso meta_game.py)
6. Reverter para GTO se oponente se adaptar (loop)
```

O MR_POKER ja implementa todos os 6 passos. O diferencial competitivo e que **nenhum app comercial combina CFR convergence + exploit pipeline + meta-game tracking numa unica plataforma.**

---

### 1.3 Ferramentas e Frameworks

#### 1.3.1 PokerKit — Integracao com Stack MR_POKER

| Aspecto | Avaliacao |
|---------|----------|
| **Compatibilidade** | Alta — ambos Python, ambos modelam NLHE completo |
| **Complementaridade** | Media — nosso engine ja e mais completo (deterministico, seedable, CQRS) |
| **Custo de integracao** | Baixo (2-3 dias) — como parser de hand histories e validador externo |
| **Recomendacao** | Usar PokerKit como **parser de historicos PHH** e **validador de regras externo**, nao substituir nosso engine |

**Integracao recomendada:**
```python
# Usar PokerKit apenas para parsing de historicos externos
from pokerkit import HandHistory
# Converter PHH → nosso formato interno via adapter em packages/exports/
```

#### 1.3.2 GPUGT — Aceleracao CFR por GPU

| Aspecto | Avaliacao |
|---------|----------|
| **Valor** | ALTO — nosso `gpu_cfr.py` ja implementa batch CFR em GPU |
| **GPUGT diferencial** | Implementacao CUDA mais madura, testada em abstracoes grandes |
| **Integracao** | Media — exigiria adaptar kernels CUDA para nosso info_set.py |
| **Recomendacao** | Estudar tecnicas de GPUGT (batching, memory layout) e incorporar otimizacoes no nosso `gpu_cfr.py`, sem dependencia externa |

#### 1.3.3 Componentes de Visao Computacional

O sistema de deteccao de cartas com 99.4% de acuracia e **marginalmente relevante** para MR_POKER. Nosso foco e training/analytics, nao assistencia em tempo real em jogos ao vivo. Entretanto:

- **Potencial futuro:** Modulo de screen capture para importar maos de plataformas online diretamente para analise
- **Risco legal:** OCR de telas de poker online pode violar ToS de plataformas
- **Recomendacao:** Nao priorizar; manter como possibilidade para v2.0

#### 1.3.4 Frameworks Academicos

| Framework | Uso Recomendado |
|-----------|----------------|
| **OpenSpiel** (DeepMind) | Benchmark externo para validar nossos CFR variants contra implementacoes de referencia |
| **RLCard** | Ambiente de treinamento alternativo; util para RL agents (PPO, DMC) em abstracoes menores |
| **Valet** (21 jogos IIG) | Testar generalizacao dos nossos CFR variants para alem de poker |

---

### 1.4 Inspiracoes Comerciais

#### 1.4.1 Modelos de Negocio Aplicaveis

| App | Modelo | Aplicabilidade ao MR_POKER | Adaptacao |
|-----|--------|---------------------------|-----------|
| **PokerX** | Freemium + assinatura premium | ★★★★★ | Tier gratuito: engine + analise basica. Premium: CFR solver + exploit pipeline + coach reports |
| **GTO Preflop Wizard** | Quizzes adaptativos + tiers | ★★★★☆ | Nosso `curriculum` ja suporta progressive difficulty. Adicionar gamificacao (streaks, badges, leaderboard) |
| **PokerAlpha** | Analise multi-way premium | ★★★☆☆ | Nosso foco e HUNL; multi-way requer expansao significativa do CFR (computacionalmente prohibitivo) |

#### 1.4.2 Features Prioritarias (Inspiradas na Concorrencia)

**Tier 1 — Diferenciais Unicos do MR_POKER (ja implementados, comunicar melhor):**
1. Pipeline comportamental integrado (nenhum concorrente tem 15+ modulos de opponent modeling)
2. 17+ variantes CFR (amplitude inigualavel)
3. Local-first com privacidade total

**Tier 2 — Features a Implementar (inspiradas nos concorrentes):**

| Feature | Inspiracao | Esforco | Impacto |
|---------|-----------|---------|---------|
| **Coach Report Automatico** | PokerX | Medio (30-40h) | ALTO — diferencial forte para amadores |
| **Quizzes Adaptativos** | GTO Preflop Wizard | Medio (20-30h) | ALTO — retencao e engajamento |
| **Replay Interativo** | PokerX | Baixo (10-15h) | MEDIO — ja temos state snapshots |
| **Analise Multi-way (3+ jogadores)** | PokerAlpha | MUITO ALTO (200h+) | MEDIO — mercado pede, mas custo computacional e barreira |
| **Dashboard Social (leaderboard)** | PokerX | Medio (20h) | MEDIO — apenas se migrar para cloud |

**Tier 3 — Diferenciais Futuros (baseados na pesquisa):**
1. **Theory of Mind Viewer** — mostrar ao usuario o que a IA "pensa" que o oponente esta pensando (Suspicion-Agent)
2. **Bias Report** — identificar vieses cognitivos do usuario e sugerir correcoes (Centaur + BiasExploiter)
3. **Persona Sandbox** — criar oponentes sinteticos com personalidades especificas para treino direcionado (DeepPersona + PGM)

---

## 2. ANALISE DE RISCOS E DESAFIOS

### 2.1 Riscos Tecnicos

#### 2.1.1 Limitacoes Computacionais para CFR em Larga Escala

| Desafio | Severidade | Mitigacao |
|---------|-----------|-----------|
| **HUNL tem ~10^161 info sets** | CRITICA | Abstraction via Embedding CFR + subgame solving |
| **Deep CFR requer GPU para treino** | ALTA | Compact CFR (8-16x reducao memoria) + Lazy CFR (custo por iteracao reduzido) |
| **GPU CFR precisa de hardware** | MEDIA | Oferecer "solver-as-a-service" opcional, manter local-first como default |
| **Convergencia lenta em abstracoes grandes** | ALTA | PDCFR+ (2-5x mais rapido) + warm start com estrategias pre-treinadas |

**Calculo de viabilidade local:**
- Kuhn Poker (12 info sets): <1s no CPU ✅
- Leduc Poker (936 info sets): ~10s no CPU ✅
- HUNL abstrado (10^4-10^5 info sets): 1-30min no CPU ⚠️
- HUNL completo (10^161 info sets): Impossivel sem subgame solving + neural approximation ❌

**Estrategia:** Oferecer resolucao em 3 tiers — (1) pre-computado para spots comuns, (2) CFR on-demand para abstracoes medias, (3) depth-limited subgame solving para decisoes criticas.

#### 2.1.2 Variancia em Jogos de Informacao Imperfeita

A variancia inerente ao poker (bad beats, coolers) cria desafios para UX:

| Problema | Impacto UX | Solucao |
|---------|-----------|---------|
| Jogador segue conselho GTO e perde | Frustacao → churn | Mostrar EV esperado vs resultado real; normalizar por 1000+ maos |
| Opponent model erra previsao | Perda de confianca | Exibir intervalo de confianca; explicar que modelo melhora com dados |
| Solver sugere bluff que falha | Percepcao de IA "ruim" | Educacao sobre mixed strategies e frequencias; replay com alternativas |

**Solucao arquitetural:** O modulo `evaluation/` deve separar claramente **qualidade da decisao** (processo) de **resultado** (outcome). Implementar "luck-adjusted winrate" usando tecnicas do paper Elo Uncovered (arXiv:2311.17295).

#### 2.1.3 Latencia para Analise em Tempo Real

| Componente | Latencia Atual | Latencia Necessaria | Viavel? |
|-----------|---------------|--------------------|---------|
| Engine (deal + validate) | <1ms | <10ms | ✅ Sim |
| Equity (Monte Carlo 10K sims) | ~50ms | <100ms | ✅ Sim |
| BehavioralPipeline (8 modulos) | ~100-200ms | <400ms | ✅ Sim |
| CFR query (pre-computado) | <5ms | <50ms | ✅ Sim |
| CFR solve (on-demand) | 1-30min | <1s para sugestao | ⚠️ Requer pre-compute + lookup |
| Deep CFR (neural forward pass) | ~10-50ms | <100ms | ✅ Com GPU |

**Conclusao:** Para analise pos-mao e coaching, a latencia nao e gargalo. Para assistencia durante o jogo, apenas consultas pre-computadas sao viaveis.

---

### 2.2 Riscos Legais e Regulatorios

#### 2.2.1 Licenciamento de Datasets e Ferramentas

| Recurso | Licenca | Restricao | Risco |
|---------|---------|-----------|-------|
| PokerBench dataset | Apache-2.0 | Nenhuma significativa | BAIXO |
| PokerBench software | Commons Clause | Nao pode "vender" o software de eval | BAIXO — ja reescrevemos |
| PHH Dataset | CC-BY-4.0 | Atribuicao obrigatoria | BAIXO |
| Nemotron-Personas | CC-BY-4.0 | Atribuicao obrigatoria | BAIXO |
| OpenSpiel | Apache-2.0 | Nenhuma significativa | BAIXO |
| DecisionHoldem | Apache-2.0 | Nenhuma significativa | BAIXO |
| PDCFR+ | MIT | Nenhuma significativa | BAIXO |
| Centaur model | Llama 3.1 Community | Restricoes comerciais >700M MAU | MEDIO — verificar se uso indireto se aplica |
| DeepSeek-R1 | MIT | Nenhuma significativa | BAIXO |

**Conclusao:** O ecossistema de poker AI e surpreendentemente aberto. A maioria dos recursos criticos usa Apache-2.0 ou MIT. O unico risco real e o modelo Centaur (Llama 3.1 Community License) — mitigavel usando destilacao ou parametros proprios calibrados contra os dados do Centaur.

#### 2.2.2 Regulamentacoes de Jogo Responsavel

| Jurisdicao | Regulamentacao | Impacto no MR_POKER |
|-----------|---------------|---------------------|
| **UE (Gaming Act)** | Ferramentas de jogo devem incluir alertas de risco | MEDIO — adicionar modulo de jogo responsavel |
| **UK (Gambling Commission)** | Proibicao de ferramentas de assistencia em tempo real | ALTO — restringir a modo training/replay apenas |
| **US (variavel por estado)** | Sem regulamentacao federal unificada | BAIXO — foco em educacao, nao assistencia |
| **Brasil** | Lei do Jogo Online (2025) recente | MEDIO — monitorar requisitos de conformidade |

**Oportunidade:** O estudo vietnamita de deteccao de problemas com ML (publicado em fevereiro/2026) pode ser adaptado como feature de **jogo responsavel**: detectar padroes de risco no comportamento do usuario (sessoes longas, tilt persistente, aumento de stakes apos perdas) e sugerir pausas. Isso transforma uma obrigacao regulatoria em diferencial de produto.

**Implementacao proposta:** Reutilizar `tilt_detector.py` e `fatigue_model.py` invertidos — ao inves de explorar tilt do oponente, detectar tilt do proprio usuario e alertar.

#### 2.2.3 Consideracoes para Aplicacoes Financeiras (Trading)

Se o MR_POKER expandir para trading algoritmico (teoria dos jogos → mercados financeiros):

| Regulacao | Jurisdicao | Impacto |
|-----------|-----------|---------|
| MiFID II | UE | Requisitos de transparencia algoritmica; auditoria obrigatoria |
| SEC Rule 15c3-5 | US | Controles de risco pre-trade; limites de exposicao |
| CVM | Brasil | Registro como gestor automatizado; compliance com regulamentacoes |

**Conclusao:** Transferencia para trading e tecnicamente viavel mas regulatoriamente complexa. Recomendacao: manter foco em poker no curto/medio prazo, desenvolver spin-off financeiro como projeto separado se validado.

---

### 2.3 Riscos de Mercado e Concorrencia

#### 2.3.1 Saturacao do Mercado de Apps de Poker AI

**Analise de saturacao:**

| Segmento | Saturacao | Players | MR_POKER Fit |
|----------|----------|---------|-------------|
| **Solvers GTO puros** | ALTA | PioSolver, GTO+, Simple GTO Trainer | Nao competir diretamente |
| **Treinadores AI** | MEDIA | PokerX, GTO Preflop Wizard | **COMPETIR** — nosso pipeline comportamental e unico |
| **Analise exploitativa** | BAIXA | PokerAlpha (parcial) | **OPORTUNIDADE** — ninguem faz exploit + GTO integrado |
| **Plataforma de pesquisa** | MUITO BAIXA | OpenSpiel (generico) | **OPORTUNIDADE** — nicho academico/pro |

**Diferencial competitivo sustentavel:**
1. **Profundidade do opponent modeling** (15+ modulos vs 1-2 nos concorrentes)
2. **Amplitude de CFR** (17+ variantes vs 1-2 nos concorrentes)
3. **Local-first** (privacidade como feature num mercado pos-GDPR)
4. **Open research** (atrair contribuicoes academicas)

#### 2.3.2 Impacto de Players Chineses (Foshan Youfou et al.)

| Ameaca | Probabilidade | Impacto | Mitigacao |
|--------|-------------|---------|-----------|
| CFR com programacao dinamica (Foshan) | MEDIA | Pode reduzir custo de resolucao | Nosso Embedding CFR + PDCFR+ sao mais avancados |
| Tencent AI Lab entrar no mercado consumer | BAIXA | Massivo se ocorrer | Diferenciar via UX, local-first, nicho ocidental |
| DecisionHoldem virar produto | BAIXA | Solver open-source competitivo | Ja estamos alem — nosso pipeline comportamental e inigualavel |

**Conclusao:** O risco real nao e de solvers chineses (nosso stack tecnica ja os supera), mas de uma big tech (Google DeepMind, Meta FAIR) lancar um produto consumer. Mitigacao: construir moat via dados proprietarios (flywheel de usuarios) e profundidade de features.

#### 2.3.3 Canibalizacao por LLMs Jogadores de Poker

O PokerBench mostra que GPT-4 atinge 53.55% de acuracia em decisoes poker — **muito abaixo de solvers GTO**. Fine-tuning melhora dramaticamente (arXiv:2501.08328).

**Cenario em 2-3 anos:**
- LLMs generalistas NAO substituirao ferramentas especializadas (PokerBench confirma gap significativo)
- LLMs fine-tuned podem se tornar **concorrentes de chatbots de coaching** (ameaca media)
- LLMs como **complemento** (explicacoes em linguagem natural de decisoes GTO) = **oportunidade**

**Recomendacao:** Integrar LLM como camada de explicacao sobre decisoes do solver, nao como substituto. Nosso `llm_agent` pode ser reposicionado como "Coach AI conversacional" que usa CFR internamente.

---

## 3. TENDENCIAS E INOVACOES RELEVANTES

### 3.1 Panorama de Longo Prazo

#### 3.1.1 LLMs como Jogadores de Poker — Timeline 2026-2029

| Horizonte | Previsao | Base | Impacto no MR_POKER |
|-----------|---------|------|---------------------|
| **2026** | LLMs fine-tuned atingem nivel mid-stakes | PokerBench + SPIRAL + PokerGPT | OPORTUNIDADE — oferecer como oponentes de treino |
| **2027** | LLMs com tool-use (ToolPoker-style) atingem nivel high-stakes | arXiv:2602.00528 tendencia | NEUTRO — solver interno e complementar |
| **2028** | LLMs jogam poker conversacional com bluff em linguagem natural | Evolucao de Suspicion-Agent | OPORTUNIDADE — novo mercado de "poker social AI" |
| **2029** | LLMs superhumanos em HUNL sem solver externo | Extrapolacao de DeepSeek-R1 + SPIRAL | RISCO — commoditizacao de solvers puros |

**Estrategia:** O valor de longo prazo do MR_POKER nao esta no solver (commoditizavel) mas no **pipeline comportamental** (15+ modulos de opponent modeling sao dificeis de replicar).

#### 3.1.2 CFR com Programacao Dinamica — Ruptura ou Evolucao?

**Avaliacao: EVOLUCAO INCREMENTAL, nao ruptura.**

Evidencias:
- Foshan Youfou claim "arvore reduzida para solucao completa" e marketing. O tamanho do game tree HUNL (~10^161) nao pode ser resolvido por nenhuma programacao dinamica em hardware existente
- A real inovacao e **subgame solving mais eficiente** (DecisionHoldem faz isso desde 2022)
- PDCFR+ (optimistic mirror descent) e Embedding CFR sao avancos mais significativos matematicamente
- AlphaEvolve (arXiv:2602.16928) mostra que **meta-otimizacao de CFR via LLMs** e o proximo salto real

**Conclusao:** Investir em PDCFR+ e Embedding CFR, nao em "CFR com programacao dinamica" generico.

#### 3.1.3 Processamento Local vs. Cloud

**Tendencia confirmada:** Apple Silicon (M-series) democratiza inference local:
- M4 Ultra: 512GB unified memory, ~30 TOPS — executa LLM 70B localmente
- Tendencia: edge computing + privacidade crescente (GDPR, LGPD)
- Contra-tendencia: solvers grandes requerem horas de compute (cloud inevitavel para treino)

**Estrategia hibrida para MR_POKER:**

| Componente | Local | Cloud | Justificativa |
|-----------|-------|-------|---------------|
| Engine + equity | ✅ | ❌ | Baixo custo, privacidade |
| BehavioralPipeline | ✅ | ❌ | Latencia critica, privacidade |
| CFR lookup (pre-computado) | ✅ | ❌ | Tabela local, rapido |
| CFR solve (on-demand) | ❌ | ✅ | Computacionalmente intensivo |
| Deep CFR training | ❌ | ✅ | Requer GPU cluster |
| LLM Coach | ⚠️ | ✅ | Depende do tamanho do modelo |
| Dados do usuario | ✅ | ❌ | Privacidade como feature |

**Recomendacao:** Manter local-first como principio, oferecer cloud opcional para features compute-heavy (solver, treinamento). Apple Silicon e a plataforma alvo prioritaria para instalacao local.

---

### 3.2 Transferencia para Outros Dominios

#### 3.2.1 Mapa de Transferibilidade

| Dominio | Tecnica Transferivel | Modulo MR_POKER | Complexidade | Potencial Comercial |
|---------|---------------------|----------------|-------------|-------------------|
| **Trading algoritmico** | CFR para informacao imperfeita, opponent modeling, bias exploitation | cfr_agent, opponent_model, bias_exploiter | ALTA | ★★★★★ |
| **Leiloes (ads, procurement)** | Teoria dos jogos, bidding strategies, Nash equilibria | cfr_agent (QRE), exploitability | MEDIA | ★★★★☆ |
| **Negociacao automatizada** | Theory of Mind, meta-game tracking, tilt detection | opponent_model (ToM, meta_game, tilt) | MEDIA | ★★★★☆ |
| **Cybersecurity (adversarial)** | Game tree search, exploit strategies, adaptation | cfr_agent, adaptive_agent | ALTA | ★★★★☆ |
| **Estrategia empresarial** | Competitive intelligence, scenario planning | decision-time planning, NfgTransformer | MEDIA | ★★★☆☆ |
| **Simulacao social** | Synthetic personas, behavioral prediction | synthetic_players, behavior_prediction | BAIXA | ★★★☆☆ |
| **Saude (diagnostico sob incerteza)** | Bayesian reasoning, sequential decision-making | bayesian_range, particle_filter | ALTA | ★★★☆☆ |

#### 3.2.2 Viabilidade de Spin-off

**Candidato mais promissor: Trading Algoritmico**

A transferencia poker → trading e direta:
- **Informacao imperfeita:** Ordem de mercado = "mao escondida" dos outros players
- **Bluff/Deception:** Spoofing, iceberg orders = equivalente de bluffs
- **Opponent modeling:** Market maker behavior, institutional flow detection
- **CFR:** Nash equilibrium para optimal execution em mercados competitivos
- **Bias exploitation:** Behavioral finance (loss aversion, disposition effect)

**Papers que validam a transferencia:**
- TwinMarket (arXiv:2502.01506): Vieses cognitivos em simulacao de mercados financeiros
- "Beyond Survival" (arXiv:2510.11389): Deception modeling transferivel

**Complexidade:** ALTA — regulamentacao financeira, dados de mercado em tempo real, infraestrutura de baixa latencia sao requisitos adicionais pesados.

**Recomendacao:** Validar conceito com paper academico ("Game-Theoretic Trading via CFR"), construir MVP em 6 meses, buscar parceria com fintech.

#### 3.2.3 Deteccao de Comportamentos de Risco

O estudo vietnamita de deteccao de problemas com ML pode ser adaptado:

**Modulos reutilizaveis do MR_POKER:**
1. `tilt_detector.py` → detectar frustration loops em usuarios
2. `fatigue_model.py` → identificar sessoes excessivamente longas
3. `meta_game.py` → detectar escalation patterns (aumento de stakes apos perdas)
4. `behavior_prediction.py` → prever risco de comportamento destrutivo

**Aplicacoes alem do poker:**
- Plataformas de apostas esportivas: deteccao precoce de jogo problematico
- Apps financeiros: detectar overtrading e revenge trading
- Gaming: identificar comportamento viciante em jogos mobile

**Impacto:** Feature de jogo responsavel = diferencial regulatorio + reputacional + moral. Implementar como prioridade media.

---

## 4. RECOMENDACOES PRATICAS E ACIONAVEIS

### 4.1 Curto Prazo (0-3 meses)

#### Acoes Imediatas — Alto Impacto, Baixo Custo

| # | Acao | Esforco | Impacto | Dependencia |
|---|------|---------|---------|------------|
| **C1** | **Integrar PDCFR+** — portar codigo de github.com/rpSebastian/PDCFRPlus para nosso trainer.py | 3-5 dias | Convergencia 2-5x mais rapida | Nenhuma |
| **C2** | **Download PHH Dataset** — 21.6M maos, configurar pipeline de ingestao | 2-3 dias | Base de dados massiva para treinamento | Storage (~50GB) |
| **C3** | **Implementar "Playing the Player"** — ciclo GTO→exploit formalizado inspirado em arXiv:2512.04714 | 5-7 dias | Framework exploitativo completo | C1 (solver melhorado) |
| **C4** | **Consistent Opponent Modeling** — adicionar convergence guarantees ao opponent model (arXiv:2508.17671) | 3-5 dias | Garantia formal de convergencia | Nenhuma |
| **C5** | **Jogo Responsavel v1** — inverter tilt_detector e fatigue_model para detectar risco no usuario | 2-3 dias | Compliance + diferencial moral | Nenhuma |
| **C6** | **Coach Report v1** — gerar PDF/HTML automatico com analise de sessao, leaks, e recomendacoes | 5-7 dias | Feature mais solicitada por amadores | Nenhuma |

**Investimento total:** ~25-35 dias de desenvolvimento
**ROI esperado:** Solver 2-5x mais rapido + framework exploitativo + compliance + feature killer (coach report)

#### Datasets Prioritarios para Download

```bash
# 1. PHH Dataset (prioridade maxima)
git clone https://github.com/uoftcprg/phh-dataset.git

# 2. Nemotron-Personas (ja planejado)
huggingface-cli download nvidia/Nemotron-Personas-Singapore

# 3. PokerStars hand histories
huggingface-cli download gb6077/pokerstars.de-nlh-6max-2024

# 4. Poker Dataset (features de decisao)
huggingface-cli download SoelMgd/Poker_Dataset

# 5. Kuhn Poker (validacao CFR)
huggingface-cli download the-acorn-ai/kuhn-poker
```

#### Contratacoes/Parcerias Estrategicas

| Tipo | Perfil | Urgencia | Justificativa |
|------|--------|----------|---------------|
| **Contratacao** | RL/Game Theory researcher (PhD ou equivalente) | ALTA | Escalar Deep CFR + subgame solving requer expertise dedicada |
| **Parceria academica** | UTokyo (Matsuo Lab — Suspicion-Agent) | MEDIA | Acesso a expertise em Theory of Mind para jogos |
| **Parceria academica** | NTU Singapore (Prof. Bo An — MARL) | MEDIA | Robustez de modelos multi-agente |
| **Parceria academica** | Tencent AI Lab / CASIA | ALTA | Grupo mais produtivo em CFR no mundo (6 papers) |
| **Advisor** | Jogador profissional high-stakes | MEDIA | Validacao de features, credibilidade, feedback loop |

---

### 4.2 Medio Prazo (3-12 meses)

#### Roadmap de Desenvolvimento

**Q2 2026 (Abril-Junho):**

| Feature | Base Cientifica | Modulo | Esforco |
|---------|----------------|--------|---------|
| Deep Predictive DCFR | arXiv:2511.08174 | cfr_agent/vr_deep_dcfr.py upgrade | 20-30 dias |
| Theory of Mind v1 | arXiv:2309.17277 (Suspicion-Agent) | opponent_model/ novo modulo | 15-20 dias |
| Quizzes Adaptativos | GTO Preflop Wizard inspiracao | curriculum/ + apps/api/ | 15-20 dias |
| Replay Interativo | State snapshots existentes | apps/api/ + frontend | 10-15 dias |

**Q3 2026 (Julho-Setembro):**

| Feature | Base Cientifica | Modulo | Esforco |
|---------|----------------|--------|---------|
| LLM Coach Conversacional | PokerGPT + ToolPoker | llm_agent/ upgrade | 30-40 dias |
| Depth-Limited Subgame Solving | DecisionHoldem (arXiv:2201.11580) | external_solver/ upgrade | 25-30 dias |
| PersonaEvolve (personas que mudam) | arXiv:2509.16457 | synthetic_players.py upgrade | 10-15 dias |
| OpenSkill Rating System | arXiv:2401.05451 | skill_estimator.py upgrade | 5-10 dias |

**Q4 2026 (Outubro-Dezembro):**

| Feature | Base Cientifica | Modulo | Esforco |
|---------|----------------|--------|---------|
| Cloud Solver (opcional) | Infraestrutura | infra/ + apps/api/ | 40-50 dias |
| Multi-way v1 (3 players) | PokerAlpha inspiracao | cfr_agent/ expansao | 50-60 dias |
| Valet Benchmark Integration | arXiv:2603.03252 | evaluation/ novo modulo | 15-20 dias |
| Beta Release Candidate | Todas as features acima | Todos | 20-30 dias (QA/polish) |

#### Investimentos em P&D

| Area | Investimento | Retorno Esperado |
|------|-------------|-----------------|
| **Neural CFR escalavel** | 2-3 meses de researcher | Resolver abstracoes 10x maiores |
| **LLM fine-tuning para poker** | 1-2 meses + GPU budget (~$5K) | Coach conversacional nivel pro |
| **Dados reais (hand histories)** | Licenciamento ou coleta ($2-5K) | Modelos calibrados contra jogo real |
| **UX/Frontend** | 3-4 meses de designer/dev | Transicao de ferramenta de pesquisa para produto |

#### Posicionamento Competitivo

**Mensagem central:** "O unico sistema que combina a precisao de um solver GTO com a inteligencia de um coach que entende o jogo do seu oponente — tudo rodando no seu computador."

**Segmentacao:**

| Segmento | Feature Principal | Tier |
|----------|------------------|------|
| **Amadores (low-stakes)** | Coach Report + Quizzes + Replay | Gratuito / $9.99/mes |
| **Regulares (mid-stakes)** | CFR Solver + Exploit Pipeline | $29.99/mes |
| **Profissionais (high-stakes)** | Full pipeline + Custom opponents + API | $99.99/mes |
| **Pesquisadores** | Open-source core + papers + benchmarks | Gratuito (MIT) |

---

### 4.3 Longo Prazo (1-3 anos)

#### Visao de Evolucao da Plataforma

**2026-2027: "O Lab que Vira Produto"**
- Transicao de alpha para release publica
- Pipeline comportamental como core IP
- Local-first com cloud opcional
- Comunidade open-source ativa

**2027-2028: "O Coach AI Definitivo"**
- LLM Coach conversacional em linguagem natural
- Theory of Mind viewer ("veja o que a IA pensa sobre seu oponente")
- Multi-way support (3-6 jogadores)
- Integracao com plataformas de poker online (API read-only)
- Mobile companion (analise pos-sessao)

**2028-2029: "Alem do Poker"**
- Spin-off para trading algoritmico (se validado)
- Plataforma de simulacao de decisao sob incerteza
- SDK para pesquisadores de Game Theory
- Parcerias com universidades para curricula de IA

#### Preparacao para Disrupcoes Tecnologicas

| Disrupcao | Probabilidade (3 anos) | Preparacao |
|-----------|----------------------|-----------|
| **LLMs superhumanos em poker** | 40% | Reposicionar para coaching (explicacao) em vez de solver (competicao) |
| **Computacao quantica para Game Theory** | 5% | Monitorar; sem acao imediata |
| **Regulamentacao proibindo bots** | 20% | Posicionar como ferramenta de treinamento, nao assistencia em tempo real |
| **Consolidacao de mercado (aquisicao)** | 30% | Construir moat via dados + comunidade; estar aberto a exit |
| **Commoditizacao de solvers** | 60% | Pipeline comportamental e o moat verdadeiro (nao o solver) |
| **Web3/blockchain poker** | 15% | Engine deterministico + seedable ja e compativel com verificacao on-chain |

#### Expansao para Novos Mercados

| Mercado | Via de Entrada | Risco | Potencial |
|---------|---------------|-------|-----------|
| **Asia (China, Coreia)** | Parceria com Tencent AI Lab ou SKT | ALTO (regulamentacao) | ★★★★★ (maior mercado de poker) |
| **America Latina (Brasil)** | Lei do Jogo Online 2025; portugues nativo | BAIXO | ★★★★☆ |
| **Europa Ocidental** | GDPR-friendly (local-first) | MEDIO | ★★★★☆ |
| **Trading (fintech)** | Spin-off com parceiro financeiro | ALTO (regulamentacao) | ★★★★★ |
| **Academia** | Open-source + papers + SDK | BAIXO | ★★★☆☆ (reputacao, nao receita) |

---

## 5. MATRIZ DE PRIORIZACAO CONSOLIDADA

### Top 20 Acoes por Impacto x Esforco

| Rank | Acao | Prazo | Esforco | Impacto | Score |
|------|------|-------|---------|---------|-------|
| 1 | Integrar PDCFR+ | Imediato | 3-5d | ★★★★★ | 25 |
| 2 | Download PHH Dataset + pipeline | Imediato | 2-3d | ★★★★★ | 25 |
| 3 | Coach Report Automatico v1 | Curto | 5-7d | ★★★★★ | 24 |
| 4 | Playing the Player (exploit framework) | Curto | 5-7d | ★★★★★ | 24 |
| 5 | Consistent Opponent Modeling | Curto | 3-5d | ★★★★☆ | 22 |
| 6 | Jogo Responsavel v1 | Curto | 2-3d | ★★★★☆ | 22 |
| 7 | Deep Predictive DCFR | Medio | 20-30d | ★★★★★ | 20 |
| 8 | Quizzes Adaptativos | Medio | 15-20d | ★★★★☆ | 19 |
| 9 | Theory of Mind v1 | Medio | 15-20d | ★★★★☆ | 19 |
| 10 | OpenSkill Rating System | Medio | 5-10d | ★★★★☆ | 21 |
| 11 | Replay Interativo | Medio | 10-15d | ★★★☆☆ | 17 |
| 12 | LLM Coach Conversacional | Medio | 30-40d | ★★★★★ | 18 |
| 13 | Depth-Limited Subgame Solving | Medio | 25-30d | ★★★★☆ | 17 |
| 14 | PersonaEvolve (personas dinamicas) | Medio | 10-15d | ★★★☆☆ | 17 |
| 15 | Valet Benchmark | Medio | 15-20d | ★★★☆☆ | 15 |
| 16 | Cloud Solver (opcional) | Longo | 40-50d | ★★★★☆ | 14 |
| 17 | Multi-way v1 (3 players) | Longo | 50-60d | ★★★★☆ | 13 |
| 18 | Mobile Companion | Longo | 60-80d | ★★★★☆ | 12 |
| 19 | Trading Spin-off (MVP) | Longo | 90-120d | ★★★★★ | 11 |
| 20 | Web3 Poker Engine | Longo | 30-40d | ★★☆☆☆ | 8 |

---

## 6. CONCLUSAO EXECUTIVA

### O Que a Pesquisa Revela

A varredura de **385+ recursos em 12 paises** confirma que o MR_POKER esta **tecnicamente a frente da maioria dos sistemas publicados** em termos de amplitude:

- **Nenhum sistema publicado** combina 17+ variantes CFR + 15+ modulos de opponent modeling + pipeline comportamental integrado
- **O gap principal e escala**, nao tecnica: passar de abstracoes academicas para resolucao de game trees reais
- **O moat sustentavel e o pipeline comportamental**, nao o solver (que sera commoditizado em 2-3 anos)

### As 5 Decisoes Estrategicas Mais Importantes

1. **Investir no pipeline comportamental como core IP** (nao no solver)
2. **Adotar PDCFR+ imediatamente** (2-5x convergencia, codigo aberto)
3. **Construir Coach Report como feature matadora** (diferenciacao UX)
4. **Manter local-first como principio** (privacidade crescente e regulamentacao)
5. **Preparar para commoditizacao de solvers** (reposicionar para coaching + exploit)

### Risco Principal

Nenhuma big tech (Google DeepMind, Meta FAIR) lancou um produto consumer de poker AI. Se isso acontecer, o MR_POKER precisa ja ter construido moat via dados proprietarios e comunidade. O momento de agir e agora.

---

## APENDICE A: Referencias Completas

Para detalhes completos de cada recurso mencionado, consultar:

1. `docs/100_Global_AI_Research_Report.md` — 385+ recursos em 12 paises
2. `docs/104_China_AI_Resources_for_Poker.md` — 52 recursos chineses, Tencent/CASIA
3. `docs/104_Poker_AI_External_Research_2024_2026.md` — 76 papers poker-especificos
4. `docs/105_Emerging_Asia_AI_Resources_for_Poker.md` — India, SEA, UAE

## APENDICE B: Mapeamento Recurso → Modulo MR_POKER

| Recurso | Modulo(s) Alvo | Tipo de Integracao |
|---------|---------------|-------------------|
| PDCFR+ (arXiv:2404.13891) | cfr_agent/trainer.py | Port direto de codigo |
| PHH Dataset (21.6M maos) | opponent_model/*, skill_estimator.py | Dados de treinamento |
| PokerBench (RZ412) | evaluation/pokerbench_eval.py | Ja integrado; expandir |
| Playing the Player (arXiv:2512.04714) | opponent_model/behavioral_pipeline.py | Arquitetura exploitativa |
| Consistent Opponent Modeling (arXiv:2508.17671) | opponent_model/profile.py | Garantias convergencia |
| Deep Predictive DCFR (arXiv:2511.08174) | cfr_agent/vr_deep_dcfr.py | Upgrade neural CFR |
| Suspicion-Agent (arXiv:2309.17277) | opponent_model/ (novo: tom.py) | Theory of Mind |
| Centaur (arXiv:2410.20268) | opponent_model/bias_exploiter | Calibracao de vieses |
| Nemotron-Personas (nvidia) | opponent_model/synthetic_players.py | PGM enriquecido |
| DeepPersona (arXiv:2511.07338) | opponent_model/synthetic_players.py | 100+ atributos |
| OpenSkill (arXiv:2401.05451) | opponent_model/skill_estimator.py | Rating Bayesiano |
| DecisionHoldem (arXiv:2201.11580) | external_solver/ | Subgame solving |
| Embedding CFR (arXiv:2511.12083) | cfr_agent/embedding_cfr.py | Ja implementado; refinar |
| Valet (arXiv:2603.03252) | evaluation/ (novo) | Benchmark multi-jogo |
| SPIRAL (arXiv:2506.24119) | cfr_agent/ + ppo_agent/ | Self-play + reasoning |
| ToolPoker (arXiv:2602.00528) | llm_agent/ + external_solver/ | LLM + solver integrado |
| TwinMarket (arXiv:2502.01506) | opponent_model/bias_exploiter | Vieses cognitivos |
| PANDA (arXiv:2504.06868) | opponent_model/synthetic_players.py | Big Five → comportamento |
| Robust Deep MCCFR (arXiv:2509.00923) | cfr_agent/robust_deep_mccfr.py | Ja implementado; expandir |
| AlphaEvolve CFR (arXiv:2602.16928) | cfr_agent/ (meta-otimizacao) | Longo prazo |
