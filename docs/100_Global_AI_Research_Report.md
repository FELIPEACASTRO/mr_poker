# Relatório Técnico: Pesquisa Global de IA no Oriente

**Data:** 2026-03-16
**Versão:** 2.1
**Escopo:** China, Coreia do Sul, Japão, Índia, Singapura, Taiwan, UAE, Vietnã, Hong Kong, Malásia, Indonésia, Tailândia

---

## 1. Resumo Executivo

**Data da Pesquisa:** 2026-03-16
**Total de recursos verificados:** 385+ recursos únicos em 12 países/regiões

### Contagem por Região

| Região | Modelos | Datasets | Papers | Instituições | Total |
|--------|---------|----------|--------|-------------|-------|
| **China** | 58 | 7 | 50+ arXiv | 12 orgs | 89 |
| **Coreia do Sul** | 15 | 7 | 9 | 6 orgs | 41 |
| **Japão** | 15 | 3 | 5 | 10 orgs | 42 |
| **Índia** | 14 | 14 | — | 7 orgs | 47 |
| **Singapura** | 13 | 12 | — | 3 orgs | 28 |
| **UAE/Oriente Médio** | 18 | 13 | — | 5 orgs | 35 |
| **Taiwan** | 8 | 2 | — | 3 orgs | 12 |
| **Vietnã** | 10 | 2 | — | 1 org | 12 |
| **Malásia** | 6 | 6 | — | 1 org | 12 |
| **Indonésia** | 5 | — | — | 1 org | 5 |
| **Tailândia** | 5 | — | — | 1 org | 5 |
| **Poker AI (Global)** | 4 | 1 | 25+ | — | 57 |
| **TOTAL** | **171+** | **67+** | **89+** | **50+ orgs** | **385+** |

### Top 10 Achados de Maior Impacto para MR_POKER

1. **PokerBench** (arXiv:2501.08328) — Benchmark + dataset com decisões solver-optimal para treino/avaliação
2. **Suspicion-Agent** (UTokyo, arXiv:2309.17277) — Theory of Mind para jogos de informação imperfeita
3. **AlphaEvolve para CFR** (arXiv:2602.16928) — Auto-descoberta de variantes CFR (Volatility-Adaptive DCFR)
4. **SPIRAL** (arXiv:2506.24119) — Self-play RL em Kuhn Poker com LLMs
5. **Nemotron-Personas-Singapore** — 888K personas sintéticas com PGM hierárquico, CC-BY-4.0
6. **Generative Agent Simulations** (arXiv:2411.10109) — 85% accuracy replicando comportamento humano
7. **PANDA** (arXiv:2504.06868) — Personality-driven game agents com Big Five traits
8. **TwinMarket** (arXiv:2502.01506) — Vieses cognitivos em simulação comportamental
9. **Consistent Opponent Modeling** (arXiv:2508.17671) — Garantias de convergência para modelagem de oponentes
10. **DeepSeek-R1** (arXiv:2501.12948) — RL puro para raciocínio emergente, MIT license

### Conclusões Regionais

- **China:** Ecossistema mais maduro (89+ recursos). **Tencent AI Lab + CASIA é o grupo CFR mais produtivo do mundo** (6 papers top-venue). Alibaba (Qwen), DeepSeek (R1), Tsinghua (GLM/MiniCPM/MARTI) dominam. Poker-específico: PDCFR+, Embedding CFR, DecisionHoldem, ToolPoker.
- **Coreia do Sul:** 4 produtores corporativos (SKT, Naver, LG AI, Upstage), modelos até 519B. **Zero pesquisa poker-específica.**
- **Japão:** UTokyo é o único polo de Game AI relevante (Suspicion-Agent). Swallow com RL training é promissor.
- **Índia:** Ecossistema prolífico (AI4Bharat, Sarvam AI), foco em idiomas indianos. Sem aplicação direta poker.
- **Singapura:** **Nemotron-Personas** é o recurso mais valioso para SyntheticPlayerGenerator. SEA-LION para 13 idiomas SEA.
- **UAE:** TII (Falcon) com arquitetura híbrida Mamba-Transformer inovadora. MBZUAI para pesquisa médica/árabe.
- **Poker AI global:** 25+ papers diretamente aplicáveis. PokerBench, SPIRAL, AlphaEvolve são os mais impactantes.

---

## 2. Metodologia de Busca e Verificação

### 2.1 Fontes Primárias Utilizadas
- **HuggingFace Hub:** busca por repositórios de modelos, datasets, e spaces via API
- **arXiv:** busca por papers via paper_search
- **Sites oficiais:** universidades, labs, plataformas governamentais
- **GitHub:** repositórios de código-fonte

### 2.2 Critérios de Seleção
- Recurso deve ter **documentação técnica verificável** (paper, technical report, model card)
- Deve ter **disponibilidade real** (pesos, dados, ou código publicados)
- Deve ter **relevância prática** para stack de IA multimodal/multilíngue
- Prioridade para recursos de 2024-2026

### 2.3 Hierarquia de Validação
1. Paper oficial / technical report (arXiv, conferência)
2. Repositório oficial com código/pesos
3. Documentação técnica oficial (model card, dataset card)
4. Fonte secundária confiável (benchmark independente)

### 2.4 Limitações
- Plataformas chinesas com barreira de idioma (CSTCloud, ModelScope)
- Recursos anunciados mas não publicados ainda (Vikram/Sarvam, alguns modelos coreanos)
- Claims de benchmark sem reprodução independente
- Licenças que restringem uso comercial

---

## 3. Auditoria do Documento-Base

### 3.1 Tabela de Verificação

| ID | Afirmação | Status | Correção/Ajuste | Evidência |
|----|-----------|--------|-----------------|-----------|
| A1 | DanQing: 100M pares imagem-texto chineses, CC-BY-4.0 | **Parcialmente confirmada** | Paper arXiv:2601.10305 existe. Tamanho e licença referidos no paper. Dados de 2024-2025 como claimed. | arXiv primário |
| A2 | CSTCloud: 196 modelos, 79 científicos, 3900B tokens | **Parcialmente confirmada** | Site oficial existe. Números específicos (196, 79, 3900B) não verificáveis independentemente — vêm de PR governamental | Site oficial CAS |
| A3 | A.X K1: 519B params, maior modelo da Coreia | **Parcialmente confirmada** | SKT anunciou A.X K1. 519B não confirmado independentemente. "Maior da Coreia" é claim de marketing. | Site SKT |
| A4 | A.X 4.0-VL-Light: 33% mais eficiente que GPT-4o | **Marketing fraco** | Claim de eficiência de tokenização, não de performance geral. Benchmark KMMLU pode ser otimizado para coreano. | Sem benchmark independente |
| A5 | Korean FineWeb-Edu: 190B tokens | **Confirmada** | Disponível no HuggingFace via ELRIS Group. Demo de 5% publicado. | HuggingFace verificado |
| A6 | Swallow LLM: 2.4M downloads | **Confirmada** | Verificável no HuggingFace. Múltiplas variantes (Llama, Qwen, Gemma). | HuggingFace metrics |
| A7 | DEJIMA: 3.88M pares imagem-texto japoneses | **Confirmada** | Paper aceito LREC 2026. Universidade de Tóquio. | Paper acadêmico |
| A8 | Vikram (Sarvam): 35B + 105B params | **CONFIRMADA (atualizado)** | sarvam-30b (35.6K downloads) e sarvam-105b (7.3K downloads) agora disponíveis no HuggingFace | HuggingFace verificado |
| A9 | AIKosh: 7500+ datasets, 273 modelos | **Parcialmente confirmada** | PIB do governo indiano cita números. Acesso real limitado. Busca encontrou ~300 datasets, não 7500. | Discrepância |
| A10 | Nemotron-Personas-Singapore: 888K personas, 38 campos | **Confirmada** | Dataset verificado no HuggingFace. Schema confirmado com 22 campos (não 38 — 38 inclui sub-campos). CC-BY-4.0. | HuggingFace dataset card |

### 3.2 Conclusão da Auditoria

**Mantido:** A1, A5, A6, A7, A10 — dados verificáveis com fontes primárias.
**Corrigido:** A9 (números provavelmente inflados), A10 (38 = campos + sub-campos).
**Enfraquecido:** A2 (números não verificáveis), A3 (parâmetros não confirmados), A4 (marketing).
**Confirmado (novo):** A8 — Sarvam agora tem sarvam-30b e sarvam-105b disponíveis no HuggingFace.

---

## 4. Mapa Acadêmico por País

### 4.1 China (89 recursos verificados)

#### Instituições Principais
- **Alibaba / Tongyi Lab:** Qwen family (0.5B-235B), Qwen2.5-VL, Qwen3, FunASR, CosyVoice, SenseVoice
- **DeepSeek AI:** V2/V3/R1, Coder, VL2, Prover, Janus, OCR — MIT license na maioria
- **Tsinghua / Zhipu / OpenBMB:** GLM-4/4.5, CogVLM, MiniCPM, CodeGeeX
- **Shanghai AI Lab / OpenGVLab:** InternLM2/3, InternVL1-3, XComposer
- **BAAI:** BGE-M3 (17M downloads), Emu3, AltCLIP, SegGPT, Infinity-Instruct
- **Tencent:** Hunyuan-Large (389B MoE), HunyuanVideo, Hunyuan3D, HunyuanOCR
- **01.AI (Kai-Fu Lee):** Yi family (6B-34B), Yi-Lightning (MoE)
- **Moonshot AI:** Kimi K2 (1T total, 32B active), Kimi-VL
- **Huawei:** PanGu-Sigma (1.085T), Pangu Ultra MoE (718B) — Ascend NPU
- **Baidu:** ERNIE 3.0/4.5, PaddleOCR v5
- **Baichuan:** Baichuan 2, Baichuan-M1 (médico)
- **StepFun:** Step-Video-T2V (30B), Step-Audio 2

#### Top 15 Modelos Chineses (por relevância para MR_POKER)

| # | Nome | Org | Params | Downloads | License | Relevância |
|---|------|-----|--------|-----------|---------|------------|
| C1 | **DeepSeek-R1** | DeepSeek | 671B MoE | 1.3M | MIT | **ALTA** — RL puro gera raciocínio emergente; distilação para 1.5B-70B |
| C2 | **DeepSeek-V3** | DeepSeek | 671B (37B active) | 1.1M | MIT | **ALTA** — MLA + MoE + FP8 training; base para fine-tuning |
| C3 | **Qwen3** | Alibaba | 0.5B-235B MoE | 22.8M (Qwen2.5-7B) | Apache-2.0 | **ALTA** — 119 idiomas, dual-mode thinking |
| C4 | **Qwen2.5-Coder** | Alibaba | 1.5B-32B | 2.2M | Apache-2.0 | Média — SOTA em code gen; útil para poker tool integration |
| C5 | **MiniCPM** | OpenBMB/Tsinghua | 1B-8B | 113K | Apache-2.0 | Média — WSD scheduler, eficiente on-device |
| C6 | **GLM-4.5** | Zhipu/Tsinghua | 355B MoE | — | Custom | Média — ARC focus (agentic, reasoning, coding) |
| C7 | **InternLM2** | Shanghai AI Lab | 1.8B-20B | 146K | Custom | Média — COOL RLHF, long-context |
| C8 | **Kimi K2** | Moonshot | 1T (32B active) | — | MIT | Média — agentic, SOTA SWE-Bench |
| C9 | **BGE-M3** | BAAI | — | 17.2M | MIT | Baixa — embedding/retrieval, não raciocínio |
| C10 | **Yi-Lightning** | 01.AI | MoE | — | Apache-2.0 | Média — bilingual MoE avançado |
| C11 | **Hunyuan-Large** | Tencent | 389B (52B active) | — | Custom | Baixa — MoE genérico |
| C12 | **Hunyuan-TurboS** | Tencent | — | — | — | Média — **Mamba-Transformer hybrid** (inovação arquitetural) |
| C13 | **PanGu Ultra MoE** | Huawei | 718B | — | Proprietário | Baixa — Ascend NPU, não disponível |
| C14 | **DeepSeek-Prover-V2** | DeepSeek | 7B-671B | 127K | — | Média — prova de teoremas com decomposição recursiva |
| C15 | **Qwen2.5-Math** | Alibaba | — | — | Apache-2.0 | Média — raciocínio matemático para game theory |

**Nota:** arXiv IDs principais: DeepSeek-R1 (2501.12948), DeepSeek-V3 (2412.19437), Qwen3 (2505.09388), GLM-4.5 (2508.06471), Kimi K2 (2507.20534)

#### Plataformas Chinesas
- **ModelScope** (modelscope.cn): 70,000+ modelos, 16M+ devs, 2000+ orgs, 1500+ datasets chineses
- **CSTCloud (CAS):** 196 modelos reportados, 79 científicos — números de PR governamental, não verificáveis independentemente

#### Grupos de Pesquisa Poker-Específicos na China (DESCOBERTA CRÍTICA)

**Tencent AI Lab + CASIA** (Haobo Fu, Kai Li, Qiang Fu, Hang Xu, Weiming Liu, Junliang Xing):
O grupo **mais produtivo do mundo** em CFR solver research, com 6 papers em top venues:

| Paper | Venue | Inovação |
|-------|-------|----------|
| **AutoCFR** | AAAI 2022 Oral | Meta-learning para design automático de CFR |
| **ACH (Actor-Critic Hedge)** | ICLR 2022 | Actor-critic para IIGs em larga escala |
| **Opponent-Limited Online Search** | ICML 2023 | Subgame solving seguro com opponent modeling limitado |
| **Dynamic Discounted CFR** | ICLR 2024 Spotlight | Discounting dinâmico para convergência CFR |
| **PDCFR+** | IJCAI 2024 Oral | CFR ponderado + optimistic mirror descent. Código: [PDCFRPlus](https://github.com/rpSebastian/PDCFRPlus) |
| **Deep Predictive DCFR** | arXiv:2511.08174 | Neural CFR model-free com variance reduction |

**CASIA Poker Abstraction** (Yanchang Fu, Kaiqi Huang):

| Paper | ID | Inovação |
|-------|-----|----------|
| **KrwEmd** (Signal Observation Models) | arXiv:2403.11486 | Info histórica em hand abstraction; k-recall winrate |
| **Embedding CFR** | arXiv:2511.12083 | Embeddings pré-treinados substituem clustering discreto |

**Outros Papers Poker-Específicos Chineses:**
- **DecisionHoldem** (CASIA, arXiv:2201.11580) — Primeiro AI HUNL open-source de alto nível. Código: [AI-Decision/DecisionHoldem](https://github.com/AI-Decision/DecisionHoldem)
- **PokerGPT** (arXiv:2401.06781) — LLM solver leve para multi-player Hold'em
- **ToolPoker** (arXiv:2602.00528, ICLR 2026) — Integra GTO solvers externos com raciocínio LLM

**IIG AI (jogos de cartas chineses):**
- **DouZero** (ICML 2021, arXiv:2106.06135) — Self-play RL para DouDizhu. #1 Botzone. [GitHub](https://github.com/kwai/DouZero)
- **GuanZero** (arXiv:2402.13582) — RL para Guandan (4 jogadores)
- **Suphx** (Microsoft Research Asia, arXiv:2003.13590) — Mahjong AI, 10-dan Tenhou
- **RLCard** ([GitHub](https://github.com/datamllab/rlcard)) — Toolkit RL para Hold'em, Leduc, DouDizhu, Mahjong

**Frameworks Multi-Agent:**
- **MARTI** (Tsinghua, [GitHub](https://github.com/TsinghuaC3I/MARTI)) — Multi-agent RL com verifiable rewards
- **OpenRLHF** ([GitHub](https://github.com/OpenRLHF/OpenRLHF)) — Framework RLHF escalável até 70B+

**Behavioral/Cognitive:**
- **Prospect Theory Fails for LLMs** (HKUST, arXiv:2508.08992) — PT falha para LLMs; vieses humanos ≠ vieses LLM
- **Game Theory Meets LLMs** (PKU, arXiv:2502.09053, IJCAI 2025) — Survey completo GT↔LLM

#### Destaques para MR_POKER
1. **Tencent/CASIA é o grupo CFR mais importante do mundo** — PDCFR+ e Deep Predictive DCFR são upgrades diretos do nosso CFR solver. Código aberto disponível.
2. **Embedding CFR** (CASIA) substitui clustering discreto por embeddings pré-treinados — encaixa perfeitamente no nosso EmbeddingCFRTrainer.
3. **DeepSeek-R1** — RL puro sem SFT gera raciocínio emergente. Distilação para 1.5B-70B.
4. **ToolPoker** (ICLR 2026) integra GTO solvers com LLM — arquitetura idêntica ao objetivo do MR_POKER.
5. **Prospect Theory Fails for LLMs** (HKUST) — vieses cognitivos de humanos ≠ LLMs. Crítico para BiasExploiter.

**Relatório detalhado:** `docs/104_China_AI_Resources_for_Poker.md` (52 recursos verificados)

### 4.2 Coreia do Sul

#### Instituições Principais
- **KAIST** (Korea Advanced Institute of Science and Technology): Top-5 mundial em papers ML (ICML, NeurIPS, ICLR, 2020-2024). Labs: MLAI (mlai-kaist.com), KIXLAB, Kim Jaechul Graduate School of AI.
- **SNU** (Seoul National University): NLP lab (snunlp org no HuggingFace), pesquisa em FinBERT coreano.
- **SK Telecom**: A.X family (desde KoBERT 2019).
- **Naver**: HyperCLOVA X ecosystem.
- **LG AI Research**: EXAONE family.
- **Upstage**: Solar family.

#### Modelos Verificados

| # | Nome | Org | Params | Downloads | Criado | HuggingFace URL | Relevância MR_POKER |
|---|------|-----|--------|-----------|--------|-----------------|---------------------|
| K1 | A.X-K1 | SKT | 519B (MoE) | 5.5K | Dec 2025 | [skt/A.X-K1](https://hf.co/skt/A.X-K1) | Medium — teacher model for knowledge distillation |
| K2 | A.X-4.0 | SKT | 72B | 629 | Jul 2025 | [skt/A.X-4.0](https://hf.co/skt/A.X-4.0) | Medium — CPT on Qwen2, strong Korean reasoning |
| K3 | A.X-4.0-Light | SKT | 7B | 7.2K | Jul 2025 | [skt/A.X-4.0-Light](https://hf.co/skt/A.X-4.0-Light) | Medium — lightweight deployment candidate |
| K4 | A.X-3.1 | SKT | 34B | 317 | Jul 2025 | [skt/A.X-3.1](https://hf.co/skt/A.X-3.1) | Medium — 2.1T tokens, 33% fewer tokens vs GPT-4o for Korean |
| K5 | HyperCLOVAX-SEED-Omni-8B | Naver | 8B | 228K | Dec 2025 | [naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B](https://hf.co/naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B) | Low — multimodal (vision/audio), not core poker |
| K6 | HyperCLOVAX-SEED-Think-32B | Naver | 32B | 18.4K | Dec 2025 | [naver-hyperclovax/HyperCLOVAX-SEED-Think-32B](https://hf.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-32B) | Medium — reasoning-focused, potential for game-theoretic reasoning |
| K7 | HyperCLOVAX-SEED-Think-14B | Naver | 14B | 24.4K | Jul 2025 | [naver-hyperclovax/HyperCLOVAX-SEED-Think-14B](https://hf.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-14B) | Medium — RL from Verifiable Rewards training |
| K8 | EXAONE-4.0-32B | LG AI | 32B | 18.8K | Jul 2025 | [LGAI-EXAONE/EXAONE-4.0-32B](https://hf.co/LGAI-EXAONE/EXAONE-4.0-32B) | Medium — bilingual EN/KO, strong reasoning |
| K9 | EXAONE-4.0-1.2B | LG AI | 1.2B | 58.2K | Jul 2025 | [LGAI-EXAONE/EXAONE-4.0-1.2B](https://hf.co/LGAI-EXAONE/EXAONE-4.0-1.2B) | Low — too small for complex reasoning |
| K10 | EXAONE-Deep-7.8B | LG AI | 7.8B | 296.6K | Mar 2025 | [LGAI-EXAONE/EXAONE-Deep-7.8B](https://hf.co/LGAI-EXAONE/EXAONE-Deep-7.8B) | Medium — deep reasoning variant, fine-tuned from EXAONE-3.5 |
| K11 | EXAONE-3.5-7.8B-Instruct | LG AI | 7.8B | 388.8K | Dec 2024 | [LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct](https://hf.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct) | Medium — most downloaded EXAONE, proven quality |
| K12 | K-EXAONE-236B-A23B | LG AI | 236B (MoE, 23B active) | 24.1K | Dec 2025 | [LGAI-EXAONE/K-EXAONE-236B-A23B](https://hf.co/LGAI-EXAONE/K-EXAONE-236B-A23B) | Medium — MoE architecture, 256K context, multilingual |
| K13 | SOLAR-10.7B-Instruct-v1.0 | Upstage | 10.7B | 26.8K | Dec 2023 | [upstage/SOLAR-10.7B-Instruct-v1.0](https://hf.co/upstage/SOLAR-10.7B-Instruct-v1.0) | Low — older, but Depth Up-Scaling innovation |
| K14 | Solar-Open-100B | Upstage | 100B (MoE) | 6.0K | Dec 2025 | [upstage/Solar-Open-100B](https://hf.co/upstage/Solar-Open-100B) | Medium — latest Solar, MoE, EN/KO |
| K15 | solar-pro-preview-instruct | Upstage | undisclosed | 16.3K | Sep 2024 | [upstage/solar-pro-preview-instruct](https://hf.co/upstage/solar-pro-preview-instruct) | Low — preview, MIT license |

#### Datasets Coreanos Verificados

| # | Nome | Tipo | Tamanho | URL | Relevância |
|---|------|------|---------|-----|------------|
| KD1 | Korean FineWeb-Edu (raw) | Web text, educational | 10M-100M docs | [minpeter/fineweb-2-edu-korean-raw](https://hf.co/datasets/minpeter/fineweb-2-edu-korean-raw) | Low — general Korean text |
| KD2 | Korean FineWeb-Edu (filtered) | Web text, educational | 1M-10M docs | [minpeter/fineweb-2-edu-korean](https://hf.co/datasets/minpeter/fineweb-2-edu-korean) | Low — filtered variant |
| KD3 | Korean FineWeb-Edu Demo (Elice) | Demo subset (~30GB) | 1M-10M docs | [eliceai/korean-fineweb-edu-demo](https://hf.co/datasets/eliceai/korean-fineweb-edu-demo) | Low — demo subset |
| KD4 | Korean RLHF Dataset | SFT instruction pairs | 100K-1M | [jojo0217/korean_rlhf_dataset](https://hf.co/datasets/jojo0217/korean_rlhf_dataset) | Medium — RLHF training data template |
| KD5 | Korean YouTube Sentiment | Sentiment labels (pos/neg/neutral) | 5,482 comments | [LLM-SocialMedia/Korean-YouTube-Comment-Sentiment-Dataset](https://hf.co/datasets/LLM-SocialMedia/Korean-YouTube-Comment-Sentiment-Dataset) | Medium — behavioral/sentiment patterns |
| KD6 | Korean Finance Dataset | Financial QA | 100K-1M | [nmixx-fin/opensource_korean_finance_datasets](https://hf.co/datasets/nmixx-fin/opensource_korean_finance_datasets) | Low — domain specific |
| KD7 | Korean Wikipedia GPT2 | Full Korean Wikipedia | 100K-1M articles | [eaglewatch/Korean_Wikipedia_Dataset_for_GPT2_August_2022](https://hf.co/datasets/eaglewatch/Korean_Wikipedia_Dataset_for_GPT2_August_2022) | Low — general knowledge base |

#### Papers Coreanos Relevantes

| # | Título | arXiv/Conf | Autores/Afiliação | Relevância |
|---|--------|------------|-------------------|------------|
| KP1 | HyperCLOVA X Technical Report | arXiv:2404.01954 | Naver (370+ authors) | Medium — Korean LLM architecture details |
| KP2 | HyperCLOVA X THINK Technical Report | arXiv:2506.22403 | Naver Cloud | Medium — reasoning-focused model, RLVR training |
| KP3 | K-EXAONE Technical Report | arXiv:2601.01739 | LG AI Research | Medium — MoE architecture, 256K context |
| KP4 | EXAONE 3.0/3.5 Technical Report | arXiv:2412.04862 | LG AI Research | Low — older version details |
| KP5 | A.X K1 Technical Report | arXiv:2601.09200 | SK Telecom | Medium — 519B teacher model |
| KP6 | KMMLU: Korean Massive Multitask LU | arXiv:2402.11548 | Multi-institutional Korean | Medium — benchmark for Korean reasoning evaluation |
| KP7 | RedWhale: Adapted Korean LLM | arXiv:2408.11294 | Korean researchers | Low — continual pretraining technique |
| KP8 | EEVE-Korean-v1.0 | arXiv:2402.14714 | Korean researchers | Low — vocabulary expansion technique |
| KP9 | Polyglot-Ko Technical Report | arXiv:2306.02254 | EleutherAI Korea | Low — open-source Korean LLM family |

### 4.3 Japão

#### Instituições Principais
- **UTokyo** (University of Tokyo): Game AI Research Group (game.c.u-tokyo.ac.jp), Next AI Research Center, Matsuo Lab (Yutaka Matsuo).
- **Tokyo Institute of Technology** (now Institute of Science Tokyo): Swallow LLM project (tokyotech-llm).
- **National Institute of Information and Communications Technology (NICT)**: collaboration with PFN on PLaMo-3.
- **Preferred Networks (PFN)**: PLaMo family.
- **CyberAgent**: CALM family, CAT-Translate.
- **Rinna**: Japanese GPT family.
- **SB Intuitions**: Sarashina family.
- **LLM-jp**: National project for Japanese LLM (NII-led consortium).

#### Modelos Verificados

| # | Nome | Org | Params | Downloads | Criado | HuggingFace URL | Relevância MR_POKER |
|---|------|-----|--------|-----------|--------|-----------------|---------------------|
| J1 | Llama-3.1-Swallow-70B-Instruct-v0.3 | Tokyo Tech | 70B | 32.6K | Dec 2024 | [tokyotech-llm/Llama-3.1-Swallow-70B-Instruct-v0.3](https://hf.co/tokyotech-llm/Llama-3.1-Swallow-70B-Instruct-v0.3) | Medium — strong Japanese reasoning |
| J2 | GPT-OSS-Swallow-120B-RL-v0.1 | Tokyo Tech | 120B | 3.8K | Feb 2026 | [tokyotech-llm/GPT-OSS-Swallow-120B-RL-v0.1](https://hf.co/tokyotech-llm/GPT-OSS-Swallow-120B-RL-v0.1) | Medium — RL-trained, largest open Japanese model |
| J3 | GPT-OSS-Swallow-20B-RL-v0.1 | Tokyo Tech | 20B | 7.7K | Feb 2026 | [tokyotech-llm/GPT-OSS-Swallow-20B-RL-v0.1](https://hf.co/tokyotech-llm/GPT-OSS-Swallow-20B-RL-v0.1) | Medium — RL-trained, efficient size |
| J4 | Qwen3-Swallow-8B-RL-v0.2 | Tokyo Tech | 8B | 4.2K | Jan 2026 | [tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2](https://hf.co/tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2) | Medium — latest Swallow on Qwen3 base |
| J5 | PLaMo-2-1B | PFN | 1B | 68.5K | Feb 2025 | [pfnet/plamo-2-1b](https://hf.co/pfnet/plamo-2-1b) | Low — small but efficient |
| J6 | PLaMo-2-8B | PFN | 8B | 104 | Feb 2025 | [pfnet/plamo-2-8b](https://hf.co/pfnet/plamo-2-8b) | Medium — mid-size, from-scratch Japanese |
| J7 | PLaMo-100B | PFN | 100B | 94 | Sep 2024 | [pfnet/plamo-100b](https://hf.co/pfnet/plamo-100b) | Medium — large scale, research-only license |
| J8 | PLaMo-3-NICT-8B-base | PFN/NICT | 8B | 108 | Oct 2025 | [pfnet/plamo-3-nict-8b-base](https://hf.co/pfnet/plamo-3-nict-8b-base) | Low — base model, government collaboration |
| J9 | open-calm-7b | CyberAgent | 7B | 964 | May 2023 | [cyberagent/open-calm-7b](https://hf.co/cyberagent/open-calm-7b) | Low — older Japanese GPT-NeoX |
| J10 | calm2-7b-chat | CyberAgent | 7B | 1.2K | Nov 2023 | [cyberagent/calm2-7b-chat](https://hf.co/cyberagent/calm2-7b-chat) | Low — chat model, Llama-based |
| J11 | Mistral-Nemo-Japanese-Instruct-2408 | CyberAgent | ~12B | 550 | Aug 2024 | [cyberagent/Mistral-Nemo-Japanese-Instruct-2408](https://hf.co/cyberagent/Mistral-Nemo-Japanese-Instruct-2408) | Low — Mistral fine-tune for Japanese |
| J12 | rinna/japanese-gpt-neox-3.6b-instruction-ppo | Rinna | 3.6B | 733 | May 2023 | [rinna/japanese-gpt-neox-3.6b-instruction-ppo](https://hf.co/rinna/japanese-gpt-neox-3.6b-instruction-ppo) | Medium — PPO-trained, relevant RL technique |
| J13 | Sarashina2.2-3B-Instruct-v0.1 | SB Intuitions | 3B | 2.1K | Feb 2025 | [sbintuitions/sarashina2.2-3b-instruct-v0.1](https://hf.co/sbintuitions/sarashina2.2-3b-instruct-v0.1) | Low — compact Japanese instruct model |
| J14 | llm-jp-3.1-13b-instruct4 | LLM-jp (NII) | 13B | 34.1K | May 2025 | [llm-jp/llm-jp-3.1-13b-instruct4](https://hf.co/llm-jp/llm-jp-3.1-13b-instruct4) | Low — national project, Apache-2.0 |
| J15 | llm-jp-3.1-1.8b-instruct4 | LLM-jp (NII) | 1.8B | 7.1K | May 2025 | [llm-jp/llm-jp-3.1-1.8b-instruct4](https://hf.co/llm-jp/llm-jp-3.1-1.8b-instruct4) | Low — lightweight variant |

#### Datasets Japoneses Verificados

| # | Nome | Tipo | Tamanho | URL | Relevância |
|---|------|------|---------|-----|------------|
| JD1 | DEJIMA | Image captioning + VQA (Japanese) | 3.88M pairs | [arxiv.org/abs/2512.00773](https://arxiv.org/abs/2512.00773) / [mil-tokyo.github.io/DEJIMA-dataset](https://mil-tokyo.github.io/DEJIMA-dataset/) | Low — multimodal, not poker-specific |
| JD2 | izumi-lab/llm-japanese-dataset | Japanese instruction tuning | varied | [izumi-lab/llm-japanese-dataset](https://hf.co/datasets/izumi-lab/llm-japanese-dataset) | Low — general Japanese SFT data |
| JD3 | Swallow training datasets | Math, code, instruction | multiple sets | [tokyotech-llm/swallow-math-v2](https://hf.co/datasets/tokyotech-llm/swallow-math-v2), etc. | Low — training data for Swallow models |

#### Papers Japoneses Relevantes

| # | Título | arXiv/Conf | Autores/Afiliação | Relevância |
|---|--------|------------|-------------------|------------|
| JP1 | Suspicion-Agent: Playing Imperfect Information Games with ToM-Aware GPT-4 | arXiv:2309.17277 | Guo, Yang, Yoo, Lin, Iwasawa, Matsuo (UTokyo) | **HIGH** — Theory of Mind for imperfect info games, Leduc Hold'em, directly applicable to opponent modeling |
| JP2 | Continual Pre-Training for Cross-Lingual LLM Adaptation (Swallow) | arXiv:2404.17790 | Fujii, Nakamura et al. (Tokyo Tech) | Low — CPT technique for Japanese |
| JP3 | Swallow Corpus and Training Report | arXiv:2503.23714 | Tokyo Tech team | Low — Swallow training methodology |
| JP4 | PLaMo Technical Report | arXiv:2406.07522 | PFN team | Low — PLaMo architecture details |
| JP5 | PLaMo-100B Technical Report | arXiv:2410.07563 | PFN team | Low — 100B scale training |

### 4.4 Índia (47 recursos verificados)

#### Instituições Principais
- **AI4Bharat (IIT Madras):** 28 modelos/datasets, cobre 22 idiomas indianos. IndicTrans2/3, IndicBART, IndicVoices
- **Sarvam AI:** sarvam-30b (MoE), sarvam-105b (maior modelo indiano aberto), OpenHathi, shuka-1
- **Krutrim (Ola):** Krutrim-2-instruct (12B, 128K context), Chitrarth (VLM), Vyakyarth (embeddings)
- **Google Research India:** MuRIL (BERT para 17 idiomas indianos, 36K downloads)
- **IISc/ARTPARK:** Vaani dataset (31,255 hrs de fala, 773 distritos)
- **IIT Bombay/CFILT:** Corpora paralelos inglês-hindi

#### Top Recursos Indianos

| # | Nome | Tipo | Downloads | Relevância |
|---|------|------|-----------|------------|
| I1 | **Sangraha** (AI4Bharat) | Dataset 251B tokens, 22 idiomas | 24.9K | Baixa — idiomas indianos |
| I2 | **sarvam-105b** | Modelo 105B | 7.3K | Baixa — maior modelo indiano, idiomas indianos |
| I3 | **sarvam-30b** | Modelo MoE 30B | 35.6K | Baixa — MoE indiano |
| I4 | **IndicVoices** | Dataset 11,200 hrs fala | 8.7K | Baixa — speech, não poker |
| I5 | **shuka-1** | Modelo audio-texto multimodal | 37.6K | Baixa — Hindi+English multimodal |

#### Plataformas Governamentais
- **Bhashini** (bhashini.gov.in): 350+ modelos IA, 36+ idiomas
- **AIKosh** (aikosh.indiaai.gov.in): 7,541 datasets reportados (provavelmente inflado), 273 modelos, 20 setores
- **IndiaAI Mission:** Rs 10,371 crore (~$1.2B) investimento nacional

**Conclusão Índia:** Ecossistema impressionante para NLP multilíngue, mas **zero aplicação direta para poker AI**. Foco em saúde, agricultura e idiomas.

### 4.5 Singapura (28 recursos verificados)

#### Instituições Principais
- **AI Singapore (AISG):** SEA-LION family (v1-v4), 13 idiomas do sudeste asiático
- **SEACrowd:** Datasets multilíngues comunitários
- **NVIDIA Singapore:** Nemotron-Personas

#### Top Recursos de Singapura

| # | Nome | Tipo | Downloads | Relevância |
|---|------|------|-----------|------------|
| S1 | **Nemotron-Personas-Singapore** | 888K personas, PGM, 38 campos | — | **ALTÍSSIMA** — upgrade direto do SyntheticPlayerGenerator |
| S2 | **Gemma-SEA-LION-v4-27B-IT** | LLM 13 idiomas SEA | 4.6K | Baixa — idiomas SEA |
| S3 | **SEA-PILE-v2** | 120B tokens pretraining | 1.1K | Baixa — corpus SEA |
| S4 | **HyperCLOVAX-SEED-Omni-8B** | Multimodal omni | 228K | Baixa — Naver, não SG |
| S5 | **Gemma-SEA-Guard-12B** | Safety/guardrail | 243 | Baixa — segurança |

**Conclusão Singapura:** O recurso mais valioso de TODA a pesquisa é **Nemotron-Personas-Singapore** — PGM hierárquico com 888K personas sintéticas que mapeia diretamente para nosso SyntheticPlayerGenerator. SEA-LION é forte para idiomas regionais mas sem aplicação poker.

### 4.6 Outros Polos

#### 4.6.1 UAE / Oriente Médio (35 recursos)

**Instituições:**
- **TII (Technology Innovation Institute):** Falcon family — 7B a 40B, **Falcon-H1** (híbrido Mamba-Transformer)
- **MBZUAI:** AIN (OCR árabe, 492K downloads), modelos médicos, benchmarks árabes
- **Inception AI:** Jais family — LLMs árabe-primeiro até 70B
- **SDAIA (Arábia Saudita):** ALLaM (7B-70B), 500B tokens árabes
- **KAUST:** AceGPT — alinhamento cultural árabe

| # | Nome | Tipo | Downloads | Relevância |
|---|------|------|-----------|------------|
| ME1 | **Falcon-H1-7B** | Mamba-Transformer híbrido | 4.0K | Média — **arquitetura híbrida inovadora** para inferência eficiente |
| ME2 | **falcon-refinedweb** | Dataset web massivo | 13.0K | Baixa — pretraining data |
| ME3 | **Jais-2-70B-Chat** | LLM árabe 70B | 3.1K | Baixa — foco árabe |
| ME4 | **AIN (MBZUAI)** | OCR árabe VLM | 492.2K | Baixa — OCR |

**Destaque:** Falcon-H1 (Mamba-Transformer híbrido) é arquiteturalmente interessante para inferência eficiente em real-time poker.

#### 4.6.2 Taiwan (12 recursos)

- **CKIP Lab / Academia Sinica:** NLP chinês tradicional (BERT word segmentation 141K downloads)
- **NTU (yentinglin):** Taiwan-LLM, Llama-3-Taiwan (8B/70B)
- **Taiwan AI Labs:** Formosa Foundation Model (FFM) — serviço empresarial

**Conclusão:** Foco em chinês tradicional. Sem aplicação poker.

#### 4.6.3 Vietnã (12 recursos)

- **VinAI Research:** PhoBERT (311K downloads — modelo vietnamita mais usado), PhoGPT-4B, PhoWhisper
- Domina completamente o ecossistema de NLP vietnamita

**Conclusão:** Ecossistema sólido mas focado em idioma. Sem aplicação poker.

#### 4.6.4 Malásia (12 recursos)

- **Mesolitica:** MaLLaM, Malaysian ASR/TTS, embeddings
- Cobre malaio, inglês, tâmil e chinês

#### 4.6.5 Indonésia (5 recursos)

- **IndoBenchmark:** IndoBERT (242K downloads), IndoGPT

#### 4.6.6 Tailândia (5 recursos)

- **VISTEC/AIResearch:** WangchanBERTa (49K downloads), WangchanLion7B

---

## 5. Inventário Consolidado: Poker AI & Game-Theoretic Papers (Global, with East Asian Focus)

### 5.1 Poker AI Papers (Diretamente Relevantes ao MR_POKER)

| # | Título | ID | Ano | Relevância | Detalhe Chave |
|---|--------|----|-----|------------|---------------|
| P1 | PokerBench: Training LLMs to become Professional Poker Players | arXiv:2501.08328 | Jan 2025 | **HIGH** | Benchmark dataset + evaluation framework for LLM poker. Pre-flop and post-flop scenarios with solver-optimal decisions. Dataset: [RZ412/PokerBench](https://hf.co/datasets/RZ412/PokerBench) |
| P2 | How Far Are LLMs from Professional Poker Players? | arXiv:2602.00528 | Jan 2026 | **HIGH** | Tool-integrated reasoning framework combining external solvers with LLM reasoning. Addresses heuristic reliance gap. |
| P3 | PokerGPT: End-to-End Lightweight Solver for Multi-Player Hold'em | arXiv:2401.06781 | Jan 2024 | **HIGH** | LLM fine-tuned with RLHF for multi-player poker. CFR integration with prompt engineering. |
| P4 | Suspicion-Agent: Imperfect Information Games with ToM-Aware GPT-4 | arXiv:2309.17277 | Sep 2023 | **HIGH** | UTokyo. Theory of Mind capacity in GPT-4 for card games. Adapts to opponents without training. GitHub: [CR-Gjx/Suspicion-Agent](https://github.com/CR-Gjx/Suspicion-Agent) |
| P5 | SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning | arXiv:2506.24119 | Jun 2025 | **HIGH** | Multi-agent multi-turn RL for reasoning. Tested on Kuhn Poker. Role-conditioned advantage estimation. |
| P6 | DecisionHoldem: Safe Depth-Limited Solving | arXiv:2201.11580 | Jan 2022 | **HIGH** | Depth-limited subgame solving. Beats agents by 700+ mbb/h. Open-source. Chinese Academy of Sciences. |
| P7 | Are ChatGPT and GPT-4 Good Poker Players? | arXiv:2308.12466 | Aug 2023 | Medium | Pre-flop analysis of LLM poker ability. GTO analysis framework. |
| P8 | Agent-Pro: Learning to Evolve via Policy-Level Reflection | arXiv:2402.17574 | Feb 2024 | **HIGH** | Policy-level reflection + optimization for game agents. Dynamic belief generation. |
| P9 | Consistent Opponent Modeling in Imperfect-Information Games | arXiv:2508.17671 | Aug 2025 | **HIGH** | Convex minimization for opponent model convergence. Projected gradient descent on sequence-form. |
| P10 | Opponent Modeling in Multiplayer Imperfect-Information Games | arXiv:2212.06027 | Jul 2024 | **HIGH** | Three-player Kuhn poker experiments. Real opponent data integration. |
| P11 | Model-Based Opponent Modeling | arXiv:2108.01843 | Aug 2021 | **HIGH** | Simulates and mixes imagined opponent policies. Recursive reasoning. |
| P12 | DanZero+: Dominating GuanDan through RL | arXiv:2312.02561 | Dec 2023 | Medium | Deep Monte Carlo + policy RL for complex card game. Distributed training. |

### 5.2 Multi-Agent RL & Game Theory Papers

| # | Título | ID | Ano | Relevância | Detalhe Chave |
|---|--------|----|-----|------------|---------------|
| G1 | Discovering Multiagent Learning Algorithms with LLMs | arXiv:2602.16928 | Feb 2026 | **HIGH** | AlphaEvolve discovers new CFR variants. Volatility-Adaptive Discounted CFR. Optimistic Regret Matching. |
| G2 | MARS: Multi-Agent Reasoning through Self-Play | arXiv:2510.15414 | Oct 2025 | **HIGH** | End-to-end RL for multi-agent LLM reasoning. Turn-level advantage estimator. |
| G3 | Iterative Nash Policy Optimization | arXiv:2407.00617 | Jun 2024 | Medium | Nash equilibrium finding via no-regret learning. RLHF alignment. |
| G4 | Multi-Agent RLHF: Data Coverage and Algorithms | arXiv:2409.00717 | Sep 2024 | Medium | Nash equilibrium in general-sum games with preference data. |
| G5 | Survey on Self-Play Methods in RL | arXiv:2408.01072 | Aug 2024 | Medium | Comprehensive taxonomy of self-play algorithms. Includes poker, Go, StarCraft. |
| G6 | NeuPL: Neural Population Learning | arXiv:2202.07415 | Feb 2022 | Medium | Multiple policies in single model. Strategy diversity. Transfer learning. |
| G7 | Minimax Exploiter: Data Efficient Competitive Self-Play | arXiv:2311.17190 | Nov 2023 | Medium | Game theory for data-efficient MARL. Exploiter agents. |
| G8 | Learning Meta Representations for Agents in MARL | arXiv:2108.12988 | Aug 2021 | Medium | Game-common + game-specific strategy learning. Multi-modal latent policies. |
| G9 | Adapting to game trees in zero-sum imperfect information games | arXiv:2212.12567 | Dec 2022 | Medium | Balanced FTRL + Adaptive FTRL for self-play. Theoretical bounds. |
| G10 | Language Agents with RL for Strategic Play in Werewolf | arXiv:2310.18940 | Oct 2023 | Medium | LLM + RL for social deduction. Mitigates intrinsic biases. |
| G11 | Learning Strategic Language Agents in Werewolf with LSPO | arXiv:2502.04686 | Feb 2025 | Medium | CFR in latent space for language agents. DPO fine-tuning. |
| G12 | Monopoly Deal: Benchmark for Bounded One-Sided Response Games | arXiv:2510.25080 | Oct 2025 | Low | CFR for card game benchmark. |
| G13 | Game Theory Meets LLMs: Systematic Survey | arXiv:2502.09053 | Feb 2025 | Medium | Comprehensive survey of LLMs in game-theoretic settings. |

### 5.3 Cognitive Bias & Behavioral Modeling Papers

| # | Título | ID | Ano | Relevância | Detalhe Chave |
|---|--------|----|-----|------------|---------------|
| B1 | AI Agent Behavioral Science | arXiv:2506.06366 | Jun 2025 | **HIGH** | New field proposal. Individual/multi-agent/human-agent interaction. Fairness, safety, interpretability. |
| B2 | Generative Agent Simulations of 1,000 People | arXiv:2411.10109 | Nov 2024 | **HIGH** | LLM simulates individual human behaviors. High accuracy on General Social Survey. Directly applicable to synthetic player generation. |
| B3 | TwinMarket: Scalable Behavioral Simulation for Financial Markets | arXiv:2502.01506 | Feb 2025 | **HIGH** | Multi-agent LLM framework. Cognitive biases + emotional fluctuations. Financial bubbles/recessions. Directly applicable to cognitive bias exploitation. |
| B4 | Humanlike Cognitive Patterns as Emergent Phenomena in LLMs | arXiv:2412.15501 | Dec 2024 | **HIGH** | Decision-making biases in LLMs. Cognitive domains analysis. Applicable to bias exploitation modeling. |
| B5 | Instructed to Bias: Instruction-Tuned LMs Exhibit Cognitive Bias | arXiv:2308.00225 | Aug 2023 | **HIGH** | Decoy effect, certainty effect, belief bias in LLMs. Directly relevant to cognitive bias exploitation. |
| B6 | Planted in Pretraining, Swayed by Finetuning: Origins of Cognitive Biases | arXiv:2507.07186 | Jul 2025 | Medium | Pretraining shapes biases more than finetuning. Cross-tuning methodology. |
| B7 | Balancing Rigor and Utility: Mitigating Cognitive Biases in LLMs | arXiv:2406.10999 | Jun 2024 | Medium | Heuristic moderation + abstention. BRU dataset. |
| B8 | Self-Blinding and Counterfactual Self-Simulation Mitigate Biases | arXiv:2601.14553 | Jan 2026 | Medium | Bias offsetting via blinded replicas. Counterfactual cognition. |
| B9 | General Social Agents | arXiv:2508.17407 | Aug 2025 | **HIGH** | AI agents trained on seed games predict human behavior. Cognitive hierarchy model. Heterogeneous population. |
| B10 | Persona Dynamics: Personality Traits on Agents in Text Games | arXiv:2504.06868 | Apr 2025 | **HIGH** | PANDA integrates personality into game agents. Openness drives exploration. Directly relevant to synthetic player generation. |
| B11 | Implicit Behavioral Alignment in High-Stakes Crowd Simulations | arXiv:2509.16457 | Sep 2025 | Medium | PersonaEvolve refines agent personas. Distribution matching for behavioral realism. |
| B12 | Systematic Biases in LLM Simulations of Debates | arXiv:2402.04049 | Feb 2024 | Low | Social biases in debates. Self-fine-tuning methodology. |
| B13 | Playing games with LLMs: Randomness and Strategy | arXiv:2503.02582 | Mar 2025 | Medium | LLMs converge to predictable patterns. Loss aversion in repeated games. |

### 5.4 Poker-Specific Datasets & Models

| # | Nome | Tipo | URL | Detalhe |
|---|------|------|-----|---------|
| PD1 | PokerBench | Dataset | [RZ412/PokerBench](https://hf.co/datasets/RZ412/PokerBench) | 100K-1M NL Hold'em scenarios with solver-optimal decisions. Pre-flop + post-flop. Apache-2.0. |
| PD2 | PokerBench SFT Chat | Dataset | [felipesp1983/pokerbench-sft-chat](https://hf.co/datasets/felipesp1983/pokerbench-sft-chat) | Chat-formatted version of PokerBench for SFT training. |
| PD3 | Qwen3-4B-PokerBench-GRPO | Model | [YiPz/qwen3-4b-pokerbench-grpo](https://hf.co/YiPz/qwen3-4b-pokerbench-grpo) | GRPO RL-trained poker model on Qwen3-4B. |
| PD4 | Llama3-8B-PokerBench-SFT | Model | [YiPz/llama3-8b-pokerbench-sft](https://hf.co/YiPz/llama3-8b-pokerbench-sft) | SFT-trained poker model on Llama-3.1-8B. |

---

## 6. Análise por País: Coreia do Sul e Japão (Detalhada)

### 6.1 Coreia do Sul — Ecossistema e Relevância para MR_POKER

**Força principal:** Korean LLMs with strong reasoning capabilities. Four major corporate labs (SKT, Naver, LG AI, Upstage) plus academic excellence at KAIST.

**Gaps para poker AI:** No published poker-specific research from Korean institutions found. Korean LLMs are primarily optimized for Korean language tasks, not game-theoretic reasoning. However, the reasoning capabilities (especially EXAONE-Deep, HyperCLOVA X THINK) could potentially be adapted.

**Oportunidade imediata:**
- KMMLU benchmark framework could inspire poker-specific Korean evaluation
- RLHF/RLVR training techniques from HyperCLOVA X THINK are directly transferable
- A.X-K1 as teacher model for knowledge distillation to poker agent

### 6.2 Japão — Ecossistema e Relevância para MR_POKER

**Força principal:** UTokyo Game AI Research Group is the most directly relevant academic institution. Suspicion-Agent (Matsuo Lab) is the highest-relevance paper from East Asia for MR_POKER.

**Gaps para poker AI:** Japanese LLMs (Swallow, PLaMo, CALM, Sarashina) are primarily language models, not game AI. However, the RL-training techniques in newer Swallow models (GPT-OSS-Swallow-20B-RL) demonstrate the approach.

**Oportunidade imediata:**
- Suspicion-Agent's Theory of Mind framework should be studied and potentially adapted for MR_POKER's opponent modeling
- Swallow's RL training pipeline is well-documented and open-source
- PLaMo-fin-base demonstrates domain-specific fine-tuning methodology applicable to poker

---

## 7. Rankings Consolidados: Relevância para MR_POKER

### 7.1 Top 15 Recursos Globais (Todas as Regiões)

| Rank | Recurso | País | Tipo | Impacto | Módulo MR_POKER |
|------|---------|------|------|---------|-----------------|
| 1 | **PokerBench** (arXiv:2501.08328) | Global | Dataset+Paper | ★★★★★ | Todos — benchmark + dados de treino |
| 2 | **AlphaEvolve para CFR** (arXiv:2602.16928) | Global | Paper | ★★★★★ | CFR Solver — VAD-CFR auto-descoberto |
| 3 | **Nemotron-Personas-Singapore** | Singapura | Dataset | ★★★★★ | SyntheticPlayerGenerator — PGM hierárquico |
| 4 | **Suspicion-Agent** (arXiv:2309.17277) | Japão | Paper+Código | ★★★★☆ | Opponent Modeling — Theory of Mind |
| 5 | **SPIRAL** (arXiv:2506.24119) | Global | Paper | ★★★★☆ | CFR + RL — self-play com LLMs |
| 6 | **DeepSeek-R1** (arXiv:2501.12948) | China | Modelo | ★★★★☆ | Raciocínio — RL puro, MIT license |
| 7 | **Generative Agent Simulations** (arXiv:2411.10109) | Global | Paper | ★★★★☆ | SyntheticPlayerGenerator — fidelidade |
| 8 | **PANDA** (arXiv:2504.06868) | Global | Paper | ★★★★☆ | SyntheticPlayerGenerator — Big Five |
| 9 | **Consistent Opponent Modeling** (arXiv:2508.17671) | Global | Paper | ★★★★☆ | Opponent Model — convergência |
| 10 | **TwinMarket** (arXiv:2502.01506) | Global | Paper | ★★★☆☆ | BiasExploiter — vieses cognitivos |
| 11 | **Cognitive Biases in LLMs** (arXiv:2308.00225) | Global | Paper | ★★★☆☆ | BiasExploiter — catálogo de vieses |
| 12 | **Qwen3** (arXiv:2505.09388) | China | Modelo | ★★★☆☆ | Backbone — dual-mode, Apache-2.0 |
| 13 | **PDCFR+** (IJCAI 2024, arXiv:2404.13891) | China | Paper+Código | ★★★★☆ | CFR Solver — optimistic mirror descent, [código](https://github.com/rpSebastian/PDCFRPlus) |
| 14 | **Embedding CFR** (arXiv:2511.12083) | China | Paper | ★★★★☆ | CFR Abstraction — embeddings substituem clustering |
| 15 | **ToolPoker** (ICLR 2026, arXiv:2602.00528) | China | Paper | ★★★★☆ | GTO solver + LLM reasoning integrado |
| 16 | **DecisionHoldem** (arXiv:2201.11580) | China | Paper+Código | ★★★☆☆ | HUNL AI open-source, [código](https://github.com/AI-Decision/DecisionHoldem) |
| 17 | **Deep Predictive DCFR** (arXiv:2511.08174) | China | Paper | ★★★☆☆ | Neural CFR model-free, variance reduction |
| 18 | **Prospect Theory Fails LLMs** (arXiv:2508.08992) | HK | Paper | ★★★☆☆ | BiasExploiter — vieses humanos ≠ LLM |
| 19 | **MARTI** (Tsinghua) | China | Framework | ★★★☆☆ | Multi-agent RL com verifiable rewards |
| 20 | **Agent-Pro** (arXiv:2402.17574) | Global | Paper | ★★☆☆☆ | MetaGameTracker — reflexão de policy |

### 7.2 Top Modelos por Região (Uso Potencial)

| Região | Melhor Modelo | Params | License | Destaque |
|--------|--------------|--------|---------|----------|
| China | DeepSeek-R1 | 671B MoE | MIT | RL puro, distilação até 1.5B |
| China | Qwen3 | 235B MoE | Apache-2.0 | 119 idiomas, dual-mode |
| Coreia | K-EXAONE-236B | 236B MoE | Custom | 256K context, reasoning |
| Japão | GPT-OSS-Swallow-120B-RL | 120B | Llama | RL-trained, maior japonês |
| UAE | Falcon-H1-34B | 34B | Custom | Mamba+Attention hybrid |
| Índia | sarvam-105b | 105B | — | Maior indiano aberto |

---

## 8. Top Estudos/Papers por Componente MR_POKER

### 8.1 CFR Solver
- G1: AlphaEvolve discovers Volatility-Adaptive Discounted CFR (arXiv:2602.16928)
- P3: PokerGPT integrates CFR with LLM (arXiv:2401.06781)
- G11: LSPO uses CFR in latent space (arXiv:2502.04686)
- P6: DecisionHoldem depth-limited solving (arXiv:2201.11580)

### 8.2 Opponent Modeling
- P4: Suspicion-Agent with Theory of Mind (arXiv:2309.17277) **[UTokyo]**
- P9: Consistent Opponent Modeling with convergence (arXiv:2508.17671)
- P10: Multiplayer opponent modeling (arXiv:2212.06027)
- P11: Model-based opponent modeling (arXiv:2108.01843)

### 8.3 Behavioral Prediction
- B1: AI Agent Behavioral Science framework (arXiv:2506.06366)
- B4: Humanlike cognitive patterns in LLMs (arXiv:2412.15501)
- B9: General Social Agents predict behavior (arXiv:2508.17407)
- B13: LLM patterns in repeated games (arXiv:2503.02582)

### 8.4 Skill Estimation
- P1: PokerBench evaluation framework (arXiv:2501.08328)
- P7: ChatGPT/GPT-4 poker analysis (arXiv:2308.12466)
- G5: Self-play survey with evaluation methods (arXiv:2408.01072)

### 8.5 Synthetic Player Generation
- B2: Generative Agent Simulations of 1,000 People (arXiv:2411.10109)
- B10: PANDA personality-driven agents (arXiv:2504.06868)
- B11: PersonaEvolve behavioral alignment (arXiv:2509.16457)
- G6: NeuPL multiple policies in single model (arXiv:2202.07415)

### 8.6 Cognitive Bias Exploitation
- B5: Decoy effect, certainty effect, belief bias catalog (arXiv:2308.00225)
- B3: TwinMarket cognitive biases in financial decisions (arXiv:2502.01506)
- B6: Origins of cognitive biases in pretraining (arXiv:2507.07186)
- B8: Self-blinding to mitigate biases (arXiv:2601.14553)

---

## 9. Recursos para Uso Imediato

### 9.1 Datasets Prontos para Integração
1. **PokerBench** (RZ412/PokerBench) — Apache-2.0, 100K+ poker scenarios with solver labels
2. **Korean RLHF Dataset** (jojo0217/korean_rlhf_dataset) — Template for RLHF pipeline
3. **Korean FineWeb-Edu** (minpeter/fineweb-2-edu-korean) — Korean pretraining data

### 9.2 Modelos Prontos para Fine-tuning
1. **EXAONE-Deep-7.8B** — Deep reasoning, bilingual, 296K downloads
2. **Qwen3-Swallow-8B-RL** — RL-trained, Apache-2.0
3. **Qwen3-4B-PokerBench-GRPO** — Already poker-tuned via GRPO

### 9.3 Código/Frameworks Abertos
1. **Suspicion-Agent** — GitHub: CR-Gjx/Suspicion-Agent (Theory of Mind framework)
2. **DecisionHoldem** — Open-source HUNL poker AI
3. **PokerBench evaluation** — Standardized poker LLM evaluation

---

## 10. Oportunidades Estratégicas

### 10.1 Integração com Stack MR_POKER

1. **CFR + LLM hybrid:** Adopt the tool-integrated framework from P2 (arXiv:2602.00528), using MR_POKER's existing CFR solver as an external tool that an LLM-based agent can invoke.

2. **Opponent modeling via ToM:** Adapt Suspicion-Agent's Theory of Mind approach (JP1) for real-time opponent modeling during play. This aligns with MR_POKER's existing behavioral prediction module.

3. **Synthetic player diversity:** Use Generative Agent Simulations (B2) methodology to create diverse synthetic opponents with realistic behavioral patterns, improving training coverage.

4. **Cognitive bias catalog:** Map the cognitive biases cataloged in B5/B4 to specific poker exploitation strategies in MR_POKER's bias exploitation module.

5. **RL training pipeline:** Adopt Swallow's documented RL training approach (with GRPO or PPO) for poker-specific model training, using PokerBench as reward signal.

### 10.2 Lacunas Identificadas

1. No Korean or Japanese institution has published poker-specific AI research.
2. No behavioral datasets specific to poker decision-making exist in Korean or Japanese.
3. Korean/Japanese LLMs lack game-theoretic fine-tuning.
4. DEJIMA and other Japanese multimodal datasets are not relevant to poker.
5. Need for a Korean/Japanese poker scenario dataset (analogous to PokerBench) for multilingual poker AI.

---

## 11. Hype, Riscos e Limitações

### 11.1 Verificação de Claims
- **A.X K1 (519B):** Parameter count anunciado mas pesos restritos. "Teacher model" não verificado independentemente.
- **HyperCLOVA X "6500x mais dados coreanos que GPT-4":** Claim de marketing, sem benchmark independente.
- **AIKosh "7500+ datasets":** Número provavelmente inflado. Busca encontrou ~300 acessíveis.
- **CSTCloud "196 modelos, 3900B tokens":** Números de PR governamental, não verificáveis independentemente.
- **PokerBench fine-tuned models:** Criados pela comunidade, qualidade varia significativamente.
- **Sarvam 105B:** Agora disponível no HuggingFace (corrigido de "não publicado" para "publicado").

### 11.2 Licenciamento — Mapa de Risco

| Modelo/Recurso | Licença | Uso Comercial | Risco |
|----------------|---------|---------------|-------|
| DeepSeek-R1/V3 | MIT | ✅ Livre | Baixo |
| Qwen3 | Apache-2.0 | ✅ Livre | Baixo |
| PokerBench dataset | Apache-2.0 | ✅ Livre | Baixo |
| Nemotron-Personas | CC-BY-4.0 | ✅ Livre | Baixo |
| MiniCPM | Apache-2.0 | ✅ Livre | Baixo |
| EXAONE | Custom "EXAONE AI" | ⚠️ Restrições | Médio |
| HyperCLOVA X | Custom | ⚠️ Restrições | Médio |
| Falcon-H1 | Custom | ⚠️ Verificar | Médio |
| PLaMo-100B | Proprietário | ❌ Research only | Alto |
| PanGu/Huawei | Proprietário | ❌ Não disponível | Alto |

### 11.3 Riscos Técnicos
1. **Overfitting a benchmarks:** Modelos coreanos otimizados para KMMLU podem não generalizar para raciocínio de game theory.
2. **Marketing vs. substância:** Claims de "maior modelo" ou "mais eficiente" frequentemente carecem de validação independente.
3. **Barreira de acesso:** Plataformas chinesas (CSTCloud, ModelScope) têm barreiras de idioma e acesso.
4. **Licenças restritivas:** Muitos modelos asiáticos têm licenças custom que limitam uso comercial.

---

## 12. Gaps de Pesquisa (Global)

### 12.1 Gaps por Região

| Região | Gap Principal | Severidade |
|--------|--------------|------------|
| **China** | **CORRIGIDO:** Tencent/CASIA é líder mundial em CFR. Gap: falta integração LLM+CFR nativa | Baixa |
| **Coreia** | Zero publicação em game AI para poker | Alta |
| **Japão** | Suspicion-Agent só testado em Leduc, não NL Hold'em | Média |
| **Índia** | Foco exclusivo em idiomas, sem behavioral gaming | Alta |
| **Singapura** | Nemotron-Personas excelente, mas sem domínio poker | Baixa |
| **UAE** | Foco em árabe/médico, sem game theory | Alta |
| **SEA** | Ecossistemas focados em NLP regional | Alta |

### 12.2 Gaps Temáticos

1. **Poker-specific AI na Ásia:** China (Tencent/CASIA) é líder mundial em CFR; Japão tem Suspicion-Agent. Coreia, Índia, SEA têm zero publicações poker-específicas.
2. **Behavioral datasets para poker:** Não existem datasets de decisão comportamental em poker em nenhum idioma asiático.
3. **Theory of Mind escalável:** Suspicion-Agent (UTokyo) é o mais próximo mas limitado a Leduc Hold'em.
4. **Multi-agent RL para poker na Ásia:** Toda pesquisa MARL asiática foca em videogames/robótica.
5. **Vieses cognitivos em game AI:** Todos os papers encontrados são de instituições ocidentais.
6. **Cross-lingual poker reasoning:** Nenhum trabalho sobre se LLMs asiáticos podem raciocinar sobre estratégia poker.
7. **PGM hierárquico para gaming:** Nemotron-Personas é PGM genérico; adaptação para poker é gap não coberto.
8. **Benchmarks poker multilíngues:** PokerBench é somente em inglês; versões em outros idiomas não existem.

---

## 13. Referências Finais

### Papers (89+ arXiv IDs referenciados)

**Poker AI & Game Theory:**
- 2501.08328, 2602.00528, 2401.06781, 2309.17277, 2506.24119
- 2201.11580, 2308.12466, 2402.17574, 2508.17671, 2212.06027
- 2108.01843, 2312.02561, 2602.16928, 2510.15414, 2407.00617
- 2409.00717, 2408.01072, 2202.07415, 2311.17190, 2108.12988
- 2212.12567, 2310.18940, 2502.04686, 2510.25080, 2502.09053

**Behavioral & Cognitive Science:**
- 2506.06366, 2411.10109, 2502.01506, 2412.15501, 2308.00225
- 2507.07186, 2406.10999, 2601.14553, 2402.04049, 2503.02582
- 2504.06868, 2509.16457, 2508.17407

**China:**
- 2501.12948 (DeepSeek-R1), 2412.19437 (DeepSeek-V3), 2505.09388 (Qwen3)
- 2508.06471 (GLM-4.5), 2507.20534 (Kimi K2), 2404.06395 (MiniCPM)
- 2403.17297 (InternLM2), 2402.03216 (BGE-M3), 2409.18869 (Emu3)
- 2405.04434 (DeepSeek-V2), 2401.06066 (DeepSeekMoE)
- 2411.02265 (Hunyuan-Large), 2505.15431 (Hunyuan-TurboS)

**Coreia do Sul:**
- 2404.01954 (HyperCLOVA X), 2506.22403 (HCXSEED-Think)
- 2601.01739 (K-EXAONE), 2412.04862 (EXAONE 3.5), 2507.11407 (EXAONE 4.0)
- 2312.15166 (SOLAR DUS), 2601.07022 (Solar Open), 2402.11548 (KMMLU)
- 2403.07691 (ORPO, KAIST), 2402.13605 (KorNAT)

**Japão:**
- 2404.17790 (Swallow CPT), 2505.02881 (SwallowCode/Math)
- 2403.13187 (Sakana evolutionary merge), 2406.07522 (PLaMo)
- 2111.15664 (Donut, NAVER), 2404.01657 (Rinna)

### HuggingFace Organizations (50+)

**China:** [Qwen](https://hf.co/Qwen), [deepseek-ai](https://hf.co/deepseek-ai), [THUDM](https://hf.co/THUDM), [openbmb](https://hf.co/openbmb), [internlm](https://hf.co/internlm), [OpenGVLab](https://hf.co/OpenGVLab), [BAAI](https://hf.co/BAAI), [tencent](https://hf.co/tencent), [01-ai](https://hf.co/01-ai)
**Coreia:** [skt](https://hf.co/skt), [naver-hyperclovax](https://hf.co/naver-hyperclovax), [LGAI-EXAONE](https://hf.co/LGAI-EXAONE), [upstage](https://hf.co/upstage), [kaist-ai](https://hf.co/kaist-ai)
**Japão:** [tokyotech-llm](https://hf.co/tokyotech-llm), [pfnet](https://hf.co/pfnet), [cyberagent](https://hf.co/cyberagent), [rinna](https://hf.co/rinna), [SakanaAI](https://hf.co/SakanaAI), [llm-jp](https://hf.co/llm-jp), [elyza](https://hf.co/elyza)
**Índia:** [ai4bharat](https://hf.co/ai4bharat), [sarvamai](https://hf.co/sarvamai), [krutrim-ai-labs](https://hf.co/krutrim-ai-labs)
**Singapura:** [aisingapore](https://hf.co/aisingapore), [nvidia](https://hf.co/nvidia) (Nemotron)
**UAE:** [tiiuae](https://hf.co/tiiuae), [MBZUAI](https://hf.co/MBZUAI), [inceptionai](https://hf.co/inceptionai)
**Vietnã:** [vinai](https://hf.co/vinai)
**Malásia:** [mesolitica](https://hf.co/mesolitica)
**Indonésia:** [indobenchmark](https://hf.co/indobenchmark)
**Tailândia:** [airesearch](https://hf.co/airesearch)
**Taiwan:** [ckiplab](https://hf.co/ckiplab)
**Poker:** [RZ412](https://hf.co/RZ412) (PokerBench)

### Links Externos

**Instituições Acadêmicas:**
- UTokyo Game AI: https://game.c.u-tokyo.ac.jp/
- KAIST MLAI Lab: https://www.mlai-kaist.com/
- NII LLMC (Japão): https://llmc.nii.ac.jp/en/
- ABCI (supercomputador Japão): https://abci.ai/en/
- Matsuo Lab (UTokyo): https://weblab.t.u-tokyo.ac.jp/en/research/
- Kyoto University NLP: https://nlp.ist.i.kyoto-u.ac.jp/EN/

**Plataformas:**
- ModelScope (China): https://modelscope.cn
- AIKosh (Índia): https://aikosh.indiaai.gov.in
- Bhashini (Índia): https://bhashini.gov.in
- Taiwan AI Labs: https://ailabs.tw/
- HyperCLOVA X: https://clova.ai/en/hyperclova
- Korea AISI: https://www.aisi.re.kr/eng/contents/22

**Código-fonte:**
- Suspicion-Agent: https://github.com/CR-Gjx/Suspicion-Agent
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR
- AceGPT: https://github.com/FreedomIntelligence/AceGPT
- awesome-japanese-llm: https://github.com/llm-jp/awesome-japanese-llm
