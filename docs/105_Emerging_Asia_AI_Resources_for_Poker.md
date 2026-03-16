# Emerging Asia AI Resources for Poker AI (MR_POKER)

**Date:** 2026-03-16
**Scope:** India, Singapore, UAE, Vietnam, Malaysia, Indonesia, Thailand
**Supplements:** docs/100_Global_AI_Research_Report.md (fills in sections 4.4, 4.5, 4.6 with verified data)

---

## 1. Executive Summary

**Total verified resources:** 52 unique resources across 7 countries/regions
**Search methodology:** HuggingFace Hub API, arXiv paper_search, web search
**All resources verified** against primary sources (HuggingFace repos, arXiv papers, official websites)

### Resource Count by Region

| Region | Models | Datasets | Papers | Institutions | Total |
|--------|--------|----------|--------|-------------|-------|
| **India** | 12 | 8 | 0 | 7 orgs | 27 |
| **Singapore** | 5 | 1 | 3 | 6 orgs | 12 |
| **UAE** | 13 | 0 | 0 | 3 orgs | 16 |
| **Vietnam** | 5 | 2 | 0 | 2 orgs | 9 |
| **Malaysia** | 0 | 0 | 0 | 1 org | 1 |
| **Indonesia** | 0 | 0 | 0 | 0 | 0 |
| **Thailand** | 2 | 0 | 0 | 2 orgs | 4 |
| **Global (Poker AI papers)** | 0 | 2 | 20 | — | 22 |
| **TOTAL** | **37** | **13** | **23** | **21 orgs** | **91** |

Note: Some papers and the PersonaHub/Nemotron-Personas datasets are global resources, not region-specific.

### Top 10 Findings Most Relevant to MR_POKER

1. **Nemotron-Personas-Singapore** (nvidia) -- 888K synthetic personas, PGM hierarchical, CC-BY-4.0. **Directly usable for SyntheticPlayerGenerator.**
2. **Nemotron-Personas-India** (nvidia) -- 1M-10M Indian personas, CC-BY-4.0. **Player persona diversity.**
3. **PersonaHub** (Tencent AI Lab) -- 1 billion diverse personas from web data. arXiv:2406.20094. **Scaling synthetic player generation.**
4. **PokerBench** (RZ412) -- 100K+ NL Hold'em scenarios with solver-optimal decisions. arXiv:2501.08328. **Direct training data.**
5. **Understanding LLM Agent Behaviours via Game Theory** (arXiv:2512.07462) -- Cognitive bias patterns in LLM agents. **Bias exploitation.**
6. **FAIRGAME** (arXiv:2504.14325) -- Framework for AI agent bias recognition using game theory. **Bias identification.**
7. **NTU Singapore Robust MARL** (Prof. Bo An) -- MIR3 for robustness against diverse threats. **Hardening opponent models.**
8. **Falcon3-10B** / **Falcon-H1-34B** (TII UAE) -- Strong SLMs, hybrid Mamba-Transformer. **Base model candidates.**
9. **DeepPersona** (arXiv:2511.07338) -- Taxonomy-guided synthetic persona generation. **Persona diversity methodology.**
10. **Opponent Model Deep CFR (ODCFR)** -- Neural opponent model integrated with Deep CFR. **Direct CFR enhancement.**

---

## 2. India (27 Resources)

### 2.1 Institutions

| Institution | Focus | Game Theory Relevance |
|------------|-------|----------------------|
| **Sarvam AI** | Multilingual LLMs (105B, 30B MoE, 24B), translation | None |
| **AI4Bharat (IIT Madras)** | 22-language NLP ecosystem, TTS, ASR | None |
| **Krutrim AI Labs (Ola)** | Indic LLM, VLM, embeddings | None |
| **IIIT Hyderabad** | Game theory, mechanism design, multiagent systems | **Medium** -- Prof. Sujit Gujar, Prof. Praveen Paruchuri |
| **IIT Bombay** | BharatGen consortium, 16 AIKosh datasets | None |
| **IISc/ARTPARK** | Speech data (Vaani) | None |
| **AIKosh (IndiaAI/MeitY)** | National AI data repository | Low -- potential decision datasets |

### 2.2 Models

| # | Name | Type | Org | Params | Downloads | Created | HuggingFace URL | Poker Relevance |
|---|------|------|-----|--------|-----------|---------|-----------------|----------------|
| IN1 | sarvam-105b | LLM | sarvamai | 105B | 7.3K | Mar 2026 | [link](https://hf.co/sarvamai/sarvam-105b) | Medium -- largest open Indian model, custom MLA arch |
| IN2 | sarvam-30b | LLM (MoE) | sarvamai | 30B | 35.6K | Mar 2026 | [link](https://hf.co/sarvamai/sarvam-30b) | Medium -- MoE for efficient inference |
| IN3 | sarvam-m | LLM | sarvamai | 24B | 3.5K | May 2025 | [link](https://hf.co/sarvamai/sarvam-m) | Low -- Apache-2.0, Mistral-Small-3.1 finetune |
| IN4 | sarvam-1 | LLM | sarvamai | ~7B | 8.8K | Oct 2024 | [link](https://hf.co/sarvamai/sarvam-1) | Low -- Llama-based, 11 Indian langs |
| IN5 | shuka-1 | Audio-LLM | sarvamai | ~8B | 37.6K | Aug 2024 | [link](https://hf.co/sarvamai/shuka-1) | Low -- audio-text multimodal |
| IN6 | sarvam-translate | Translation | sarvamai | Gemma3 | 21.8K | Jun 2025 | [link](https://hf.co/sarvamai/sarvam-translate) | Low -- 22 scheduled languages |
| IN7 | Krutrim-2-instruct | LLM | krutrim-ai-labs | ~7B | 185 | Feb 2025 | [link](https://hf.co/krutrim-ai-labs/Krutrim-2-instruct) | Low -- Mistral-based, 13 langs |
| IN8 | Chitrarth | VLM | krutrim-ai-labs | VLM | 259 | Feb 2025 | [link](https://hf.co/krutrim-ai-labs/Chitrarth) | Low -- arXiv:2502.15392 |
| IN9 | Vyakyarth | Embeddings | krutrim-ai-labs | XLM-R | 3.5K | Feb 2025 | [link](https://hf.co/krutrim-ai-labs/Vyakyarth) | Low -- multilingual semantic search |
| IN10 | indic-parler-tts | TTS | ai4bharat | TTS | 513.7K | Oct 2024 | [link](https://hf.co/ai4bharat/indic-parler-tts) | Low -- most downloaded Indian AI model |
| IN11 | indictrans2-en-indic-1B | Translation | ai4bharat | 1B | 6.7K | Sep 2023 | [link](https://hf.co/ai4bharat/indictrans2-en-indic-1B) | Low |
| IN12 | IndicNER | NER | ai4bharat | BERT | 25.1K | May 2022 | [link](https://hf.co/ai4bharat/IndicNER) | Low |

### 2.3 Datasets

| # | Name | Org | Size | Downloads | URL | Poker Relevance |
|---|------|-----|------|-----------|-----|----------------|
| IND1 | **Nemotron-Personas-India** | nvidia | 1M-10M personas | 8.8K | [link](https://hf.co/datasets/nvidia/Nemotron-Personas-India) | **HIGH** -- Synthetic personas grounded in Indian demographics, geography, personality traits (Big Five). CC-BY-4.0. Hindi+English bilingual. Directly usable for SyntheticPlayerGenerator. |
| IND2 | sangraha | ai4bharat | 251B tokens, 22 langs | 24.9K | [link](https://hf.co/datasets/ai4bharat/sangraha) | Low -- pretraining data |
| IND3 | MILU | ai4bharat | 10K-100K | 533 | [link](https://hf.co/datasets/ai4bharat/MILU) | Low -- multi-task benchmark |
| IND4 | IndicVoices | ai4bharat | 11,200 hrs audio | 8.7K | [link](https://hf.co/datasets/ai4bharat/IndicVoices) | Low -- speech data |
| IND5 | samvaad-hi-v1 | sarvamai | 100K conversations | 102 | [link](https://hf.co/datasets/sarvamai/samvaad-hi-v1) | Low -- Hindi dialogue |
| IND6 | mmlu-indic | sarvamai | 100K-1M | 648 | [link](https://hf.co/datasets/sarvamai/mmlu-indic) | Low -- translated MMLU |
| IND7 | IndicCorpV2 | ai4bharat | Massive | 1.6K | [link](https://hf.co/datasets/ai4bharat/IndicCorpV2) | Low |
| IND8 | AIKosh (platform) | IndiaAI/MeitY | 300+ datasets, 80+ models | N/A | [aikosh.indiaai.gov.in](https://aikosh.indiaai.gov.in/) | Medium -- National repository with sandbox. IIT Bombay 16 datasets. Potential decision-making data. |

### 2.4 India Analysis for MR_POKER

**Strengths:** Most prolific multilingual AI ecosystem globally. Sarvam AI now producing 105B-scale models. Nemotron-Personas-India is directly high-value for persona generation.

**Gaps:** Zero poker-specific or game-theoretic AI publications from Indian institutions. IIIT Hyderabad has game theory faculty but focuses on mechanism design (auctions, crowdsourcing), not poker. No behavioral decision-making datasets found.

**Opportunities:**
- Nemotron-Personas-India for Indian player persona generation
- AIKosh platform as potential future data source
- IIIT Hyderabad game theory expertise as collaboration target

---

## 3. Singapore (12 Resources)

### 3.1 Institutions

| Institution | Focus | Game Theory Relevance |
|------------|-------|----------------------|
| **Sea AI Lab (SAIL)** | Sailor2 multilingual LLM (15+ SEA langs), GitHub: sail-sg (101 repos) | None |
| **NTU (Prof. Bo An)** | MARL, adversarial learning, game theory, fraud detection | **HIGH** -- NSGZero, robust MARL, MIR3 |
| **NUS** | MARMoT Lab (multi-agent robotics), Bayesian optimization + RL | Medium |
| **A*STAR** | Government research. AI + behavioral science with NTU | Medium |
| **NVIDIA Singapore** | SIT x NVIDIA AI Centre (SNAIC). 50+ projects since 2024 | Low |
| **Google DeepMind Singapore** | Asia-Pacific AI lab | Low |

### 3.2 Models

| # | Name | Type | Org | Params | Downloads | Created | HuggingFace URL | Poker Relevance |
|---|------|------|-----|--------|-----------|---------|-----------------|----------------|
| SG1 | Sailor2-20B-Chat | LLM | sail (Sea AI Lab) | 20B (Qwen2) | 10 | Dec 2024 | [link](https://hf.co/sail/Sailor2-20B-Chat-1203) | Medium -- 15+ SEA languages (ID, TH, VI, MS, LO, MY, JV, KM, SU, TL, etc.) |
| SG2 | Sailor2-8B-Chat | LLM | sail (Sea AI Lab) | 8B (Qwen2) | 870 | Dec 2024 | [link](https://hf.co/sail/Sailor2-8B-Chat) | Medium -- compact SEA multilingual |
| SG3 | Sailor2-1B-Chat | LLM | sail (Sea AI Lab) | 1B (Qwen2) | 479 | Dec 2024 | [link](https://hf.co/sail/Sailor2-1B-Chat) | Low -- edge-deployable |
| SG4 | SeaLLM-13B-Chat | LLM | SeaLLMs | 13B | 0 | Oct 2023 | [link](https://hf.co/SeaLLMs/SeaLLM-13B-Chat) | Low -- older, arXiv:2312.00738 |
| SG5 | Agnes-SeaLLM-8b | LLM | Agnes-AI | 8B (Qwen3) | 5 | Jan 2026 | [link](https://hf.co/Agnes-AI/Agnes-SeaLLM-8b) | Low -- community fine-tune |

### 3.3 Datasets

| # | Name | Org | Size | Downloads | URL | Poker Relevance |
|---|------|-----|------|-----------|-----|----------------|
| SGD1 | **Nemotron-Personas-Singapore** | nvidia | 888K personas | 4.9K | [link](https://hf.co/datasets/nvidia/Nemotron-Personas-Singapore) | **HIGH** -- PGM hierarchical generation grounded in Singaporean demographics, geography, personality traits. CC-BY-4.0. 22+ fields. Directly usable for SyntheticPlayerGenerator. |

### 3.4 Academic Research

| # | Topic | Institution | Details | Relevance |
|---|-------|-------------|---------|-----------|
| SGR1 | Robust MARL (MIR3) | NTU (Prof. Bo An) | Mutual information regularization for robustness against diverse adversarial threats. TNNLS 2025. [PDF](https://personal.ntu.edu.sg/boan/papers/2025TNNLS_RobustMARL.pdf) | **HIGH** -- Directly applicable to robust opponent modeling against exploitative strategies |
| SGR2 | NSGZero + Game Theory | NTU (Prof. Bo An) | Deep RL for game theory, scalable adaptive behavior in complex environments. Applications: fraud detection, ride-hailing, recommender systems. | **HIGH** -- Game-theoretic reasoning for multi-agent decision-making |
| SGR3 | MARMoT Lab MARL | NUS | Multi-agent coordination, collaborative surveillance. IEEE MRS 2025. [marmotlab.org](https://marmotlab.org/) | Medium -- Multi-agent coordination techniques |

### 3.5 Singapore Analysis for MR_POKER

**Strengths:** Strongest hub for game-theoretic multi-agent research in the target region. Prof. Bo An at NTU is world-class in adversarial MARL. Nemotron-Personas-Singapore is the single most valuable resource across all regions for SyntheticPlayerGenerator.

**Gaps:** No poker-specific research from Singapore institutions. Sea AI Lab focuses on language models, not game AI.

**Opportunities:**
- Nemotron-Personas-Singapore for persona generation (already documented as top-5 resource globally)
- NTU robust MARL (MIR3) for hardening opponent models against adversarial exploitation
- Sailor2 for multilingual poker interface covering SEA markets (ID, TH, VI, MS, etc.)
- NVIDIA Singapore RL research infrastructure

---

## 4. UAE (16 Resources)

### 4.1 Institutions

| Institution | Focus | Game Theory Relevance |
|------------|-------|----------------------|
| **Technology Innovation Institute (TII)** | Falcon model family (1B-180B), Mamba-Transformer hybrids | None |
| **MBZUAI** | LaMini, MobiLlama, Arabic AI, medical AI | None |
| **Inception AI (G42)** | Jais Arabic LLM family (13B-30B) | None |

### 4.2 Models

| # | Name | Type | Org | Params | Downloads | Created | HuggingFace URL | Poker Relevance |
|---|------|------|-----|--------|-----------|---------|-----------------|----------------|
| UAE1 | **Falcon3-10B-Instruct** | LLM | tiiuae | 10B | 10.5K | Dec 2024 | [link](https://hf.co/tiiuae/Falcon3-10B-Instruct) | **Medium** -- #1 SLM under 13B on HF leaderboard. Trained on 14T tokens. Apache-2.0 based. Potential poker fine-tuning base. |
| UAE2 | Falcon3-7B-Instruct | LLM | tiiuae | 7B | 24.1K | Nov 2024 | [link](https://hf.co/tiiuae/Falcon3-7B-Instruct) | Medium -- good size/performance tradeoff |
| UAE3 | Falcon3-3B-Instruct | LLM | tiiuae | 3B | 11.4K | Dec 2024 | [link](https://hf.co/tiiuae/Falcon3-3B-Instruct) | Low -- edge deployment |
| UAE4 | Falcon3-1B-Instruct | LLM | tiiuae | 1B | 11.4K | Dec 2024 | [link](https://hf.co/tiiuae/Falcon3-1B-Instruct) | Low -- ultra-compact |
| UAE5 | **Falcon-H1-34B-Instruct** | Hybrid LLM | tiiuae | 34B | 2.8K | May 2025 | [link](https://hf.co/tiiuae/Falcon-H1-34B-Instruct) | **Medium** -- Innovative Mamba-Transformer hybrid. Efficient long-context processing for game histories. |
| UAE6 | **Falcon3-Mamba-7B-Instruct** | SSM | tiiuae | 7B | 1.8K | Dec 2024 | [link](https://hf.co/tiiuae/Falcon3-Mamba-7B-Instruct) | **Medium** -- Pure Mamba SSM. O(n) complexity vs O(n^2) transformers for long poker session histories. arXiv:2410.05355 |
| UAE7 | Falcon-E-3B-Instruct | BitNet LLM | tiiuae | 3B | 932 | Apr 2025 | [link](https://hf.co/tiiuae/Falcon-E-3B-Instruct) | Low -- 1-bit quantization edge model |
| UAE8 | falcon-mamba-7b | SSM | tiiuae | 7B | 15.6K | Jul 2024 | [link](https://hf.co/tiiuae/falcon-mamba-7b) | Medium -- first pure Mamba foundation model |
| UAE9 | falcon-180B | LLM | tiiuae | 180B | 185 | Aug 2023 | [link](https://hf.co/tiiuae/falcon-180B) | Low -- legacy large model |
| UAE10 | falcon-40b | LLM | tiiuae | 40B | 19.6K | May 2023 | [link](https://hf.co/tiiuae/falcon-40b) | Low -- Apache-2.0 |
| UAE11 | jais-13b-chat | LLM | inceptionai | 13B | 7.4K | Aug 2023 | [link](https://hf.co/inceptionai/jais-13b-chat) | Low -- Arabic/English bilingual, arXiv:2308.16149 |
| UAE12 | jais-30b-chat-v3 | LLM | inceptionai | 30B | 142 | Feb 2024 | [link](https://hf.co/inceptionai/jais-30b-chat-v3) | Low -- largest Arabic LLM |
| UAE13 | MobiLlama-05B | LLM | MBZUAI | 500M | 186 | Feb 2024 | [link](https://hf.co/MBZUAI/MobiLlama-05B) | Low -- mobile, arXiv:2402.16840, MIT license |

### 4.3 UAE Analysis for MR_POKER

**Strengths:**
- Falcon3-10B is top SLM globally under 13B, trained on 14T tokens with strong reasoning. Viable poker fine-tuning base.
- Falcon-H1-34B hybrid Mamba-Transformer is architecturally innovative for long-context reasoning (game histories).
- Falcon3-Mamba-7B pure SSM offers O(n) inference complexity -- excellent for processing long poker session histories.
- TII is the most prolific open-source model producer in the Middle East.

**Gaps:** No game-theoretic, poker-specific, or behavioral research from any UAE institution. MBZUAI focuses on medical AI and Arabic NLP.

---

## 5. Vietnam (9 Resources)

### 5.1 Institutions

| Institution | Focus | Game Theory Relevance |
|------------|-------|----------------------|
| **VinAI Research** | PhoBERT, PhoGPT, PhoWhisper, BERTweet. Vietnamese NLP/ASR. | None |
| **Zalo AI (VNG)** | Vietnamese LLM (surpasses GPT-4 on VMLU). Annual AI Challenge. 30% monthly users use AI features. | None |

### 5.2 Models

| # | Name | Type | Org | Params | Downloads | Created | HuggingFace URL | Poker Relevance |
|---|------|------|-----|--------|-----------|---------|-----------------|----------------|
| VN1 | phobert-base | BERT | vinai | 135M | 311.3K | Mar 2022 | [link](https://hf.co/vinai/phobert-base) | Low -- Vietnamese NLP, arXiv:2003.00744, MIT |
| VN2 | PhoGPT-4B | LLM | vinai | 4B | 791 | Jan 2024 | [link](https://hf.co/vinai/PhoGPT-4B) | Low -- Vietnamese GPT, arXiv:2311.02945, BSD-3 |
| VN3 | PhoWhisper-small | ASR | vinai | Whisper | 3.3K | Feb 2024 | [link](https://hf.co/vinai/PhoWhisper-small) | Low -- Vietnamese ASR |
| VN4 | bertweet-base | BERT | vinai | 135M | 261.4K | Mar 2022 | [link](https://hf.co/vinai/bertweet-base) | Low -- Twitter sentiment, MIT. Minor applicability to chat-based behavioral analysis |
| VN5 | bartpho-syllable | Seq2Seq | vinai | ~400M | 116.5K | Mar 2022 | [link](https://hf.co/vinai/bartpho-syllable) | Low -- arXiv:2109.09701 |

### 5.3 Datasets

| # | Name | Org | Size | Downloads | URL | Poker Relevance |
|---|------|-----|------|-----------|-----|----------------|
| VND1 | RecGPT-datasets | vinai | 100K-1M | 22 | [link](https://hf.co/datasets/vinai/RecGPT-datasets) | Low -- recommendation data |
| VND2 | PhoST | vinai | 508 audio hours | 34 | [link](https://hf.co/datasets/vinai/PhoST) | Low -- EN-VI speech |

### 5.4 Vietnam Analysis

No game-theoretic or poker-specific research. Zalo AI's Vietnamese LLM (not publicly on HuggingFace) is relevant only for Vietnamese-market poker interface. VinAI's bertweet has minor applicability to chat-based behavioral analysis.

---

## 6. Malaysia, Indonesia, Thailand (5 Resources)

### 6.1 Malaysia
- **Universiti Sains Malaysia:** Active in cooperative multi-agent RL research (robotics domain). MARL paper at SAGE Journals 2025.
- No major foundation model producers. Malaysian researchers primarily contribute community fine-tunes.
- **Poker relevance:** Minimal.

### 6.2 Indonesia
- Covered by Sea AI Lab's **Sailor2** models for Indonesian language (primary language in the model).
- **IndoBenchmark:** IndoBERT (242K downloads).
- No game-theoretic or poker AI research found.
- **Poker relevance:** None beyond language coverage via Sailor2.

### 6.3 Thailand
| # | Name | Type | Org | Details | Poker Relevance |
|---|------|------|-----|---------|----------------|
| TH1 | openthaigpt-1.0.0-7b-chat | LLM | openthaigpt | 7B (Llama2), arXiv:2411.07238 | Low |
| TH2 | Pathumma LLM | LLM | NECTEC | Text + Vision + Audio. Plans for Agentic AI by 2025. Not on HuggingFace. [source](https://www.nstda.or.th/en/news/news-years-2025/pathumma-llm-ai-technology-tailored-to-thai-context-and-culture.html) | Low |

- **NECTEC:** National AI lab developing Pathumma LLM (3 modalities). Government-driven.
- **AIAT + AIEAT:** ThaiGPT development with NECTEC across 8 sectors.
- **Poker relevance:** None. Covered by Sailor2 for Thai language.

---

## 7. Global Poker AI & Game Theory Papers (New additions from this search)

These supplement section 5 of the main report (100_Global_AI_Research_Report.md).

### 7.1 New Poker AI Papers

| # | Title | arXiv ID | Year | Relevance | Key Details |
|---|-------|----------|------|-----------|-------------|
| P13 | A Survey on Game Theory Optimal Poker | arXiv:2401.06168 | Jan 2024 | **HIGH** | Comprehensive survey of GTO poker approaches. Reviews CFR variants, neural network solvers, profit-maximizing strategies. |
| P14 | Beyond GTO: Profit-Maximizing Poker Agents for NL Hold'em | arXiv:2509.23747 | Sep 2025 | **HIGH** | Largest profits from detecting and exploiting opponent deviations from equilibrium in real time. Current poker AIs are either purely GTO or trained on static population tendencies. |
| P15 | On Deep CFR Integrated with Opponent Model (ODCFR) | ScienceDirect 2025 | 2025 | **HIGH** | General online neural-based implicit opponent model. Derived from Deep CFR. Tested on Leduc Poker (2 colors, 26 cards, 6 bets/round). Exploits irrational opponent idiosyncrasies. |
| P16 | LLM-Guided Strategy and Opponent Modeling in Multi-Agent Games | Stanford CS224R | 2025 | Medium | Course project combining LLM strategy with opponent modeling. |
| P17 | Combining Deep RL and Search with Generative Models for Game-Theoretic Opponent Modeling | arXiv:2302.00797 | Feb 2023 | **HIGH** | Generative Best Response (GenBR). AlphaZero-style RL + MCTS for large imperfect information games. Test-time computation with learned world model. |

### 7.2 New Game Theory + LLM Papers

| # | Title | arXiv ID | Year | Relevance | Key Details |
|---|-------|----------|------|-----------|-------------|
| G14 | Understanding LLM Agent Behaviours via Game Theory | arXiv:2512.07462 | Dec 2025 | **HIGH** | FAIRGAME framework extension. Systematic model-specific behavioral biases: Claude maintains prosocial tendencies even under selfish framing; GPT-4o combines instruction adherence with linguistic sensitivity; Mistral shows language-invariant stability. Payoff-scaled Prisoner's Dilemma + Public Goods Game. **Directly applicable to cognitive bias exploitation -- each LLM-simulated opponent has predictable bias patterns.** |
| G15 | FAIRGAME: Framework for AI Agent Bias Recognition using Game Theory | arXiv:2504.14325 | Apr 2025 | **HIGH** | Explores how personalities, cognitive behavioral traits, and environmental factors affect group dynamics. Identifies optimal configurations prioritizing collective outcomes. **Directly applicable to bias identification in opponent agents.** |
| G16 | Game Theory Meets LLMs: Systematic Survey with Taxonomy | arXiv:2502.09053 (IJCAI 2025 proceedings) | Feb 2025 | Medium | LLMs exhibit pro-social biases, prioritizing fairness/cooperation over game-theoretic rationality. Higher cooperation rates than humans in social dilemmas. Inequity aversion in Ultimatum Game. |
| G17 | Do LLM Agents Have Regret? Online Learning and Games | arXiv:2403.16843 | Mar 2024 | Medium | Introduces regret-loss for improving no-regret behavior in LLM agents. Generalization bound analysis. |
| G18 | Can LLMs Serve as Rational Players in Game Theory? | arXiv:2312.05488 | Dec 2023 | Medium | LLM rationality evaluation in dictator game, RPS, ring-network game. Significant differences from human players. |
| G19 | Learning Robust Social Strategies with LLMs | arXiv:2511.19405 | Nov 2025 | Medium | RL-trained LLM agents in social dilemmas. Advantage alignment for cooperative outcomes. Non-exploitability. |

### 7.3 New Persona / Behavioral Modeling Papers

| # | Title | arXiv ID | Year | Relevance | Key Details |
|---|-------|----------|------|-----------|-------------|
| B14 | Scaling Synthetic Data Creation with 1B Personas (PersonaHub) | arXiv:2406.20094 | Jun 2024 | **HIGH** | Tencent AI Lab. 1 billion diverse personas from web data. Persona-driven data synthesis for math, reasoning, NPCs, tools. Dataset: [proj-persona/PersonaHub](https://hf.co/datasets/proj-persona/PersonaHub) (716 likes, 211K downloads). **Directly applicable to scaling SyntheticPlayerGenerator.** |
| B15 | DeepPersona: Generative Engine for Scaling Deep Synthetic Personas | arXiv:2511.07338 | Nov 2025 | **HIGH** | Taxonomy-guided approach. Attribute sampling + narrative completeness. Personality testing + behavioral simulation. **Methodology transferable to poker persona generation.** |
| B16 | Socially-Grounded Persona Framework for User Simulation | arXiv:2601.07110 | Jan 2026 | Medium | Sociopsychological protocols > demographic templates for behavioral prediction alignment. |
| B17 | PsyPlay: Personality-Infused Role-Playing Agents | arXiv:2502.03821 | Feb 2025 | Medium | LLM agents consistently express specific personality traits. High accuracy with positive personality roles. |
| B18 | PERSONA: Reproducible Testbed for Pluralistic Alignment | arXiv:2407.17387 | Jul 2024 | Medium | Procedurally generated synthetic personas with demographic + idiosyncratic attributes. PERSONA Bench evaluation. |
| B19 | Simulating Deceptive Behaviors in Long-Horizon Interactions | arXiv:2510.03999 | Oct 2025 | **HIGH** | Multi-agent deception framework: concealment, equivocation, falsification. Trust erosion modeling. **Directly relevant to poker bluffing/deception modeling.** |
| B20 | Theory of Mind for Multi-Agent Collaboration via LLMs | arXiv:2310.10701 | Oct 2023 | Medium | Multi-agent text game. ToM capabilities in LLMs. Explicit belief state representations improve planning. |
| B21 | Learning to Deceive in Multi-Agent Hidden Role Games | arXiv:2209.01551 | Sep 2022 | Medium | Bayesian belief manipulation (BBM) model for deception in mixed cooperative-competitive settings. **Applicable to bluffing strategies.** |

### 7.4 New Skill Estimation Papers

| # | Title | arXiv ID | Year | Relevance | Key Details |
|---|-------|----------|------|-----------|-------------|
| SE1 | OpenSkill: Faster Asymmetric Multi-Team Rating System | arXiv:2401.05451 | Jan 2024 | Medium | Bayesian inference (Plackett-Luce + TrueSkill). Time decay. Faster than TrueSkill for multiplayer ranking. Python library. |
| SE2 | Chess Rating Estimation from Moves and Clock Times | arXiv:2409.11506 | Sep 2024 | Medium | CNN + bidirectional LSTM estimates player rating from moves. No hand-crafted features. **Methodology transferable to poker skill estimation from hand histories.** |
| SE3 | Learning to Move Like Professional CS:GO Players | arXiv:2408.13934 | Aug 2024 | Low | Transformer-based movement model. TrueSkill self-play evaluation. Data-driven professional behavior modeling. |

### 7.5 New Datasets

| # | Name | Type | Size | Downloads | URL | Poker Relevance |
|---|------|------|------|-----------|-----|----------------|
| PD5 | **PersonaHub** | Synthetic personas | 100M-1B personas | 211.3K | [proj-persona/PersonaHub](https://hf.co/datasets/proj-persona/PersonaHub) | **HIGH** -- 1 billion diverse personas. CC-BY-NC-SA-4.0. Includes NPC personas, reasoning, tools. Tencent AI Lab. arXiv:2406.20094. |
| PD6 | **Nemotron-Personas-France** | Synthetic personas | 1M-10M | 81 | [nvidia/Nemotron-Personas-France](https://hf.co/datasets/nvidia/Nemotron-Personas-France) | Low -- French personas, same PGM methodology as Singapore/India variants |
| PD7 | **Nemotron-Personas-Japan** | Synthetic personas | 1M-10M | 10.1K | [nvidia/Nemotron-Personas-Japan](https://hf.co/datasets/nvidia/Nemotron-Personas-Japan) | Medium -- Japanese personas, CC-BY-4.0. Cross-cultural persona generation. |
| PD8 | PokerBench SFT Chat | Chat-formatted poker | 100K+ | varies | [felipesp1983/pokerbench-sft-chat](https://hf.co/datasets/felipesp1983/pokerbench-sft-chat) | HIGH -- Chat-formatted PokerBench for SFT training |
| PD9 | stackexchange_poker | Q&A | 1K-10K | 6 | [mlfoundations-dev/stackexchange_poker](https://hf.co/datasets/mlfoundations-dev/stackexchange_poker) | Low -- Poker strategy discussions |

---

## 8. Mapping to MR_POKER Components

### 8.1 CFR Solver
- **P13** Survey on GTO Poker (arXiv:2401.06168) -- comprehensive CFR variant review
- **P15** ODCFR -- Deep CFR integrated with neural opponent model
- **P17** GenBR -- AlphaZero-style RL + MCTS for imperfect information games
- **G1** AlphaEvolve CFR variants (from main report)

### 8.2 Opponent Modeling
- **P14** Beyond GTO: real-time exploitation of opponent deviations
- **P15** ODCFR: neural implicit opponent model
- **P17** GenBR: generative model for opponent behavior
- **SGR1** NTU Robust MARL (MIR3): hardening against adversarial opponents
- **G14** FAIRGAME: systematic bias patterns per LLM type

### 8.3 Behavioral Prediction
- **G14** LLM agent behavioral biases (model-specific patterns)
- **B14** PersonaHub: 1B personas for behavioral diversity
- **B15** DeepPersona: taxonomy-guided persona generation
- **B19** Deceptive behavior simulation (concealment, equivocation, falsification)

### 8.4 Skill Estimation
- **SE1** OpenSkill: Bayesian multi-team rating with time decay
- **SE2** Chess rating from moves: CNN+LSTM, transferable to poker
- **P1** PokerBench evaluation framework (from main report)

### 8.5 Synthetic Player Generation
- **IND1** Nemotron-Personas-India: Indian player personas (CC-BY-4.0)
- **SGD1** Nemotron-Personas-Singapore: Singaporean personas (CC-BY-4.0)
- **PD5** PersonaHub: 1B diverse personas (CC-BY-NC-SA-4.0)
- **PD7** Nemotron-Personas-Japan: Japanese personas (CC-BY-4.0)
- **B14** Scaling methodology (arXiv:2406.20094)
- **B15** DeepPersona taxonomy methodology (arXiv:2511.07338)

### 8.6 Cognitive Bias Exploitation
- **G14** Model-specific biases: Claude=prosocial, GPT-4o=instruction-adherent, Mistral=stable
- **G15** FAIRGAME: personality/cognition/environment factors in group dynamics
- **B19** Deception modeling in long-horizon interactions
- **B21** Bayesian belief manipulation for bluffing

---

## 9. Priority Integration Recommendations

### 9.1 Immediate Integration (Week 1-2)
1. **Nemotron-Personas-Singapore** + **Nemotron-Personas-India** -- Download and integrate into SyntheticPlayerGenerator pipeline. Both CC-BY-4.0.
2. **PersonaHub** -- Evaluate 1B persona dataset for scaling player archetypes. Note CC-BY-NC-SA-4.0 license limitation.
3. **PokerBench** (already in pipeline) -- Verify integration with ODCFR findings.

### 9.2 Short-term Research (Month 1)
4. **ODCFR** (P15) -- Study neural opponent model integration with existing Deep CFR implementation.
5. **FAIRGAME** (G14/G15) -- Map systematic bias patterns to poker exploitation strategies.
6. **DeepPersona** methodology (B15) -- Adapt taxonomy-guided approach for poker player persona generation.

### 9.3 Medium-term Exploration (Month 2-3)
7. **Falcon3-10B** / **Falcon-H1-34B** -- Evaluate as alternative base models for poker agent (strong reasoning + efficient architecture).
8. **NTU Robust MARL** (SGR1) -- Study MIR3 for hardening opponent models.
9. **OpenSkill** (SE1) -- Evaluate Bayesian rating system for real-time skill estimation.
10. **Chess Rating CNN+LSTM** (SE2) -- Adapt move-based rating estimation methodology for poker hand histories.

---

## 10. Licensing Summary

| Resource | License | Commercial Use |
|----------|---------|---------------|
| Nemotron-Personas-Singapore | CC-BY-4.0 | Yes |
| Nemotron-Personas-India | CC-BY-4.0 | Yes |
| PersonaHub | CC-BY-NC-SA-4.0 | **No** (non-commercial) |
| PokerBench | Apache-2.0 | Yes |
| Falcon3 family | TII Falcon License (Apache-2.0 based) | Yes (with acceptable use policy) |
| Sailor2 family | Various (check per model) | Varies |
| Sarvam models | Custom / Apache-2.0 (varies) | Varies |
| VinAI models | MIT / BSD-3 | Yes |
| Jais models | Apache-2.0 | Yes |

---

## 11. Sources

### HuggingFace Repositories
- [sarvamai](https://hf.co/sarvamai) | [ai4bharat](https://hf.co/ai4bharat) | [krutrim-ai-labs](https://hf.co/krutrim-ai-labs)
- [tiiuae](https://hf.co/tiiuae) | [inceptionai](https://hf.co/inceptionai) | [MBZUAI](https://hf.co/MBZUAI)
- [sail](https://hf.co/sail) | [SeaLLMs](https://hf.co/SeaLLMs) | [vinai](https://hf.co/vinai)
- [nvidia](https://hf.co/nvidia) | [proj-persona](https://hf.co/datasets/proj-persona/PersonaHub)

### Web Sources
- [AIKosh - IndiaAI](https://aikosh.indiaai.gov.in/)
- [Sea AI Lab Research](https://sail.sea.com/research)
- [NTU Prof. Bo An](https://www.ntu.edu.sg/ias/news-events/news/detail/from-algorithmic-and-reinforcement-learning-based-to-llm-powered-agents-by-prof-bo-an)
- [Falcon 3 Launch](https://www.tii.ae/news/falcon-3-uaes-technology-innovation-institute-launches-worlds-most-powerful-small-ai-models)
- [Zalo AI Summit 2025](https://summit.zalo.ai/)
- [NECTEC Pathumma LLM](https://www.nstda.or.th/en/news/news-years-2025/pathumma-llm-ai-technology-tailored-to-thai-context-and-culture.html)
- [MBZUAI](https://mbzuai.ac.ae/)
- [arXiv:2512.07462](https://arxiv.org/abs/2512.07462) - LLM Agent Behaviours via Game Theory
- [arXiv:2504.14325](https://arxiv.org/abs/2504.14325) - FAIRGAME
- [arXiv:2502.09053](https://arxiv.org/abs/2502.09053) - Game Theory Meets LLMs Survey
