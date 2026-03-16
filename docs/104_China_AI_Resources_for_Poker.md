# China / HK / Taiwan AI Resources for MR_POKER

**Date:** 2026-03-16
**Scope:** China (mainland), Hong Kong, Taiwan
**Target System:** MR_POKER (CFR solver, opponent modeling, behavioral prediction, skill estimation, synthetic player generation, cognitive bias exploitation)

---

## Summary

This report catalogs 52 verified AI resources from Greater China that are relevant to the MR_POKER poker AI system. Resources span foundation models, game-solving algorithms, reinforcement learning frameworks, behavioral science papers, and datasets. All entries have been verified through web search, HuggingFace API, or arXiv lookup.

---

## 1. Foundation Models (Reasoning / Agent Backbone)

These models can serve as the LLM backbone for opponent modeling, behavioral prediction, natural language reasoning about game states, or as teacher models for distillation.

### 1.1 DeepSeek-R1
- **Type:** Model (MoE, reasoning-optimized)
- **Source:** [huggingface.co/deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)
- **Paper:** arXiv:2501.12948
- **Size:** 671B total params (37B active), MoE architecture based on DeepSeek-V3-Base
- **Origin:** DeepSeek (Hangzhou, China)
- **Relevance:** **HIGH** -- Pure RL training (GRPO) produces emergent self-reflection and dynamic strategy adaptation. Directly applicable to poker reasoning chains. Distilled variants available at 1.5B/7B/14B/32B/70B.
- **Key detail:** First open model to show reasoning emerges from pure RL without SFT. Distilled versions (DeepSeek-R1-Distill-Qwen-32B) outperform OpenAI o1-mini on benchmarks.

### 1.2 DeepSeek-V3
- **Type:** Model (MoE, general)
- **Source:** [huggingface.co/deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)
- **Paper:** arXiv:2412.19437
- **Size:** 685B total (671B main + 14B MTP module), 37B active per token
- **Origin:** DeepSeek (Hangzhou, China)
- **Relevance:** **MEDIUM** -- Multi-head Latent Attention (MLA), auxiliary-loss-free load balancing. Pre-trained on 14.8T tokens. Strong general backbone; DeepSeek-R1 is built on this base.
- **Key detail:** Only 2.788M H800 GPU hours for full training. Multi-token prediction training objective.

### 1.3 Qwen3 Series
- **Type:** Model family (dense + MoE)
- **Source:** [huggingface.co/Qwen](https://huggingface.co/Qwen)
- **Paper:** arXiv:2505.09388
- **Size:** Dense: 0.6B, 1.7B, 4B, 8B, 14B, 32B; MoE: 30B-A3B, 235B-A22B
- **Origin:** Alibaba Cloud (Hangzhou, China)
- **Relevance:** **HIGH** -- Four-stage RL pipeline (CoT cold start -> reasoning RL -> thinking mode fusion -> general RL). Agent capabilities with tool integration. Leading open-source model for fine-tuning.
- **Key detail:** Qwen3.5-397B-A17B extends with million-agent RL environments. Native thinking/non-thinking mode switching.

### 1.4 Qwen3.5 Series
- **Type:** Model family (MoE + dense)
- **Source:** [huggingface.co/Qwen/Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)
- **Size:** 397B-A17B (flagship), 35B-A3B, plus small series 0.8B/2B/4B/9B
- **Origin:** Alibaba Cloud (Hangzhou, China)
- **Relevance:** **MEDIUM** -- Scaled RL across million-agent environments. Strong for on-device deployment (small variants) or full-scale reasoning (large variants).

### 1.5 GLM-4 / GLM-4.5
- **Type:** Model family
- **Source:** [huggingface.co/THUDM/glm-4-9b-chat-hf](https://huggingface.co/THUDM/glm-4-9b-chat-hf)
- **Paper:** arXiv:2406.12793
- **Size:** GLM-4-9B (open), GLM-4.5: 355B and 106B (closed/limited)
- **Origin:** Zhipu AI / Tsinghua University (Beijing, China)
- **Relevance:** **MEDIUM** -- Function calling, 128K context, web browsing. GLM-4-9B-Chat-1M supports 1M context. Good for long game history analysis.
- **Key detail:** GLM-4-32B-0414 adds reasoning/rumination capabilities.

### 1.6 Kimi K2
- **Type:** Model (MoE, agentic)
- **Source:** [huggingface.co/moonshotai/Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct)
- **Size:** 1T total params, 32B active. Pre-trained on 15.5T tokens with Muon optimizer.
- **Origin:** Moonshot AI (Beijing, China)
- **Relevance:** **MEDIUM** -- Optimized for agentic capabilities. K2.5 introduces swarm-like multi-agent coordination with dynamic sub-task decomposition.
- **Key detail:** Kimi-K2-Thinking variant reasons step-by-step while invoking tools.

### 1.7 Kimi K1.5
- **Type:** Model (multimodal reasoning)
- **Source:** GitHub: [MoonshotAI/Kimi-k1.5](https://github.com/MoonshotAI/Kimi-k1.5)
- **Paper:** arXiv:2501.12599
- **Origin:** Moonshot AI (Beijing, China)
- **Relevance:** **MEDIUM** -- RL framework without MCTS, value functions, or process reward models. Long context scaling + improved policy optimization. 77.5 AIME, 96.2 MATH 500.

### 1.8 InternLM3
- **Type:** Model (8B, general + reasoning)
- **Source:** [huggingface.co/internlm](https://huggingface.co/internlm), GitHub: [InternLM/InternLM](https://github.com/InternLM/InternLM)
- **Size:** 8B params, trained on 4T tokens
- **Origin:** Shanghai AI Laboratory (Shanghai, China)
- **Relevance:** **LOW** -- Conversational + deep thinking modes via system prompts. Integrated tool-use capabilities. 75% training cost reduction.

### 1.9 Yi-Lightning
- **Type:** Model (MoE, fast inference)
- **Source:** [huggingface.co/01-ai](https://huggingface.co/01-ai)
- **Paper:** arXiv:2412.01253
- **Size:** 200B+ training scale, MoE architecture
- **Origin:** 01.AI (Beijing, China)
- **Relevance:** **LOW** -- 200+ tokens/sec on RTX 4090. Enhanced MoE with optimized KV-caching. Apache 2.0 license.

### 1.10 Baichuan-M3-235B
- **Type:** Model (medical-enhanced, but general MoE)
- **Source:** [huggingface.co/baichuan-inc/Baichuan-M3-235B](https://huggingface.co/baichuan-inc/Baichuan-M3-235B)
- **Origin:** Baichuan Intelligence (Beijing, China)
- **Relevance:** **LOW** -- Medical focus limits direct poker use, but clinical decision-making pipeline could inspire sequential decision modeling.

### 1.11 DeepSeek-R1 Distilled Models
- **Type:** Distilled reasoning models
- **Source:** HuggingFace `deepseek-ai/DeepSeek-R1-Distill-Qwen-{1.5B,7B,14B,32B}` and `DeepSeek-R1-Distill-Llama-{8B,70B}`
- **Origin:** DeepSeek (Hangzhou, China)
- **Relevance:** **HIGH** -- 800K synthetic reasoning samples from R1 used for distillation. The 32B Qwen distill outperforms o1-mini. Practical sizes for deployment in poker reasoning pipeline.

---

## 2. Poker-Specific AI Systems and Papers

### 2.1 DecisionHoldem
- **Type:** Poker AI system + paper
- **Source:** arXiv:2201.11580 (updated May 2024), GitHub: [AI-Decision/DecisionHoldem](https://github.com/AI-Decision/DecisionHoldem)
- **Authors:** Dongdong Bai et al., Institute of Automation, Chinese Academy of Sciences (CASIA), Beijing
- **Relevance:** **HIGH** -- First open-source high-level HUNL AI. Safe depth-limited subgame solving with diverse opponent modeling. CFR-based. Defeats Slumbot and OpenStack.
- **Key detail:** Provides toolkit for playing against Slumbot/OpenStack and human play platform.

### 2.2 Signal Observation Models and Historical Information in Poker Hand Abstraction
- **Type:** Paper
- **Source:** arXiv:2403.11486 (March 2024, revised Jan 2025)
- **Authors:** Yanchang Fu, Pei Xu, Dongdong Bai, Lingyun Zhao, Kaiqi Huang (CASIA, Beijing)
- **Relevance:** **HIGH** -- KrwEmd algorithm integrates historical information into hand abstraction. Introduces signal observation ordered game model and k-recall winrate feature. Novel resolution bound evaluation metric.

### 2.3 No-Regret Strategy Solving via Pre-Trained Embedding
- **Type:** Paper
- **Source:** arXiv:2511.12083 (Nov 2025)
- **Authors:** Yanchang Fu, Shengda Liu, Pei Xu, Kaiqi Huang (CASIA, Beijing)
- **Relevance:** **HIGH** -- Embedding CFR algorithm: pre-trains information set abstractions through low-dimensional embedding instead of discrete clustering. First to use NLP-style embeddings for poker info-set abstraction. Faster exploitability convergence.

### 2.4 PokerGPT
- **Type:** LLM poker solver + paper
- **Source:** arXiv:2401.06781 (Jan 2024)
- **Authors:** Chenghao Huang, Yanbo Cao, Yinlong Wen, Tao Zhou, Yanru Zhang
- **Relevance:** **HIGH** -- End-to-end lightweight LLM solver for multi-player Texas Hold'em. Fine-tunes LLM using RLHF on textual game records. Outperforms prior approaches in win rate, model size, and speed.

### 2.5 ToolPoker / "How Far Are LLMs from Professional Poker Players?"
- **Type:** Paper + framework
- **Source:** arXiv:2602.00528 (Jan 2026), published at ICLR 2026
- **Authors:** Minhua Lin, Enyan Dai, Hui Liu, Xianfeng Tang, et al.
- **Relevance:** **HIGH** -- Identifies three LLM failure modes in poker: heuristic reliance, factual misunderstandings, knowing-doing gap. ToolPoker framework integrates external GTO solvers with LLM reasoning. State-of-the-art gameplay with game-theoretic reasoning traces.

### 2.6 Deep (Predictive) Discounted CFR
- **Type:** Paper
- **Source:** arXiv:2511.08174 (Nov 2025)
- **Authors:** Hang Xu, Kai Li, Haobo Fu, Qiang Fu, Junliang Xing, Jian Cheng (Tencent AI Lab + CASIA)
- **Relevance:** **HIGH** -- Model-free neural CFR with variance-reduced sampled advantages, bootstrapping, and discounting. Faster convergence than prior neural CFR methods. Stronger adversarial performance in large poker games.

### 2.7 Minimizing Weighted Counterfactual Regret (PDCFR+)
- **Type:** Paper
- **Source:** arXiv:2404.13891 (April 2024), IJCAI 2024 Oral
- **Authors:** Kai Li, Hang Xu, Haobo Fu, Qiang Fu, Junliang Xing (Tencent AI Lab + CASIA)
- **Relevance:** **HIGH** -- PDCFR+ integrates PCFR+ and Discounted CFR via optimistic online mirror descent. Swiftly mitigates dominated actions while leveraging predictions for convergence speedup. Open source: [github.com/rpSebastian/PDCFRPlus](https://github.com/rpSebastian/PDCFRPlus)

### 2.8 Opponent-Limited Online Search for IIGs
- **Type:** Paper
- **Source:** ICML 2023 (Proceedings v202, pp. 21567-21585)
- **Authors:** Weiming Liu, Haobo Fu, Qiang Fu, Wei Yang (Tencent)
- **Relevance:** **HIGH** -- Safe-1-KLSS and Opponent-Limited Subgame Solving (OLSS). Limits opponent's strategy to reduce subgame size while preserving safety. Directly applicable to real-time poker search.

### 2.9 Actor-Critic Hedge (ACH) for Mahjong
- **Type:** Paper
- **Source:** ICLR 2022, OpenReview: DTXZqTNV5nW
- **Authors:** Haobo Fu, Weiming Liu, Shuang Wu, et al. (Tencent)
- **Relevance:** **MEDIUM** -- Actor-Critic policy optimization for large-scale imperfect-information games (1-on-1 Mahjong). Minimizes weighted cumulative counterfactual regret. Transferable concepts to poker.

### 2.10 AutoCFR
- **Type:** Paper
- **Source:** AAAI 2022 (Oral), related: Dynamic Discounted CFR at ICLR 2024 Spotlight
- **Authors:** Kai Li, Hang Xu, Haobo Fu, Qiang Fu, Junliang Xing (Tencent + CASIA)
- **Relevance:** **MEDIUM** -- Automatically designs CFR algorithms for solving IIGs. Meta-learning approach to CFR variant design.

---

## 3. Imperfect Information Game AI (Non-Poker)

### 3.1 DouZero
- **Type:** Game AI system
- **Source:** arXiv:2106.06135, GitHub: [kwai/DouZero](https://github.com/kwai/DouZero) (ICML 2021)
- **Authors:** Daochen Zha et al.
- **Relevance:** **MEDIUM** -- Self-play deep RL for DouDizhu (Chinese card game). Monte-Carlo methods + deep neural networks + parallel actors. Ranked #1 on Botzone among 344 agents. Architecture transferable to poker.

### 3.2 OADMCDou (Enhanced DouDizhu with Oracle Guiding)
- **Type:** Paper
- **Source:** IJCAI 2024 proceedings/660
- **Relevance:** **MEDIUM** -- Oracle Guiding trains with both imperfect and perfect information, gradually reducing perfect info reliance. Adaptive Deep Monte Carlo with gradient weight clipping. Outperforms DouZero by 28.6% loss reduction. Directly applicable to poker training with oracle information.

### 3.3 GuanZero (Mastering Guandan)
- **Type:** Paper + system
- **Source:** arXiv:2402.13582 (Feb 2024)
- **Authors:** Yifan Yanggong, Hao Pan, Lei Wang
- **Relevance:** **MEDIUM** -- Four-player imperfect information card game with cooperative+competitive dynamics. Behavior regulating through neural network encoding. Monte-Carlo + deep NN framework. 130M+ Guandan players in China.

### 3.4 Tjong (Transformer-based Mahjong AI)
- **Type:** Paper
- **Source:** CAAI Transactions on Intelligence Technology, 2024, 9(4):982-995
- **Authors:** Xiali Li, Bo Liu, Zhi Wei, Zhaoqi Wang, Licheng Wu
- **Relevance:** **MEDIUM** -- Transformer with hierarchical decision-making (action decision + tile decision). Fan backward technique for sparse reward allocation. 15M params, top 1% on Botzone. Self-attention for tile pattern capture applicable to card hand modeling.

### 3.5 LsAc*-MJ (Low-Resource Mahjong RL)
- **Type:** Paper
- **Source:** International Journal of Intelligent Systems, 2024
- **Relevance:** **LOW** -- LSTM + optimized A2C with experience replay. Two-stage training: expert-guided then self-play. Low-resource consumption approach relevant to efficient poker training.

### 3.6 Suphx (Mahjong AI)
- **Type:** Game AI system
- **Source:** arXiv:2003.13590, Microsoft Research Asia
- **Relevance:** **MEDIUM** -- Global reward prediction, oracle guiding, run-time policy adaptation for imperfect information. First AI to achieve Tenhou 10-dan. "Prior Coach" technology for learning under imperfect info. Techniques directly transferable to poker opponent modeling.

### 3.7 RLCard Toolkit
- **Type:** Framework / toolkit
- **Source:** GitHub: [datamllab/rlcard](https://github.com/datamllab/rlcard), arXiv:1910.04376
- **Authors:** Daochen Zha et al. (Rice/Texas A&M)
- **Relevance:** **HIGH** -- RL toolkit supporting Blackjack, Leduc Hold'em, Texas Hold'em, DouDizhu, Mahjong, UNO. Standardized interfaces for algorithm development. Easy benchmarking environment for poker AI research.

---

## 4. Multi-Agent RL and Training Frameworks

### 4.1 OpenRL
- **Type:** RL framework
- **Source:** GitHub: [OpenRL-Lab/openrl](https://github.com/OpenRL-Lab/openrl), arXiv:2312.16189
- **Origin:** China (academic)
- **Relevance:** **MEDIUM** -- Unified interface for single-agent, multi-agent, offline RL, self-play, NLP tasks. Arena module for competitive agent evaluation. PyTorch-based.

### 4.2 MARTI (Tsinghua C3I)
- **Type:** Multi-agent RL framework for LLMs
- **Source:** GitHub: [TsinghuaC3I/MARTI](https://github.com/TsinghuaC3I/MARTI)
- **Origin:** Tsinghua University, Beijing
- **Relevance:** **HIGH** -- Built on OpenRLHF. Supports centralized multi-agent interactions + distributed policy training. Multi-turn async rollouts. Dynamic workflows with rule-based verifiable rewards + LLM-based generative rewards. Implements REINFORCE++, GRPO, PPO. MARTI-v2 adds tree search augmented RL.

### 4.3 OpenRLHF
- **Type:** RLHF framework
- **Source:** GitHub: [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), arXiv:2405.11143
- **Origin:** China (community-driven)
- **Relevance:** **MEDIUM** -- Ray + vLLM architecture. Scales to 70B+ models. PPO, REINFORCE++, GRPO, RLOO. 1.22x-1.68x speedup over alternatives. Foundation for MARTI and other multi-agent RL work.

### 4.4 MARFT (Multi-Agent Reinforcement Fine-Tuning)
- **Type:** Paper + framework
- **Source:** arXiv:2504.16129 (April 2025)
- **Authors:** Junwei Liao, Muning Wen, Jun Wang, Weinan Zhang
- **GitHub:** [jwliao-ai/MARFT](https://github.com/jwliao-ai/MARFT)
- **Relevance:** **MEDIUM** -- Novel MARFT paradigm for LLM-based MARL. Flex-MG game formulation for multi-agent LLM systems. Universal algorithmic framework. Multi-agent systems outperform single-agent within same inference budget.

### 4.5 Safe-RLHF (PKU Alignment)
- **Type:** Framework + dataset
- **Source:** GitHub: [PKU-Alignment/safe-rlhf](https://github.com/PKU-Alignment/safe-rlhf), arXiv:2406.15513
- **Origin:** Peking University, Beijing
- **Relevance:** **LOW** -- Constrained value alignment via safe RL. ICLR 2024 Spotlight. Separate helpfulness/harmlessness optimization. Concept of constrained optimization applicable to poker strategy safety bounds.

---

## 5. Game Theory + LLM Research

### 5.1 Game Theory Meets LLMs: A Systematic Survey
- **Type:** Survey paper
- **Source:** arXiv:2502.09053 (Feb 2025), IJCAI 2025
- **Authors:** Haoran Sun, Yusen Wu, Peng Wang, Wei Chen, Yukun Cheng, Xiaotie Deng, Xu Chu (Peking University)
- **Relevance:** **HIGH** -- First comprehensive survey of bidirectional GT<->LLM relationship. Four perspectives: evaluating LLMs in games, improving LLMs with GT concepts, using LLMs for GT problems, co-evolution. Essential reference for poker AI + LLM integration.

### 5.2 LLM Strategic Reasoning via Behavioral Game Theory
- **Type:** Paper
- **Source:** arXiv:2502.20432 (Feb 2025)
- **Authors:** Jingru Jia, Zehua Yuan, Junhao Pan, Paul McNamara, Deming Chen
- **Relevance:** **MEDIUM** -- Tests 22 LLMs including DeepSeek-R1 in game-theoretic settings. GPT-o3-mini, GPT-o1, DeepSeek-R1 dominate most games. Chain-of-Thought not universally effective. Demographic features impact decision patterns. Directly informs LLM selection for poker.

### 5.3 Understanding LLM Agent Behaviours via Game Theory
- **Type:** Paper
- **Source:** arXiv:2512.07462 (Dec 2025)
- **Authors:** Trung-Kiet Huynh et al. (Vietnamese + international team)
- **Relevance:** **MEDIUM** -- FAIRGAME framework for evaluating LLM behavior in repeated social dilemmas. Model-specific behavioral biases resist personality prompting. Linguistic framing as strategic variable. Relevant to modeling LLM-based synthetic opponents.

### 5.4 Learning in Games: A Systematic Review
- **Type:** Survey paper
- **Source:** Science China Information Sciences, 2024, 67:171101
- **URL:** [link.springer.com/article/10.1007/s11432-023-3955-x](https://link.springer.com/article/10.1007/s11432-023-3955-x)
- **Origin:** Chinese institution
- **Relevance:** **MEDIUM** -- Covers fictitious play, no-regret learning, online learning advances, deep RL breakthroughs. Documents evolution from search+learning to pure RL methods. Comprehensive reference for game-solving algorithm selection.

---

## 6. Behavioral Science / Cognitive Bias Research

### 6.1 Prospect Theory Fails for LLMs
- **Type:** Paper
- **Source:** arXiv:2508.08992 (Aug 2025), GitHub: [HKUST-KnowComp/MarPT](https://github.com/HKUST-KnowComp/MarPT)
- **Authors:** Rui Wang, Qihan Lin, Jiayu Liu, Qing Zong, Tianshi Zheng, Weiqi Wang, Yangqiu Song (HKUST, Hong Kong)
- **Relevance:** **HIGH** -- Shows Prospect Theory fails to model LLM decision-making under epistemic uncertainty. Smaller models fail PT-like behavior; larger models (Qwen2.5-32B) show better alignment. Epistemic markers disrupt decision consistency. Critical for cognitive bias exploitation module: real human biases differ from LLM biases.

### 6.2 Behavioral Economics of AI: LLM Biases and Corrections
- **Type:** Paper
- **Source:** arXiv:2602.09362 (Feb 2026)
- **Relevance:** **MEDIUM** -- Documents LLM biases in stock return forecasts, inequality aversion, malleable preferences in allocation games. Relevant to modeling decision biases in poker opponents.

### 6.3 Bias-Adjusted LLM Agents for Human-Like Decision-Making
- **Type:** Paper
- **Source:** arXiv:2508.18600 (Aug 2025)
- **Authors:** Ayato Kitadai, Yusuke Fukasawa, Nariaki Nishino (University of Tokyo)
- **Relevance:** **MEDIUM** -- Persona-based approach using Econographics dataset to adjust LLM biases. Applied to ultimatum game. Improves alignment between simulated and empirical behavior. Directly applicable to synthetic player generation with realistic bias profiles.

### 6.4 Investigating Impact of LLM Personality on Cognitive Bias
- **Type:** Paper
- **Source:** arXiv:2502.14219 (Feb 2025)
- **Authors:** Shengjie Ma, Honghao Liu et al.
- **Relevance:** **MEDIUM** -- How LLM personality traits manifest as cognitive biases in automated decision-making. Relevant to personality-driven opponent modeling.

### 6.5 Multi-Agent Cooperative Decision-Making Survey
- **Type:** Survey paper
- **Source:** arXiv:2503.13415 (March 2025)
- **Authors:** Weiqiang Jin, Hongyang Du et al. (University of Hong Kong)
- **Relevance:** **MEDIUM** -- Covers rule-based, game-theory, evolutionary, deep MARL, and LLM reasoning approaches. Comprehensive taxonomy for multi-agent decision systems. Hong Kong affiliation.

---

## 7. Datasets

### 7.1 PokerBench
- **Type:** Dataset (poker scenarios)
- **Source:** [huggingface.co/datasets/RZ412/PokerBench](https://huggingface.co/datasets/RZ412/PokerBench), GitHub: [pokerllm/pokerbench](https://github.com/pokerllm/pokerbench)
- **Paper:** arXiv:2501.08328, AAAI 2025
- **Size:** Pre-flop: 60K train + 1K test; Post-flop: 500K train + 10K test
- **Relevance:** **HIGH** -- 11,000 critical poker scenarios with solver-optimal decisions in NL Hold'em. Natural language prompts + structured game data. Validated: higher benchmark scores correlate with higher win rates. Fine-tuning Llama-3-8B reached 78.26% accuracy.

### 7.2 PKU-SafeRLHF Dataset
- **Type:** Dataset (human preference / safety alignment)
- **Source:** [huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF)
- **Paper:** arXiv:2406.15513
- **Size:** 44.6K prompts, 265K QA pairs, 166.8K preference pairs
- **Origin:** Peking University
- **Relevance:** **LOW** -- Separate helpfulness/harmlessness annotations. 19 harm categories, 3 severity levels. Useful as reference for dual-objective preference optimization methodology (applicable to balancing exploitation vs. risk in poker).

### 7.3 RLCard Environments
- **Type:** Simulated game environments (dataset generation)
- **Source:** GitHub: [datamllab/rlcard](https://github.com/datamllab/rlcard)
- **Games:** Texas Hold'em, Leduc Hold'em, DouDizhu, Mahjong, Blackjack, UNO
- **Relevance:** **HIGH** -- Generate unlimited training data for poker and imperfect information games. Standard API for RL agent training. Widely used benchmark platform.

---

## 8. Tencent AI Lab Game-Solving Research Group

This group (Haobo Fu, Kai Li, Qiang Fu, Hang Xu, Weiming Liu, Junliang Xing) at Tencent + CASIA has produced a concentrated body of work directly relevant to poker AI:

| Paper | Venue | Key Innovation |
|-------|-------|----------------|
| AutoCFR | AAAI 2022 Oral | Meta-learning to design CFR algorithms |
| ACH (Actor-Critic Hedge) | ICLR 2022 | Actor-critic for large-scale IIGs (Mahjong) |
| Opponent-Limited Online Search | ICML 2023 | Safe subgame solving with limited opponent modeling |
| Dynamic Discounted CFR | ICLR 2024 Spotlight | Dynamic discounting for CFR convergence |
| PDCFR+ | IJCAI 2024 Oral | Weighted CFR + optimistic mirror descent |
| Deep Predictive Discounted CFR | arXiv:2511.08174, Nov 2025 | Model-free neural CFR with variance reduction |

**Relevance to MR_POKER:** This group's body of work covers the entire CFR solver pipeline. Their code releases (PDCFRPlus, DecisionHoldem) provide implementation references. The progression from AutoCFR to Deep Predictive DCFR shows a clear evolution toward practical, scalable poker solving.

---

## 9. CASIA Poker Abstraction Research Group

Yanchang Fu and Kaiqi Huang at the Institute of Automation, Chinese Academy of Sciences:

| Paper | Date | Key Innovation |
|-------|------|----------------|
| Signal Observation Models + KrwEmd | arXiv:2403.11486, Mar 2024 | Historical info in hand abstraction; k-recall winrate |
| Embedding CFR | arXiv:2511.12083, Nov 2025 | Pre-trained embeddings replace discrete clustering |

**Relevance to MR_POKER:** Directly addresses the information set abstraction problem that is central to scaling CFR to full No-Limit Hold'em. Embedding-based abstraction is a natural fit for neural network integration.

---

## 10. Relevance Matrix

| MR_POKER Component | Most Relevant Resources |
|--------------------|------------------------|
| **CFR Solver** | Deep Predictive DCFR (#2.6), PDCFR+ (#2.7), Embedding CFR (#2.3), DecisionHoldem (#2.1), AutoCFR (#2.10) |
| **Opponent Modeling** | Opponent-Limited Online Search (#2.8), Signal Observation Models (#2.2), DouZero+ (opponent modeling), ACH (#2.9) |
| **Behavioral Prediction** | Prospect Theory Fails (#6.1), LLM Personality/Bias (#6.4), Behavioral Economics of AI (#6.2), Game Theory Meets LLMs (#5.1) |
| **Skill Estimation** | PokerBench (#7.1), ToolPoker (#2.5), LLM Strategic Reasoning (#5.2) |
| **Synthetic Player Generation** | Bias-Adjusted LLM Agents (#6.3), DeepSeek-R1 Distilled (#1.11), MARTI (#4.2), MARFT (#4.4) |
| **Cognitive Bias Exploitation** | Prospect Theory Fails (#6.1), LLM Behavioural Biases (#5.3), PokerGPT RLHF pipeline (#2.4) |
| **Training Infrastructure** | OpenRLHF (#4.3), MARTI (#4.2), RLCard (#3.7), OpenRL (#4.1) |
| **Foundation Model Backbone** | DeepSeek-R1/Distilled (#1.1, #1.11), Qwen3 (#1.3), GLM-4 (#1.5) |

---

## 11. Recommendations for MR_POKER Integration

### Immediate (High Priority)
1. **Integrate PDCFR+ / Deep Predictive DCFR** -- Replace or augment current CFR solver with latest Tencent/CASIA advances. Code available at [PDCFRPlus](https://github.com/rpSebastian/PDCFRPlus).
2. **Adopt Embedding CFR abstraction** -- Replace discrete clustering in information set abstraction with pre-trained embeddings (arXiv:2511.12083).
3. **Use PokerBench for evaluation** -- 560K labeled poker scenarios provide immediate evaluation infrastructure.
4. **Fine-tune DeepSeek-R1-Distill-Qwen-32B** on poker reasoning -- Best cost/performance ratio for reasoning backbone.
5. **Study ToolPoker framework** -- External GTO solver integration with LLM reasoning is directly applicable to MR_POKER's architecture.

### Medium-Term
6. **Adapt MARTI framework** for multi-agent poker training with verifiable rewards.
7. **Implement oracle guiding** (from OADMCDou / Suphx) for training with perfect information that transfers to imperfect-info play.
8. **Build synthetic player profiles** using Bias-Adjusted LLM Agents methodology with Econographics-style behavioral indicators.

### Research Monitoring
9. Track Tencent AI Lab game-solving group for new publications.
10. Track CASIA poker abstraction group (Fu, Huang) for embedding/abstraction advances.
11. Monitor HKUST KnowComp for cognitive bias + LLM research.

---

## 12. Resource Count Summary

| Category | Count |
|----------|-------|
| Foundation Models | 11 |
| Poker-Specific Papers/Systems | 10 |
| Imperfect Info Game AI (non-poker) | 7 |
| Multi-Agent RL Frameworks | 5 |
| Game Theory + LLM Research | 4 |
| Behavioral Science / Cognitive Bias | 5 |
| Datasets | 3 |
| Research Groups Profiled | 2 |
| **Total Unique Resources** | **52** |

| Region | Count |
|--------|-------|
| Mainland China | 43 |
| Hong Kong | 3 |
| Taiwan | 1 (NTU reference in surveys) |
| Cross-regional / International collaborations | 5 |
