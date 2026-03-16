# Global AI Research Report: Poker AI, Game Theory, and Behavioral Modeling

> Comprehensive search across arXiv, HuggingFace, GitHub, and academic databases
> Focus: 2024-2026 resources verified through direct search
> Date: 2026-03-16

---

## 1. POKER-SPECIFIC AI

### 1.1 Benchmarks and Evaluation

| # | Resource | Type | ID | Key Contribution | Relevance to Poker AI | Year |
|---|----------|------|----|-------------------|----------------------|------|
| 1 | **PokerBench** | Paper/Dataset | arXiv:2501.08328 | 11,000 poker scenarios (1K pre-flop, 10K post-flop) evaluating LLM poker ability. GPT-4 achieves 53.55% accuracy; fine-tuning dramatically improves play. AAAI 2025. | Direct benchmark for poker LLM evaluation | 2025 |
| 2 | **How Far Are LLMs from Professional Poker Players?** | Paper | arXiv:2602.00528 | Systematic study of LLMs in poker with tool-integrated reasoning framework combining external solvers with professional-style explanations. Identifies reasoning-action gaps. | Advances agentic poker AI with solver integration | 2026 |
| 3 | **PokerGPT** | Paper | arXiv:2401.06781 | End-to-end lightweight LLM solver for multi-player Texas Hold'em using RLHF fine-tuning. Outperforms prior approaches in multiplayer settings. | Lightweight poker LLM architecture | 2024 |
| 4 | **Are ChatGPT and GPT-4 Good Poker Players?** | Paper | arXiv:2308.12466 | Pre-flop analysis showing GPT-4 and ChatGPT deviate from GTO; GPT-4 plays aggressively, ChatGPT conservatively. Tests persona-based prompting (nit, maniac). | Baseline for LLM poker behavior | 2023 |
| 5 | **A Survey on Game Theory Optimal Poker** | Survey | arXiv:2401.06168 | Comprehensive survey covering GTO vs exploitative poker, abstraction techniques, betting models, and strategies of Tartanian/Pluribus. | Foundational survey for poker AI landscape | 2024 |
| 6 | **Beyond Game Theory Optimal: Profit-Maximizing Poker Agents** | Paper | arXiv:2509.23747 | Two-phase approach: first converge to GTO baseline via self-play, then adapt to exploit opponent behavior. MCCFR performs best in heads-up. | Directly relevant exploit-over-GTO architecture | 2025 |
| 7 | **Playing the Player: A Heuristic Framework for Adaptive Poker AI** | Paper | arXiv:2512.04714 | "Patrick" AI engine: maximally exploitative rather than unexploitable. Opponent modeling builds statistical profiles to attack psychological weaknesses. Profitable over 64,267 hands against 7,159 players. | Core reference for exploitative opponent modeling | 2025 |

### 1.2 Poker Datasets

| # | Resource | Type | ID | Key Contribution | Relevance to Poker AI | Year |
|---|----------|------|----|-------------------|----------------------|------|
| 8 | **PokerBench Dataset** | HF Dataset | HF: RZ412/PokerBench | Natural language game scenarios with optimal solver-computed decisions. Pre-flop and post-flop splits with train/test. | Direct training data for poker LLMs | 2025 |
| 9 | **PHH Dataset (Poker Hand Histories)** | GitHub/Zenodo | github.com/uoftcprg/phh-dataset | 21.6M uncorrupted no-limit hold'em hands across 11 variants in standardized PHH format. | Massive-scale training corpus for poker models | 2024 |
| 10 | **PHS Dataset (Poker Hand Strengths)** | GitHub | github.com/uoftcprg/phs-dataset | Pre-computed poker hand strength tables. | Lookup tables for equity computation | 2024 |

### 1.3 Poker Toolkits and Environments

| # | Resource | Type | ID | Key Contribution | Relevance to Poker AI | Year |
|---|----------|------|----|-------------------|----------------------|------|
| 11 | **OpenHoldem** | Toolkit/Paper | arXiv:2012.06168 | Standardized evaluation protocol for NLTH AIs with four baselines (rule-based, CFR, DeepStack-like, deep RL). Online testing platform with APIs. | Standard benchmark infrastructure for poker AI | 2020 |
| 12 | **RLCard** | Toolkit/Paper | arXiv:1910.04376 | RL toolkit specifically for card games (Blackjack, Leduc, Texas Hold'em, UNO, Dou Dizhu, Mahjong). Bridges RL and imperfect information games. | Training environment for poker RL agents | 2019 |
| 13 | **DecisionHoldem** | Paper | arXiv:2201.11580 | Depth-limited subgame solving for Texas Hold'em. Outperforms top agents by >700 mbb/h. Open-source. | State-of-the-art subgame solving reference | 2022 |

---

## 2. CFR VARIANTS AND ADVANCES

| # | Resource | Type | ID | Key Contribution | Relevance to Poker AI | Year |
|---|----------|------|----|-------------------|----------------------|------|
| 14 | **DCFR (Discounted CFR)** | Paper | arXiv:1809.04040 | Discounts regrets from earlier iterations; outperforms CFR+ in every game tested including large poker subgames. Enables order-of-magnitude pruning. By Brown/Sandholm. | Foundation algorithm for poker solving | 2019 |
| 15 | **Deep (Predictive) Discounted CFR** | Paper | arXiv:2511.08174 | Model-free neural CFR using variance-reduced sampled advantages, bootstrapping, discounting/clipping to simulate advanced CFR variants. Faster convergence and stronger performance in large poker games. | Next-gen neural CFR for large poker | 2025 |
| 16 | **Robust Deep Monte Carlo CFR** | Paper | arXiv:2509.00923 | Addresses scale-dependent theoretical risks in neural MCCFR (distribution shifts, action collapse, variance explosion). 63.5% improvement on Kuhn Poker, 29.5% on Leduc. | Robust neural MCCFR framework | 2025 |
| 17 | **Discovering Multiagent Learning Algorithms (AlphaEvolve)** | Paper | arXiv:2602.16928 | DeepMind's LLM-powered evolutionary agent discovers novel CFR variants: Volatility-Adaptive Discounted CFR (VAD-CFR) and SHOR-PSRO. Outperforms hand-designed DCFR+. | Automated discovery of better CFR variants | 2026 |
| 18 | **Comparative Analysis of Extensive-Form Zero-Sum Game Algorithms** | Paper | Nature Sci. Reports 15:2917 | Comprehensive comparison of MCCFR, DCFR, CFR+, DeepCFR, NFSP, PSRO on Kuhn/Leduc/Royal Poker (2-5 players). DCFR best for scalability; MCCFR clearest GTO convergence. | Definitive algorithm comparison for poker | 2025 |
| 19 | **Monopoly Deal as CFR Benchmark** | Paper | arXiv:2510.25080 | Bounded One-Sided Response Games (BORGs) as new benchmark for CFR in structured environments beyond poker. | Extends CFR benchmarking to new game types | 2025 |

---

## 3. IMPERFECT INFORMATION GAME SOLVING

### 3.1 New Algorithms

| # | Resource | Type | ID | Key Contribution | Relevance to Poker AI | Year |
|---|----------|------|----|-------------------|----------------------|------|
| 20 | **Policy-Gradient Approach with Best-Iterate Convergence** | Paper | arXiv:2408.00751 | First principled policy gradient for imperfect-info EFGs with best-iterate convergence. Uses trajectory Q-values and bidilated regularizer. No importance sampling needed. ICLR 2025. | Alternative to CFR for poker solving | 2024 |
| 21 | **Reevaluating Policy Gradient Methods for Imperfect-Info Games** | Paper | arXiv:2502.08938 | Over 5600 training runs show generic PPO is competitive with or superior to FP/DO/CFR-based DRL approaches. First broadly accessible exact exploitability computations for four large games. | May simplify poker AI training pipelines | 2025 |
| 22 | **Adapting to Game Trees in Zero-Sum Imperfect-Info Games** | Paper | arXiv:2212.12567 | Balanced FTRL and Adaptive FTRL algorithms with theoretical bounds for learning in imperfect-info games via self-play. | Theoretical foundations for poker self-play | 2022 |
| 23 | **Decision Making under Imperfect Recall** | Paper | arXiv:2602.15252 | First benchmark suite for imperfect-recall problems (61 instances). Introduces regret matching for nonlinear constrained optimization beyond two-player zero-sum. By Sandholm/Conitzer team. | Extends game-solving to imperfect recall settings | 2026 |
| 24 | **Solving Football via 2p0s Differential Games** | Paper | arXiv:2502.00560 | Equilibrium strategies concentrate on limited action prototypes, reducing game tree complexity. Applies MARL and model predictive control. | Technique for reducing poker action space | 2025 |

### 3.2 Benchmarks and Frameworks

| # | Resource | Type | ID | Key Contribution | Relevance to Poker AI | Year |
|---|----------|------|----|-------------------|----------------------|------|
| 25 | **OpenSpiel** | Framework/Paper | arXiv:1908.09453 | DeepMind's comprehensive framework: n-player, zero/general-sum, perfect/imperfect information games. Includes Kuhn/Leduc poker, evaluation tools. | Standard framework for game AI research | 2019 |
| 26 | **Valet** | Benchmark/Paper | arXiv:2603.03252 | 21 traditional imperfect-information card games encoded in RECYCLE. Standardized benchmarking suite with MCTS baselines. | Diverse card game benchmarks beyond poker | 2026 |
| 27 | **OpenGuanDan** | Benchmark/Paper | arXiv:2602.00676 | Large-scale 4-player Chinese card game benchmark. 25M steps/hour. RL agents outperform rule-based but not yet superhuman. | Multiplayer imperfect-info game benchmark | 2026 |
| 28 | **Open RL Benchmark** | Framework/Paper | arXiv:2402.03046 | Comprehensive database of reproducible RL experiments with detailed metrics and analysis tools. | Infrastructure for reproducible game AI | 2024 |

---

## 4. OPPONENT MODELING AND BEHAVIORAL PREDICTION

| # | Resource | Type | ID | Key Contribution | Relevance to Poker AI | Year |
|---|----------|------|----|-------------------|----------------------|------|
| 29 | **Model-Based Opponent Modeling** | Paper | arXiv:2108.01843 | Simulates and mixes imagined opponent policies based on real behaviors. Adapts to fixed, naive, and reasoning opponents. | Core opponent modeling architecture | 2021 |
| 30 | **Suspicion-Agent: ToM-Aware GPT-4** | Paper | arXiv:2309.17277 | Theory of Mind framework for GPT-4 in imperfect-info card games (Leduc Hold'em). Predicts opponents' thought processes and susceptibilities. | ToM-based poker opponent modeling | 2023 |
| 31 | **HARBOR: Persona Dynamics in Multi-Agent Competition** | Paper | arXiv:2502.12149 | LLM agents in competitive auctions use personas, memory of history, and theory of mind to profile competitors and gain advantages. | Persona-based competitive profiling | 2025 |
| 32 | **K-Level Reasoning with LLMs** | Paper | arXiv:2402.01521 | Recursive k-level thinking for strategic decision-making. Enhances LLM performance in dynamic interactive scenarios. | Models depth of reasoning in opponent modeling | 2024 |
| 33 | **Minimax Exploiter** | Paper | arXiv:2311.17190 | Game-theory enhanced competitive self-play. Minimax Exploiter improves data efficiency and stability for exploit agents. | Efficient exploiter training for poker | 2023 |
| 34 | **General Social Agents** | Paper | arXiv:2508.17407 | AI agents trained on seed games predict human behavior in novel settings without theory modifications. Outperforms traditional cognitive models. | Human behavior prediction transferable to poker | 2025 |

---

## 5. SKILL ESTIMATION AND RATING SYSTEMS

| # | Resource | Type | ID | Key Contribution | Relevance to Poker AI | Year |
|---|----------|------|----|-------------------|----------------------|------|
| 35 | **Chess Rating from Moves and Clock Times (CNN-LSTM)** | Paper | arXiv:2409.11506 | CNN+BiLSTM estimates chess ratings from moves and clock times without hand-crafted features. MAE of 182 rating points. 1.2M games from Lichess. | Architecture transferable to poker skill estimation from betting patterns | 2024 |
| 36 | **OpenSkill** | Paper/Library | arXiv:2401.05451 | Bayesian Plackett-Luce rating system for asymmetric multi-team multiplayer games. 3x faster than TrueSkill. Five distinct models. github.com/vivekjoshy/openskill.py | Direct application for poker player rating | 2024 |
| 37 | **Maia-2: Unified Human-AI Alignment in Chess** | Paper | arXiv:2409.20553 | Skill-aware attention mechanism captures human decision-making across skill levels. NeurIPS 2024. Most accurate human move predictor. | Architecture for skill-conditioned behavior modeling | 2024 |
| 38 | **Maia: Aligning Superhuman AI with Human Behavior** | Paper | arXiv:2006.01855 | Customized AlphaZero predicting human chess moves at granular skill levels. Personalized models surface idiosyncratic mistakes. | Foundational work on human-like AI behavior | 2020 |
| 39 | **Behavior-Based Knowledge Representation for Chess** | Paper | arXiv:2504.05425 | Expert knowledge + ML to predict human moves. Domain-specific feature engineering uncovers patterns in intermediate players. 25% prediction improvement. | Behavior representation transferable to poker | 2025 |
| 40 | **Predicting Human Chess Moves with Skill-Group n-grams** | Paper | arXiv:2512.01880 | Skill-group specific n-gram language models for move prediction. Seven skill groups from novice to expert. Computationally efficient. | Skill-stratified behavioral modeling applicable to poker | 2025 |
| 41 | **Elo Uncovered: Robustness and Best Practices** | Paper | arXiv:2311.17295 | Studies Elo reliability and transitivity in LLM evaluation. Reveals volatility and suggests enhanced evaluation methods. | Rating system design principles | 2023 |
| 42 | **Multiagent Evaluation under Incomplete Information** | Paper | arXiv:1909.09849 | Compares Elo and alpha-Rank in noisy, incomplete information settings including Kuhn poker. Adaptive algorithms with sample complexity guarantees. | Rating methods specifically for poker-like games | 2019 |

---

## 6. SYNTHETIC DATA AND PERSONA GENERATION

| # | Resource | Type | ID | Key Contribution | Relevance to Poker AI | Year |
|---|----------|------|----|-------------------|----------------------|------|
| 43 | **Persona Hub** | Paper | arXiv:2406.20094 | 1 billion diverse personas for LLM-driven synthetic data creation. Applications include game NPCs, reasoning, knowledge-rich texts. 104 upvotes on HF. | Persona-driven synthetic poker player generation | 2024 |
| 44 | **DeepPersona** | Paper | arXiv:2511.07338 | Taxonomy-guided generative engine producing narrative-complete synthetic personas (~1MB text, hundreds of attributes). 32% higher coverage, 44% greater uniqueness vs baselines. NeurIPS LAW 2025. | Rich persona generation for diverse poker player profiles | 2025 |
| 45 | **Generative Agent Simulations of 1,000 People** | Paper | arXiv:2411.10109 | Novel architecture simulating individual behaviors using LLMs. High accuracy replicating General Social Survey responses. Reduces demographic biases. | Individual-level behavioral simulation for poker | 2024 |
| 46 | **PERSONA: Reproducible Testbed for Pluralistic Alignment** | Paper | arXiv:2407.17387 | Procedurally generated synthetic personas with demographic and idiosyncratic attributes. PERSONA Bench for evaluation. | Framework for generating diverse player types | 2024 |
| 47 | **Socially-Grounded Persona Framework** | Paper | arXiv:2601.07110 | Sociopsychological frameworks capture behavioral alignment better than demographic attributes alone. | Better persona construction for poker player simulation | 2026 |
| 48 | **PersonaEvolve (PEBA Framework)** | Paper | arXiv:2509.16457 | Refines agent personas for distributional behavioral realism in high-stakes crowd simulations. | Behavioral alignment for poker agent populations | 2025 |

---

## 7. COGNITIVE BIAS AND DECISION-MAKING UNDER UNCERTAINTY

| # | Resource | Type | ID | Key Contribution | Relevance to Poker AI | Year |
|---|----------|------|----|-------------------|----------------------|------|
| 49 | **Prospect Theory Emerges in LLMs** | Paper | arXiv:2508.00902 | First tests of Kahneman/Tversky's prospect theory with LLMs (GPT-4o, o3, Claude Sonnet 4, Gemini). Semantic context drives risk patterns. | Models loss aversion and framing effects exploitable in poker | 2025 |
| 50 | **Prospect Theory Fails for LLMs** | Paper | arXiv:2508.08992 | Epistemic uncertainty causes instability in LLM decision-making. Prospect theory patterns break down under linguistic uncertainty. | Limits of cognitive bias modeling in AI poker agents | 2025 |
| 51 | **Robust Mean-Field Games with Risk Aversion and Bounded Rationality** | Paper | arXiv:2602.13353 | Mean-field risk-averse quantal response equilibrium (MF-RQE) modeling bounded rationality deviations. | Mathematical framework for bounded-rational poker agents | 2026 |
| 52 | **Bounded Risk-Sensitive Markov Games** | Paper | arXiv:2009.01495 | Framework capturing bounded intelligence and risk-sensitivity. Cumulative Prospect Theory integrated with iterative reasoning and inverse reward learning. | Direct model for risk-sensitive poker decision-making | 2020 |
| 53 | **Seeing Through Risk: Symbolic Prospect Theory** | Paper | arXiv:2504.14448 | Transparent symbolic modeling replacing opaque utility curves with interpretable effect-size-guided features. | Interpretable risk modeling for poker AI | 2025 |
| 54 | **Centaur: Foundation Model of Human Cognition** | Paper/Model | arXiv:2410.20268, HF: marcelbinz/Llama-3.1-Centaur-70B | Llama 3.1 70B fine-tuned on Psych-101 (60K participants, 10M choices, 160 experiments). Predicts human behavior across cognitive domains. Published in Nature 2025. | Foundation for modeling human cognitive biases in poker | 2024 |
| 55 | **Emergence of Strategic Reasoning in LLMs** | Paper | arXiv:2412.13013 | LLMs show varied strategic reasoning matching/exceeding human performance. Evaluates hierarchical models of bounded rationality. | Characterizes LLM strategic reasoning levels for poker | 2024 |
| 56 | **Drift Diffusion Model for Preference Estimation** | Paper | arXiv:2507.20403 | Methodology for estimating preference parameters from choice and response time data. Fast convergence rates for DDM. | Models timing-based decision processes in poker | 2025 |
| 57 | **Playing Games with LLMs: Randomness and Strategy** | Paper | arXiv:2503.02582 | LLMs struggle with randomness, converge to predictable patterns, exhibit loss aversion in repeated games (RPS, Prisoner's Dilemma). | Documents exploitable LLM biases in games | 2025 |

---

## 8. MULTI-AGENT RL FOR GAMES

| # | Resource | Type | ID | Key Contribution | Relevance to Poker AI | Year |
|---|----------|------|----|-------------------|----------------------|------|
| 58 | **SPIRAL: Self-Play on Zero-Sum Games** | Paper | arXiv:2506.24119 | Self-play framework on Kuhn Poker/TicTacToe/Negotiation. Transfers to 8.6% math reasoning improvement. Role-conditioned advantage estimation (RAE). 51 upvotes on HF. | Self-play poker training with reasoning transfer | 2025 |
| 59 | **NeuPL: Neural Population Learning** | Paper | arXiv:2202.07415 | Represents multiple policies in single conditional model. Enables transfer learning and strategy diversity. Convergence guarantees. | Efficient multi-strategy poker agent | 2022 |
| 60 | **Survey on Self-play Methods in RL** | Survey | arXiv:2408.01072 | Unified framework classifying self-play algorithms, applications, and challenges in multi-agent RL. | Comprehensive reference for poker self-play design | 2024 |
| 61 | **LSPO: Latent Space Policy Optimization (Werewolf)** | Paper | arXiv:2502.04686 | Maps text to discrete latent space for strategic learning using CFR, then fine-tunes via DPO. Improves reasoning and communication in social deduction. | CFR in latent space applicable to poker language agents | 2025 |
| 62 | **MARS: Multi-Agent Reasoning via Self-Play** | Paper | arXiv:2510.15414 | End-to-end RL framework with agent-specific advantage estimation and turn-level advantage estimator. Improves on AIME and GPQA benchmarks. | Multi-agent self-play architecture for poker | 2025 |
| 63 | **MaKTO: Multi-agent KTO for Werewolf** | Paper | arXiv:2501.14225 | Multi-agent Kahneman and Tversky's Optimization. Outperforms GPT-4 in social deduction with superior strategic adaptation. | Prospect-theory-based multi-agent optimization | 2025 |
| 64 | **SPC: Self-Play Critic via Adversarial Games** | Paper | arXiv:2504.19162 | Adversarial self-play improves error detection in reasoning chains without manual step-level annotation. | Self-play critic for poker decision validation | 2025 |
| 65 | **LLM Agents Have Regret? Online Learning and Games** | Paper | arXiv:2403.16843 | Studies LLM no-regret behavior in online learning/game theory. Introduces regret-loss for improving learning. | Regret minimization in LLM poker agents | 2024 |

---

## 9. GAME THEORY AND LLM INTERSECTION

| # | Resource | Type | ID | Key Contribution | Relevance to Poker AI | Year |
|---|----------|------|----|-------------------|----------------------|------|
| 66 | **Game Theory Meets LLMs: Systematic Survey** | Survey | arXiv:2502.09053 | First comprehensive bidirectional survey: evaluating LLMs in games, improving LLMs with game theory, modeling competitive LLM landscape, using LLMs for games. IJCAI 2025. | Definitive reference for game theory + LLM intersection | 2025 |
| 67 | **GTBench: Game-Theoretic Evaluation of LLMs** | Paper | arXiv:2402.12348 | Evaluates LLM strategic reasoning across complete/incomplete information games. Code-pretraining helps more than CoT/ToT. | Benchmark methodology for poker LLM evaluation | 2024 |
| 68 | **TMGBench: Systematic Game Benchmark** | Paper | arXiv:2410.10479 | Uses Robinson-Goforth topology to evaluate LLMs on diverse 2x2 game scenarios. Tests accuracy, consistency, and Theory-of-Mind. | Game-theoretic evaluation methodology | 2024 |
| 69 | **ALYMPICS: LLM Agents Meet Game Theory** | Paper | arXiv:2311.03220 | Systematic simulation framework using LLM agents for game theory research. Multi-round auction with scarce resources. | LLM game theory simulation framework | 2023 |
| 70 | **GAMEBoT: Transparent Assessment of LLM Reasoning** | Paper | arXiv:2412.13602 | Modular gaming arena with CoT prompts and rule-based ground truth. Addresses interpretability and data contamination. | Transparent evaluation of poker AI reasoning | 2024 |
| 71 | **LLMsPark: Evaluating LLMs in Strategic Gaming** | Paper | arXiv:2509.16610 | Multi-agent game theory benchmark with leaderboard rankings and scoring mechanisms for strategic intelligence. | Strategic evaluation framework | 2025 |

---

## 10. SOCIAL DEDUCTION AND RELATED GAME AI

| # | Resource | Type | ID | Key Contribution | Relevance to Poker AI | Year |
|---|----------|------|----|-------------------|----------------------|------|
| 72 | **MultiMind: Werewolf with Multimodal ToM** | Paper | arXiv:2504.18039 | Integrates facial expressions, vocal tones, ToM, and MCTS for social deduction. Multi-modal opponent modeling. | Multi-modal tell detection applicable to poker | 2025 |
| 73 | **Beyond Survival: Social Deduction with Human-Aligned Strategies** | Paper | arXiv:2510.11389 | High-quality Werewolf dataset with strategy-alignment evaluation including deception and counterfactual reasoning. | Deception modeling transferable to poker bluffing | 2025 |
| 74 | **Welfare Diplomacy** | Paper | arXiv:2310.08901 | Balanced variant of Diplomacy with cooperative AI benchmarks. Tests cooperation, negotiation, and strategy. | Negotiation and cooperation modeling for poker | 2023 |
| 75 | **Human-Level Competitive Pokemon via Offline RL** | Paper | arXiv:2504.04395 | Reconstructs first-person perspective from spectator logs for offline RL. Outperforms heuristic search and LLM approaches. | Offline RL from imperfect-info game logs | 2025 |
| 76 | **Quantitative Rule-Based Strategy in Classic Indian Rummy** | Paper | arXiv:2601.00024 | Metric optimization approach for card game strategy modeling. | Card game strategy methodology | 2025 |

---

## SUMMARY STATISTICS

| Category | Count | Key Takeaway |
|----------|-------|-------------|
| Poker-Specific AI | 13 | PokerBench (2025) is the gold standard; exploitative approaches outperform pure GTO |
| CFR Variants | 6 | Deep neural CFR (2025) and AlphaEvolve-discovered VAD-CFR (2026) push frontiers |
| Imperfect-Info Solving | 8 | Policy gradient methods competitive with CFR; Valet/OpenGuanDan expand benchmarks |
| Opponent Modeling | 6 | Theory of Mind + persona profiling + k-level reasoning define state of art |
| Skill Estimation | 8 | CNN-LSTM from moves (chess), OpenSkill for multiplayer, Maia-2 for skill-aware prediction |
| Synthetic Data/Personas | 6 | PersonaHub (1B personas), DeepPersona (taxonomy-guided), sociopsychological frameworks |
| Cognitive Bias/Decision | 9 | Prospect theory in LLMs confirmed; Centaur foundation model for human cognition |
| Multi-Agent RL | 8 | SPIRAL (self-play reasoning transfer), NeuPL (population learning), MaKTO (prospect-theory RL) |
| Game Theory + LLM | 6 | First comprehensive survey (IJCAI 2025); multiple strategic reasoning benchmarks |
| Social Deduction/Related | 5 | Werewolf/Diplomacy provide deception and ToM modeling transferable to poker |
| **TOTAL** | **76** | |

---

## HIGH-PRIORITY INTEGRATION RECOMMENDATIONS FOR MR_POKER

### Immediate Value (already partially implemented)
1. **PokerBench** (arXiv:2501.08328) - Already integrated as benchmark; update to latest evaluation protocol
2. **DCFR** (arXiv:1809.04040) - Already implemented; consider upgrading to Deep Predictive DCFR (arXiv:2511.08174)
3. **OpenSkill** (arXiv:2401.05451) - Directly applicable to player rating system

### High Priority (new capabilities)
4. **Playing the Player** (arXiv:2512.04714) - Patrick's exploit-first philosophy aligns with opponent modeling goals
5. **Chess Rating from Moves** (arXiv:2409.11506) - CNN-LSTM architecture transferable to poker skill estimation from betting patterns
6. **Centaur** (arXiv:2410.20268) - Foundation model for human cognitive biases; HF model available at marcelbinz/Llama-3.1-Centaur-70B
7. **Deep Predictive DCFR** (arXiv:2511.08174) - Direct upgrade to current CFR implementation
8. **Robust Deep MCCFR** (arXiv:2509.00923) - Scale-aware neural MCCFR with diagnostics

### Medium Priority (research directions)
9. **SPIRAL** (arXiv:2506.24119) - Self-play on Kuhn Poker transfers to math reasoning
10. **DeepPersona** (arXiv:2511.07338) - Taxonomy-guided persona generation for synthetic poker players
11. **AlphaEvolve CFR** (arXiv:2602.16928) - Automated discovery of novel CFR variants (VAD-CFR)
12. **Prospect Theory in LLMs** (arXiv:2508.00902) - Confirms exploitable cognitive biases in AI agents
13. **Policy Gradient for IIG** (arXiv:2408.00751) - Alternative to CFR with best-iterate convergence (ICLR 2025)

### Monitoring (emerging)
14. **Valet** (arXiv:2603.03252) - Standardized card game benchmarks (March 2026)
15. **Decision Making under Imperfect Recall** (arXiv:2602.15252) - New theoretical framework by Sandholm team
16. **Reevaluating Policy Gradient** (arXiv:2502.08938) - PPO may match CFR-based approaches (5600+ runs)
