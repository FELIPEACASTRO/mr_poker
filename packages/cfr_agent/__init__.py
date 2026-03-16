"""CFR (Counterfactual Regret Minimization) poker agent.

Uses self-play to learn mixed-strategy Nash equilibrium approximations.
Supports Vanilla CFR, Discounted CFR (DCFR), and Monte Carlo CFR (MCCFR).
Deep CFR extends tabular CFR with neural network approximation.

Advanced variants:
- Regret-based Pruning: 10x speedup by skipping low-regret actions
- Compact CFR: 8-16x memory reduction via int8 quantization
- Warm Starting: 2-5x faster convergence with prior strategy
- Lazy-CFR: Partial updates for reduced per-iteration cost
"""

from packages.cfr_agent.agent import CFRAgent
from packages.cfr_agent.trainer import CFRTrainer
from packages.cfr_agent.pruning import RegretPruningCFRTrainer
from packages.cfr_agent.compact_cfr import CompactCFRState
from packages.cfr_agent.warm_start import (
    warm_start_from_strategy,
    warm_start_from_cfr_state,
    extract_strategy_from_cfr,
)
from packages.cfr_agent.lazy_cfr import LazyCFRTrainer, LazyCFRConfig
from packages.cfr_agent.gpu_cfr import BatchCFRTrainer
from packages.cfr_agent.embedding_cfr import EmbeddingCFRTrainer
from packages.cfr_agent.hdcfr import HDCFRTrainer, HierarchicalPolicy, Skill
from packages.cfr_agent.qre import QRESolver, QREState
from packages.cfr_agent.vr_deep_dcfr import VRDeepDCFRTrainer
from packages.cfr_agent.exploitability import ExploitabilityCalculator
from packages.cfr_agent.nfsp import NFSPAgent

__all__ = [
    "CFRAgent",
    "CFRTrainer",
    "RegretPruningCFRTrainer",
    "CompactCFRState",
    "warm_start_from_strategy",
    "warm_start_from_cfr_state",
    "extract_strategy_from_cfr",
    "LazyCFRTrainer",
    "LazyCFRConfig",
    "BatchCFRTrainer",
    "EmbeddingCFRTrainer",
    "HDCFRTrainer",
    "HierarchicalPolicy",
    "Skill",
    "QRESolver",
    "QREState",
    "VRDeepDCFRTrainer",
    "ExploitabilityCalculator",
    "NFSPAgent",
]
