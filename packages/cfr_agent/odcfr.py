"""ODCFR — Opponent-aware Deep CFR.

Integrates opponent modeling into the Deep CFR value function by extending
the feature vector with opponent statistics (VPIP, PFR, AF, etc.).

This allows the advantage network to learn opponent-conditional strategies,
automatically adjusting between GTO and exploitative play based on the
quality of available opponent information.

Reference: Inspired by Suspicion-Agent and opponent-aware MCCFR literature.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass

from packages.cfr_agent.deep_cfr import (
    ACTION_INDEX,
    FEATURE_DIM,
    NUM_ACTIONS,
    AdvantageMemory,
    SimpleNN,
)
from packages.common.types import ActionType
from packages.opponent_model.classifier import PlayerStats
from packages.strategy.mixed import ActionDistribution

logger = logging.getLogger(__name__)

# 8 opponent features appended to the standard feature vector
OPPONENT_FEATURE_DIM = 8
EXTENDED_FEATURE_DIM = FEATURE_DIM + OPPONENT_FEATURE_DIM  # 15 + 8 = 23


def extract_opponent_features(stats: PlayerStats) -> list[float]:
    """Extract normalized opponent features from PlayerStats.

    Returns 8 floats in [0, 1] range:
        0: VPIP
        1: PFR
        2: Aggression Factor (capped at 5, normalized)
        3: 3-bet percentage
        4: Fold to 3-bet percentage
        5: C-bet percentage
        6: WTSD (went to showdown)
        7: Fold percentage
    """
    return [
        min(stats.vpip, 1.0),
        min(stats.pfr, 1.0),
        min(stats.aggression_factor / 5.0, 1.0),
        min(stats.three_bet_pct, 1.0),
        min(stats.fold_to_3bet_pct, 1.0),
        min(stats.cbet_pct, 1.0),
        min(stats.wtsd, 1.0),
        min(stats.fold_pct, 1.0),
    ]


def default_opponent_features() -> list[float]:
    """Return neutral opponent features (unknown player)."""
    return [0.5, 0.2, 0.4, 0.1, 0.5, 0.5, 0.3, 0.3]


class OpponentAwareDeepCFR:
    """Deep CFR with opponent-conditioned advantage network.

    The advantage network takes an extended feature vector:
        [standard_features (15)] + [opponent_features (8)] = 23 dims

    This lets the network learn different strategies against different
    opponent types without explicit archetype classification.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        seed: int = 42,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.rng = random.Random(seed)

        # Extended advantage networks (one per player)
        self.adv_nets = [
            SimpleNN(EXTENDED_FEATURE_DIM, hidden_dim, NUM_ACTIONS, seed=seed),
            SimpleNN(EXTENDED_FEATURE_DIM, hidden_dim, NUM_ACTIONS, seed=seed + 1),
        ]
        # Extended advantage memories
        self.adv_memories = [AdvantageMemory(), AdvantageMemory()]

        # Strategy network also uses extended features
        self.strategy_net = SimpleNN(
            EXTENDED_FEATURE_DIM, hidden_dim, NUM_ACTIONS, seed=seed + 2
        )
        self.strategy_memory = AdvantageMemory()
        self.iterations: int = 0

    def build_features(
        self,
        base_features: list[float],
        opponent_stats: PlayerStats | None = None,
    ) -> list[float]:
        """Combine base features with opponent features into extended vector."""
        if opponent_stats is not None:
            opp_feats = extract_opponent_features(opponent_stats)
        else:
            opp_feats = default_opponent_features()
        return base_features + opp_feats

    def get_strategy(
        self,
        player: int,
        base_features: list[float],
        legal_actions: set[ActionType],
        opponent_stats: PlayerStats | None = None,
    ) -> ActionDistribution:
        """Get strategy from opponent-aware advantage network."""
        extended = self.build_features(base_features, opponent_stats)
        raw = self.adv_nets[player].forward(extended)

        positive = {}
        for action in legal_actions:
            idx = ACTION_INDEX.get(action)
            if idx is not None:
                positive[action] = max(0.0, raw[idx])

        total = sum(positive.values())
        if total > 0:
            probs = {a: v / total for a, v in positive.items()}
        else:
            n = len(legal_actions) or 1
            probs = {a: 1.0 / n for a in legal_actions}

        return ActionDistribution(probabilities=probs)

    def add_advantage_sample(
        self,
        player: int,
        base_features: list[float],
        iteration: int,
        advantages: list[float],
        opponent_stats: PlayerStats | None = None,
    ) -> None:
        """Store an advantage sample with extended features."""
        extended = self.build_features(base_features, opponent_stats)
        self.adv_memories[player].add(extended, iteration, advantages)

    def add_strategy_sample(
        self,
        base_features: list[float],
        iteration: int,
        strategy_target: list[float],
        opponent_stats: PlayerStats | None = None,
    ) -> None:
        """Store a strategy sample with extended features."""
        extended = self.build_features(base_features, opponent_stats)
        self.strategy_memory.add(extended, iteration, strategy_target)

    def train_advantage(self, player: int, epochs: int = 2, batch_size: int = 64) -> float:
        """Train the advantage network on buffered samples."""
        memory = self.adv_memories[player]
        if len(memory.buffer) < 10:
            return 0.0

        total_loss = 0.0
        count = 0
        for _ in range(epochs):
            batch = memory.sample_batch(min(batch_size, len(memory.buffer)), self.rng)
            for feat_vec, _iteration, target in batch:
                loss = self.adv_nets[player].train_step(feat_vec, target, lr=0.001)
                total_loss += loss
                count += 1

        return total_loss / max(count, 1)

    def train_strategy(self, epochs: int = 2, batch_size: int = 64) -> float:
        """Train the strategy network on buffered samples."""
        if len(self.strategy_memory.buffer) < 10:
            return 0.0

        total_loss = 0.0
        count = 0
        for _ in range(epochs):
            batch = self.strategy_memory.sample_batch(
                min(batch_size, len(self.strategy_memory.buffer)), self.rng
            )
            for feat_vec, _iteration, target in batch:
                loss = self.strategy_net.train_step(feat_vec, target, lr=0.001)
                total_loss += loss
                count += 1

        return total_loss / max(count, 1)
