"""CFR-MIX — Counterfactual Regret Minimisation for multiplayer coordination.

Extends standard CFR to handle combinatorial / joint action spaces by
factorising joint actions into per-player marginals and using a mixing
network to attribute joint utility back to individual players.

Each player maintains its own regret table and strategy, and the mixing
network learns to decompose joint utility into per-player contributions.

Reference: Inspired by QMIX (Rashid et al., 2018) adapted to CFR setting.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from packages.cfr_agent.deep_cfr import (
    FEATURE_DIM,
    NUM_ACTIONS,
    ACTION_INDEX,
    SimpleNN,
)
from packages.cfr_agent.trainer import CFRState, CFR_ACTIONS
from packages.common.types import ActionType
from packages.strategy.mixed import ActionDistribution


def _softmax(logits: list[float]) -> list[float]:
    max_v = max(logits) if logits else 0.0
    exps = [math.exp(v - max_v) for v in logits]
    total = sum(exps)
    if total == 0:
        return [1.0 / len(logits)] * len(logits)
    return [e / total for e in exps]


class MixingNetwork:
    """Combines per-player utilities into a joint utility estimate.

    Takes per-player Q-values and a global state, and produces a joint
    Q-value that respects monotonicity (higher individual Q => higher joint Q).
    """

    def __init__(self, num_players: int, hidden_dim: int = 32, *, seed: int = 42):
        self.num_players = num_players
        self.rng = random.Random(seed)

        # Hyper-network that produces mixing weights from global state
        # Input: num_players individual utilities
        # Output: 1 joint utility
        self.net = SimpleNN(num_players, hidden_dim, 1, seed=seed)

        # Ensure positive weights for monotonicity (abs transform)
        for i in range(len(self.net.w1)):
            for j in range(len(self.net.w1[i])):
                self.net.w1[i][j] = abs(self.net.w1[i][j])
        for i in range(len(self.net.w2)):
            for j in range(len(self.net.w2[i])):
                self.net.w2[i][j] = abs(self.net.w2[i][j])

    def mix(self, per_player_utilities: list[float]) -> float:
        """Combine per-player utilities into joint utility.

        Uses absolute-value weights to maintain monotonicity:
        increasing any player's utility never decreases the joint utility.
        """
        result = self.net.forward(per_player_utilities)
        return result[0]

    def attribute(self, per_player_utilities: list[float], joint_utility: float) -> list[float]:
        """Attribute joint utility back to individual players.

        Proportional attribution based on each player's contribution
        to the joint utility through the mixing network.
        """
        # Compute per-player marginal contributions via finite differences
        base = self.mix(per_player_utilities)
        attributions = []
        epsilon = 0.01

        for i in range(self.num_players):
            perturbed = list(per_player_utilities)
            perturbed[i] += epsilon
            perturbed_val = self.mix(perturbed)
            marginal = (perturbed_val - base) / epsilon
            attributions.append(marginal)

        # Scale so attributions sum to joint_utility
        total_attr = sum(abs(a) for a in attributions)
        if total_attr > 0:
            attributions = [a / total_attr * joint_utility for a in attributions]
        else:
            attributions = [joint_utility / self.num_players] * self.num_players

        return attributions


class CFRMixTrainer:
    """CFR-MIX: multiplayer CFR with joint action decomposition.

    Each player maintains independent regret and strategy tables.
    A mixing network attributes joint utility to individual players
    for regret updates.
    """

    def __init__(
        self,
        num_players: int = 3,
        *,
        hidden_dim: int = 64,
        seed: int = 42,
    ) -> None:
        self.num_players = num_players
        self.hidden_dim = hidden_dim
        self.rng = random.Random(seed)

        # Per-player CFR state
        self.player_states: dict[int, CFRState] = {
            p: CFRState() for p in range(num_players)
        }

        # Per-player advantage networks
        self.player_networks: dict[int, SimpleNN] = {
            p: SimpleNN(FEATURE_DIM, hidden_dim, NUM_ACTIONS, seed=seed + p)
            for p in range(num_players)
        }

        # Mixing network for joint utility decomposition
        self.mixing_network = MixingNetwork(
            num_players, hidden_dim=32, seed=seed + 100
        )

        self.iterations = 0

    def cfr_mix_update(
        self,
        joint_info_set: str,
        per_player_features: dict[int, list[float]],
        per_player_legal: dict[int, set[ActionType]],
        joint_utility: float,
    ) -> None:
        """Perform a CFR-MIX update step.

        1. Get each player's current strategy.
        2. Compute per-player utilities via mixing network attribution.
        3. Update each player's regrets independently.
        """
        # Get per-player strategies
        strategies: dict[int, ActionDistribution] = {}
        for p in range(self.num_players):
            if p in per_player_features and p in per_player_legal:
                strategies[p] = self._get_player_strategy(
                    p, per_player_features[p], per_player_legal[p]
                )
            else:
                strategies[p] = ActionDistribution(
                    probabilities={ActionType.CHECK: 1.0}
                )

        # Get per-player utility estimates from their networks
        per_player_utils = []
        for p in range(self.num_players):
            if p in per_player_features:
                raw = self.player_networks[p].forward(per_player_features[p])
                # Utility estimate = expected value under current strategy
                ev = 0.0
                for action, prob in strategies[p].probabilities.items():
                    idx = ACTION_INDEX.get(action, 0)
                    ev += prob * raw[idx]
                per_player_utils.append(ev)
            else:
                per_player_utils.append(0.0)

        # Attribute joint utility to individuals
        attributed = self.mixing_network.attribute(per_player_utils, joint_utility)

        # Update each player's regrets
        for p in range(self.num_players):
            if p not in per_player_features or p not in per_player_legal:
                continue

            info_key = f"p{p}_{joint_info_set}"
            player_utility = attributed[p]

            # Compute per-action utilities
            raw = self.player_networks[p].forward(per_player_features[p])
            action_utilities: dict[ActionType, float] = {}
            for action in per_player_legal[p]:
                idx = ACTION_INDEX.get(action, 0)
                action_utilities[action] = raw[idx]

            # Standard CFR regret update
            self.player_states[p].update(
                info_key, strategies[p], action_utilities, player_utility
            )

    def _get_player_strategy(
        self,
        player: int,
        features: list[float],
        legal_actions: set[ActionType],
    ) -> ActionDistribution:
        """Get a player's strategy from their advantage network."""
        raw = self.player_networks[player].forward(features)
        positive: dict[ActionType, float] = {}
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

    def train(
        self,
        iterations: int = 100,
        num_players: int | None = None,
    ) -> dict[int, CFRState]:
        """Train CFR-MIX for the given number of iterations.

        Generates synthetic multiplayer scenarios and runs CFR-MIX updates.

        Returns:
            Per-player CFR states.
        """
        np = num_players or self.num_players

        for i in range(iterations):
            # Generate random features for each player
            per_player_features = {
                p: [self.rng.random() for _ in range(FEATURE_DIM)]
                for p in range(np)
            }

            legal_actions = {ActionType.FOLD, ActionType.CHECK, ActionType.CALL,
                             ActionType.BET, ActionType.RAISE}
            per_player_legal = {p: legal_actions for p in range(np)}

            # Random joint utility (would come from game simulation in practice)
            joint_utility = self.rng.gauss(0, 10)
            info_set = f"iter_{i}_street_{self.rng.randint(0, 3)}"

            self.cfr_mix_update(
                info_set, per_player_features, per_player_legal, joint_utility
            )
            self.iterations += 1

            # Periodically train networks on accumulated regrets
            if (i + 1) % 50 == 0:
                self._train_networks()

        return self.player_states

    def _train_networks(self) -> None:
        """Update player networks from accumulated strategy sums."""
        for p in range(self.num_players):
            state = self.player_states[p]
            for info_key, strat_sum in state.strategy_sum.items():
                # Create a pseudo feature vector from the info key hash
                feat = [
                    (hash(info_key + str(j)) % 1000) / 1000.0
                    for j in range(FEATURE_DIM)
                ]
                # Target: normalized strategy sum
                target = [0.0] * NUM_ACTIONS
                total = sum(max(0, v) for v in strat_sum.values())
                if total > 0:
                    for action_val, weight in strat_sum.items():
                        for action, idx in ACTION_INDEX.items():
                            if action.value == action_val:
                                target[idx] = max(0, weight) / total
                self.player_networks[p].train_step(feat, target, lr=0.001)

    def get_joint_strategy(
        self,
        per_player_features: dict[int, list[float]],
        per_player_legal: dict[int, set[ActionType]] | None = None,
    ) -> dict[int, ActionDistribution]:
        """Get the joint strategy (one distribution per player).

        Args:
            per_player_features: Feature vectors keyed by player ID.
            per_player_legal: Legal actions per player. Defaults to all actions.

        Returns:
            Dict mapping player_id to their ActionDistribution.
        """
        default_legal = set(CFR_ACTIONS)
        result: dict[int, ActionDistribution] = {}

        for p, features in per_player_features.items():
            legal = (per_player_legal or {}).get(p, default_legal)
            result[p] = self._get_player_strategy(p, features, legal)

        return result
