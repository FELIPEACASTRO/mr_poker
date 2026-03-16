"""Neural Population Learning (NeuPL) — Conditional multi-policy network.

Implements the core idea from NeuPL (Liu, Marris, Hennes et al., DeepMind 2022):
a single neural network that generates different strategies conditioned on an
opponent style embedding. Instead of training separate CFR instances per
opponent type, one network produces specialized responses for each archetype.

Architecture:
    Input: game_features(F) ⊕ opponent_embedding(E)
    Hidden: FC(F+E, H) → ReLU → FC(H, H) → ReLU
    Output: FC(H, |A|) → softmax (action distribution)

The opponent embedding comes from StyleEmbedder (12-dim stats → 16-dim latent).

Usage::

    neupl = NeuPLNetwork(feature_dim=10, embedding_dim=16)

    # Train against different opponent styles
    for features, embedding, target_strategy in training_data:
        loss = neupl.train_step(features, embedding, target_strategy)

    # At decision time: condition on current opponent's embedding
    strategy = neupl.get_strategy(features, opponent_embedding, legal_actions)

Reference: arxiv.org/abs/2202.07415
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from packages.common.types import ActionType
from packages.strategy.mixed import ActionDistribution


# Action indexing (same as deep_cfr.py)
_ACTIONS = [
    ActionType.FOLD,
    ActionType.CHECK,
    ActionType.CALL,
    ActionType.BET,
    ActionType.RAISE,
    ActionType.ALL_IN,
]
_ACTION_INDEX = {a: i for i, a in enumerate(_ACTIONS)}
_NUM_ACTIONS = len(_ACTIONS)


def _softmax(logits: list[float]) -> list[float]:
    max_v = max(logits) if logits else 0.0
    exps = [math.exp(v - max_v) for v in logits]
    total = sum(exps)
    if total == 0:
        return [1.0 / len(logits)] * len(logits)
    return [e / total for e in exps]


class NeuPLNetwork:
    """Conditional policy network that adapts to opponent style.

    Single network produces different strategies based on opponent
    embedding, enabling transfer learning across opponent types.

    Args:
        feature_dim: Dimension of game state features.
        embedding_dim: Dimension of opponent style embedding.
        hidden_dim: Hidden layer size.
        seed: Random seed for weight initialization.
    """

    def __init__(
        self,
        feature_dim: int = 10,
        embedding_dim: int = 16,
        hidden_dim: int = 48,
        seed: int = 42,
    ) -> None:
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.input_dim = feature_dim + embedding_dim

        rng = random.Random(seed)

        # Layer 1: (features + embedding) -> hidden
        scale1 = math.sqrt(2.0 / self.input_dim)
        self.w1 = [
            [rng.gauss(0, scale1) for _ in range(self.input_dim)]
            for _ in range(hidden_dim)
        ]
        self.b1 = [0.0] * hidden_dim

        # Layer 2: hidden -> hidden
        scale2 = math.sqrt(2.0 / hidden_dim)
        self.w2 = [
            [rng.gauss(0, scale2) for _ in range(hidden_dim)]
            for _ in range(hidden_dim)
        ]
        self.b2 = [0.0] * hidden_dim

        # Output layer: hidden -> actions
        scale3 = math.sqrt(2.0 / hidden_dim)
        self.w3 = [
            [rng.gauss(0, scale3) for _ in range(hidden_dim)]
            for _ in range(_NUM_ACTIONS)
        ]
        self.b3 = [0.0] * _NUM_ACTIONS

        self.train_steps = 0

    def forward(
        self,
        features: list[float],
        embedding: list[float],
    ) -> list[float]:
        """Forward pass: returns action logits.

        Args:
            features: Game state feature vector (length feature_dim).
            embedding: Opponent style embedding (length embedding_dim).

        Returns:
            List of logits per action.
        """
        # Concatenate features and embedding
        x = list(features) + list(embedding)

        # Pad/truncate to input_dim
        if len(x) < self.input_dim:
            x.extend([0.0] * (self.input_dim - len(x)))
        x = x[: self.input_dim]

        # Layer 1: ReLU
        h1 = []
        for i in range(self.hidden_dim):
            val = self.b1[i] + sum(self.w1[i][j] * x[j] for j in range(self.input_dim))
            h1.append(max(0.0, val))

        # Layer 2: ReLU
        h2 = []
        for i in range(self.hidden_dim):
            val = self.b2[i] + sum(self.w2[i][j] * h1[j] for j in range(self.hidden_dim))
            h2.append(max(0.0, val))

        # Output: linear logits
        logits = []
        for i in range(_NUM_ACTIONS):
            val = self.b3[i] + sum(self.w3[i][j] * h2[j] for j in range(self.hidden_dim))
            logits.append(val)

        return logits

    def get_strategy(
        self,
        features: list[float],
        embedding: list[float],
        legal: set[ActionType],
    ) -> ActionDistribution:
        """Get conditional strategy for the given opponent embedding.

        Args:
            features: Game state features.
            embedding: Opponent style embedding from StyleEmbedder.
            legal: Set of legal actions.

        Returns:
            ActionDistribution conditioned on the opponent type.
        """
        logits = self.forward(features, embedding)

        # Mask illegal actions
        for i, action in enumerate(_ACTIONS):
            if action not in legal:
                logits[i] = -1e9

        probs = _softmax(logits)

        # Build distribution with only legal actions
        distribution: dict[ActionType, float] = {}
        for i, action in enumerate(_ACTIONS):
            if action in legal:
                distribution[action] = probs[i]

        # Normalize
        total = sum(distribution.values())
        if total > 0:
            distribution = {a: p / total for a, p in distribution.items()}
        else:
            n = len(legal)
            distribution = {a: 1.0 / n for a in legal}

        return ActionDistribution(probabilities=distribution)

    def train_step(
        self,
        features: list[float],
        embedding: list[float],
        target_probs: dict[ActionType, float],
        lr: float = 0.001,
    ) -> float:
        """Train one step via cross-entropy loss against target strategy.

        Args:
            features: Game state features.
            embedding: Opponent style embedding.
            target_probs: Target action distribution (from CFR or solver).
            lr: Learning rate.

        Returns:
            Cross-entropy loss value.
        """
        # Forward
        x = list(features) + list(embedding)
        if len(x) < self.input_dim:
            x.extend([0.0] * (self.input_dim - len(x)))
        x = x[: self.input_dim]

        # Layer 1
        h1_raw = []
        h1 = []
        for i in range(self.hidden_dim):
            val = self.b1[i] + sum(self.w1[i][j] * x[j] for j in range(self.input_dim))
            h1_raw.append(val)
            h1.append(max(0.0, val))

        # Layer 2
        h2_raw = []
        h2 = []
        for i in range(self.hidden_dim):
            val = self.b2[i] + sum(self.w2[i][j] * h1[j] for j in range(self.hidden_dim))
            h2_raw.append(val)
            h2.append(max(0.0, val))

        # Output logits + softmax
        logits = []
        for i in range(_NUM_ACTIONS):
            val = self.b3[i] + sum(self.w3[i][j] * h2[j] for j in range(self.hidden_dim))
            logits.append(val)

        probs = _softmax(logits)

        # Target vector
        target = [target_probs.get(a, 0.0) for a in _ACTIONS]

        # Cross-entropy loss
        loss = -sum(
            target[i] * math.log(max(probs[i], 1e-10))
            for i in range(_NUM_ACTIONS)
            if target[i] > 0
        )

        # Backward: d_loss/d_logits = probs - target (softmax + CE gradient)
        d_logits = [probs[i] - target[i] for i in range(_NUM_ACTIONS)]

        # Backward: output layer
        d_h2 = [0.0] * self.hidden_dim
        for i in range(_NUM_ACTIONS):
            for j in range(self.hidden_dim):
                d_h2[j] += d_logits[i] * self.w3[i][j]
                self.w3[i][j] -= lr * d_logits[i] * h2[j]
            self.b3[i] -= lr * d_logits[i]

        # Backward: layer 2
        d_h1 = [0.0] * self.hidden_dim
        for i in range(self.hidden_dim):
            if h2_raw[i] <= 0:
                continue
            grad = d_h2[i]
            for j in range(self.hidden_dim):
                d_h1[j] += grad * self.w2[i][j]
                self.w2[i][j] -= lr * grad * h1[j]
            self.b2[i] -= lr * grad

        # Backward: layer 1
        for i in range(self.hidden_dim):
            if h1_raw[i] <= 0:
                continue
            grad = d_h1[i]
            for j in range(self.input_dim):
                self.w1[i][j] -= lr * grad * x[j]
            self.b1[i] -= lr * grad

        self.train_steps += 1
        return loss


@dataclass
class PopulationPolicy:
    """A named policy in the population with its style embedding."""

    name: str
    embedding: list[float]
    description: str = ""


class NeuPLPopulation:
    """Manages a population of conditional policies via a single NeuPL network.

    Maintains a set of named opponent archetypes, each with a style embedding.
    The NeuPL network produces specialized counter-strategies for each.

    Usage::

        pop = NeuPLPopulation(feature_dim=10)
        pop.add_archetype("nit", nit_embedding)
        pop.add_archetype("lag", lag_embedding)

        # At decision time:
        strategy = pop.get_counter_strategy(features, opponent_embedding, legal)
    """

    def __init__(
        self,
        feature_dim: int = 10,
        embedding_dim: int = 16,
        hidden_dim: int = 48,
        seed: int = 42,
    ) -> None:
        self.network = NeuPLNetwork(
            feature_dim=feature_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            seed=seed,
        )
        self.archetypes: dict[str, PopulationPolicy] = {}

    def add_archetype(
        self,
        name: str,
        embedding: list[float],
        description: str = "",
    ) -> None:
        """Register an opponent archetype with its style embedding."""
        self.archetypes[name] = PopulationPolicy(
            name=name, embedding=embedding, description=description
        )

    def get_counter_strategy(
        self,
        features: list[float],
        opponent_embedding: list[float],
        legal: set[ActionType],
    ) -> ActionDistribution:
        """Get counter-strategy for the given opponent embedding."""
        return self.network.get_strategy(features, opponent_embedding, legal)

    def get_archetype_strategy(
        self,
        features: list[float],
        archetype_name: str,
        legal: set[ActionType],
    ) -> ActionDistribution:
        """Get strategy optimized against a named archetype."""
        if archetype_name not in self.archetypes:
            # Unknown archetype: use zero embedding (baseline)
            emb = [0.0] * self.network.embedding_dim
        else:
            emb = self.archetypes[archetype_name].embedding
        return self.network.get_strategy(features, emb, legal)

    def train_against_archetype(
        self,
        features: list[float],
        archetype_name: str,
        target_probs: dict[ActionType, float],
        lr: float = 0.001,
    ) -> float:
        """Train the network against a specific archetype."""
        if archetype_name not in self.archetypes:
            return 0.0
        emb = self.archetypes[archetype_name].embedding
        return self.network.train_step(features, emb, target_probs, lr)

    @property
    def archetype_names(self) -> list[str]:
        return list(self.archetypes.keys())
