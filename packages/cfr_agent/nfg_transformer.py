"""NfgTransformer — Equivariant attention network for equilibrium solving.

Inspired by Liu, Marris et al. (DeepMind 2024): a transformer-like
architecture that is equivariant to permutations of actions, making it
naturally suited for normal-form game solving and Deep CFR value prediction.

Key properties:
1. Action-equivariance: permuting actions in input permutes output accordingly
2. Self-attention over actions: each action "attends" to all others
3. Game-aware: processes (feature, action_embedding) pairs jointly

Architecture:
    Input: N actions × (game_features + action_embedding)
    Self-attention: Q,K,V projections with scaled dot-product attention
    Output head: per-action value/regret prediction

This replaces SimpleNN in Deep CFR for better generalization to unseen games.

Reference: arxiv.org/abs/2402.08393
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from packages.common.types import ActionType
from packages.strategy.mixed import ActionDistribution


# Standard action set
_ACTIONS = [
    ActionType.FOLD, ActionType.CHECK, ActionType.CALL,
    ActionType.BET, ActionType.RAISE, ActionType.ALL_IN,
]
_ACTION_INDEX = {a: i for i, a in enumerate(_ACTIONS)}
_NUM_ACTIONS = len(_ACTIONS)

# Per-action one-hot embedding dimension
_ACTION_EMB_DIM = _NUM_ACTIONS


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_v = max(values)
    exps = [math.exp(v - max_v) for v in values]
    total = sum(exps)
    if total == 0:
        return [1.0 / len(values)] * len(values)
    return [e / total for e in exps]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class NfgTransformerBlock:
    """Single self-attention block over actions.

    Each action has a representation vector. The block computes
    scaled dot-product attention across all actions, then applies
    a feedforward layer.

    Args:
        dim: Dimension of each action's representation.
        num_heads: Number of attention heads (must divide dim).
        seed: Random seed.
    """

    def __init__(self, dim: int = 32, num_heads: int = 4, seed: int = 42) -> None:
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        rng = random.Random(seed)
        scale = math.sqrt(2.0 / dim)

        # Q, K, V projections (dim → dim)
        self.wq = [[rng.gauss(0, scale) for _ in range(dim)] for _ in range(dim)]
        self.wk = [[rng.gauss(0, scale) for _ in range(dim)] for _ in range(dim)]
        self.wv = [[rng.gauss(0, scale) for _ in range(dim)] for _ in range(dim)]

        # Output projection
        self.wo = [[rng.gauss(0, scale) for _ in range(dim)] for _ in range(dim)]

        # Feedforward: dim → dim*2 → dim
        ff_scale = math.sqrt(2.0 / dim)
        self.ff_w1 = [[rng.gauss(0, ff_scale) for _ in range(dim)] for _ in range(dim * 2)]
        self.ff_b1 = [0.0] * (dim * 2)
        self.ff_w2 = [[rng.gauss(0, math.sqrt(2.0 / (dim * 2)))] * (dim * 2) for _ in range(dim)]
        self.ff_b2 = [0.0] * dim

    def _matmul(self, matrix: list[list[float]], vec: list[float]) -> list[float]:
        return [sum(matrix[i][j] * vec[j] for j in range(len(vec))) for i in range(len(matrix))]

    def forward(self, action_reps: list[list[float]]) -> list[list[float]]:
        """Apply self-attention over action representations.

        Args:
            action_reps: List of N action representation vectors (each dim-length).

        Returns:
            Updated action representations (same shape).
        """
        n = len(action_reps)
        if n == 0:
            return []

        # Compute Q, K, V for all actions
        queries = [self._matmul(self.wq, r) for r in action_reps]
        keys = [self._matmul(self.wk, r) for r in action_reps]
        values = [self._matmul(self.wv, r) for r in action_reps]

        scale = math.sqrt(self.head_dim)

        # Multi-head attention (simplified: single effective head computation)
        attended = []
        for i in range(n):
            # Attention scores
            scores = [_dot(queries[i], keys[j]) / scale for j in range(n)]
            weights = _softmax(scores)

            # Weighted sum of values
            out = [0.0] * self.dim
            for j in range(n):
                for d in range(self.dim):
                    out[d] += weights[j] * values[j][d]

            attended.append(self._matmul(self.wo, out))

        # Residual + feedforward
        output = []
        for i in range(n):
            # Residual connection
            residual = [action_reps[i][d] + attended[i][d] for d in range(self.dim)]

            # Feedforward: ReLU(W1*x + b1) then W2 + b2
            ff_hidden = self._matmul(self.ff_w1, residual)
            ff_hidden = [max(0.0, ff_hidden[d] + self.ff_b1[d]) for d in range(self.dim * 2)]
            ff_out = self._matmul(self.ff_w2, ff_hidden)
            ff_out = [ff_out[d] + self.ff_b2[d] for d in range(self.dim)]

            # Second residual
            final = [residual[d] + ff_out[d] for d in range(self.dim)]
            output.append(final)

        return output


class NfgTransformer:
    """Equivariant transformer for game value/strategy prediction.

    Processes game features + per-action embeddings through self-attention
    blocks, then produces per-action value or strategy predictions.

    Args:
        feature_dim: Dimension of game state features.
        hidden_dim: Internal representation dimension.
        num_blocks: Number of transformer blocks.
        num_heads: Attention heads per block.
        seed: Random seed.
    """

    def __init__(
        self,
        feature_dim: int = 15,
        hidden_dim: int = 32,
        num_blocks: int = 2,
        num_heads: int = 4,
        seed: int = 42,
    ) -> None:
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        rng = random.Random(seed)

        # Input projection: (features + action_emb) → hidden_dim
        input_dim = feature_dim + _ACTION_EMB_DIM
        scale = math.sqrt(2.0 / input_dim)
        self.input_proj = [
            [rng.gauss(0, scale) for _ in range(input_dim)]
            for _ in range(hidden_dim)
        ]
        self.input_bias = [0.0] * hidden_dim

        # Transformer blocks
        self.blocks = [
            NfgTransformerBlock(dim=hidden_dim, num_heads=num_heads, seed=seed + i)
            for i in range(num_blocks)
        ]

        # Output head: hidden_dim → 1 (per-action value)
        out_scale = math.sqrt(2.0 / hidden_dim)
        self.output_w = [rng.gauss(0, out_scale) for _ in range(hidden_dim)]
        self.output_b = 0.0

    def predict_values(
        self,
        features: list[float],
        legal: set[ActionType] | None = None,
    ) -> dict[ActionType, float]:
        """Predict value for each action.

        Args:
            features: Game state feature vector.
            legal: Legal actions (all if None).

        Returns:
            Dict mapping action to predicted value.
        """
        actions = list(legal) if legal else list(_ACTIONS)

        # Build per-action input: features ⊕ action_one_hot
        action_inputs = []
        for action in actions:
            one_hot = [0.0] * _ACTION_EMB_DIM
            idx = _ACTION_INDEX.get(action, 0)
            one_hot[idx] = 1.0
            inp = list(features) + one_hot
            # Pad to expected input size
            while len(inp) < self.feature_dim + _ACTION_EMB_DIM:
                inp.append(0.0)
            inp = inp[: self.feature_dim + _ACTION_EMB_DIM]
            action_inputs.append(inp)

        # Project to hidden dim
        reps = []
        for inp in action_inputs:
            proj = [
                max(0.0, self.input_bias[i] + sum(
                    self.input_proj[i][j] * inp[j]
                    for j in range(len(inp))
                ))
                for i in range(self.hidden_dim)
            ]
            reps.append(proj)

        # Apply transformer blocks
        for block in self.blocks:
            reps = block.forward(reps)

        # Output head: per-action scalar value
        values = {}
        for i, action in enumerate(actions):
            val = self.output_b + sum(
                self.output_w[d] * reps[i][d] for d in range(self.hidden_dim)
            )
            values[action] = val

        return values

    def predict_strategy(
        self,
        features: list[float],
        legal: set[ActionType],
    ) -> ActionDistribution:
        """Predict strategy (softmax over values) for legal actions."""
        values = self.predict_values(features, legal)

        logits = [values.get(a, -1e9) for a in sorted(legal, key=lambda a: a.value)]
        probs = _softmax(logits)

        actions_sorted = sorted(legal, key=lambda a: a.value)
        distribution = {a: p for a, p in zip(actions_sorted, probs)}

        return ActionDistribution(probabilities=distribution)

    @property
    def num_parameters(self) -> int:
        """Approximate parameter count."""
        input_params = self.hidden_dim * (self.feature_dim + _ACTION_EMB_DIM) + self.hidden_dim
        block_params = len(self.blocks) * (4 * self.hidden_dim * self.hidden_dim + 3 * self.hidden_dim * self.hidden_dim)
        output_params = self.hidden_dim + 1
        return input_params + block_params + output_params
