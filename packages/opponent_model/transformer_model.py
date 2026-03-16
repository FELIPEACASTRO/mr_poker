"""Transformer-based Opponent Model — Pure Python attention for action sequences.

Uses a simplified single-head self-attention mechanism to process sequences
of opponent actions and predict their archetype (playing style). No external
deep learning dependencies; reuses SimpleNN from deep_cfr.

Architecture:
    encode(8) -> embed(32) -> self_attention(32) -> FFN(32->16) -> output(16->8 archetypes)

Reference: Vaswani et al. (2017) "Attention Is All You Need" (simplified)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from packages.cfr_agent.deep_cfr import SimpleNN
from packages.common.types import ActionType
from packages.opponent_model.classifier import ARCHETYPES

# Action encoding constants
ACTION_ENCODE = {
    ActionType.FOLD: 0,
    ActionType.CHECK: 1,
    ActionType.CALL: 2,
    ActionType.BET: 3,
    ActionType.RAISE: 4,
    ActionType.ALL_IN: 5,
}
NUM_ACTION_TYPES = 6

# Street encoding
STREET_ENCODE = {"pre_flop": 0.0, "flop": 0.33, "turn": 0.67, "river": 1.0}

# Input feature dim per action step
ACTION_FEATURE_DIM = 8

# Sequence length (pad/truncate to this)
MAX_SEQ_LEN = 20

# Model dimensions
EMBED_DIM = 32
FFN_DIM = 16
NUM_ARCHETYPES = len(ARCHETYPES)  # 8

ARCHETYPE_NAMES = list(ARCHETYPES.keys())


def encode_action(
    action_type: ActionType,
    bet_size_ratio: float = 0.0,
    street: str = "pre_flop",
    position: float = 0.0,
) -> list[float]:
    """Encode a single opponent action as an 8-dim feature vector.

    Features:
        [0-5] one-hot action type (6 dims)
        [6]   bet_size_ratio (0-1 capped)
        [7]   street (0-1) * 0.5 + position * 0.5
    """
    vec = [0.0] * ACTION_FEATURE_DIM
    idx = ACTION_ENCODE.get(action_type, 0)
    vec[idx] = 1.0
    vec[6] = min(max(bet_size_ratio, 0.0), 1.0)
    vec[7] = STREET_ENCODE.get(street, 0.0) * 0.5 + min(max(position, 0.0), 1.0) * 0.5
    return vec


def _softmax(values: list[float]) -> list[float]:
    """Numerically stable softmax."""
    m = max(values) if values else 0.0
    exps = [math.exp(v - m) for v in values]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _vec_add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def _vec_scale(a: list[float], s: float) -> list[float]:
    return [x * s for x in a]


class SelfAttention:
    """Single-head scaled dot-product self-attention (pure Python).

    Projects inputs into Q, K, V via SimpleNN layers and computes:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
    """

    def __init__(self, dim: int, *, seed: int = 42) -> None:
        self.dim = dim
        # Q, K, V projections are linear (hidden_dim=dim to keep it linear-ish)
        self.q_proj = SimpleNN(dim, dim, dim, seed=seed)
        self.k_proj = SimpleNN(dim, dim, dim, seed=seed + 1)
        self.v_proj = SimpleNN(dim, dim, dim, seed=seed + 2)
        self.scale = 1.0 / math.sqrt(dim)

    def forward(self, sequence: list[list[float]]) -> list[list[float]]:
        """Apply self-attention over a sequence of vectors.

        Args:
            sequence: list of T vectors, each of length ``dim``.

        Returns:
            list of T attended vectors, each of length ``dim``.
        """
        T = len(sequence)
        if T == 0:
            return []

        # Project to Q, K, V
        Q = [self.q_proj.forward(x) for x in sequence]
        K = [self.k_proj.forward(x) for x in sequence]
        V = [self.v_proj.forward(x) for x in sequence]

        output: list[list[float]] = []
        for i in range(T):
            # Compute attention scores for position i
            scores = [_dot(Q[i], K[j]) * self.scale for j in range(T)]
            weights = _softmax(scores)

            # Weighted sum of V
            attended = [0.0] * self.dim
            for j in range(T):
                attended = _vec_add(attended, _vec_scale(V[j], weights[j]))
            output.append(attended)

        return output


class TransformerOpponentModel:
    """Predict opponent archetype from a sequence of their recent actions.

    Pipeline:
        1. Encode each action as 8-dim vector
        2. Embed to 32-dim via SimpleNN
        3. Self-attention over the sequence
        4. Mean-pool the attended sequence
        5. FFN: 32 -> 16
        6. Output: 16 -> 8 archetype logits -> softmax
    """

    def __init__(self, *, seed: int = 42) -> None:
        self.seed = seed
        self.embed = SimpleNN(ACTION_FEATURE_DIM, EMBED_DIM, EMBED_DIM, seed=seed)
        self.attention = SelfAttention(EMBED_DIM, seed=seed + 10)
        self.ffn = SimpleNN(EMBED_DIM, FFN_DIM, FFN_DIM, seed=seed + 20)
        self.output_layer = SimpleNN(FFN_DIM, FFN_DIM, NUM_ARCHETYPES, seed=seed + 30)

    def _prepare_sequence(
        self, action_sequence: list[list[float]]
    ) -> list[list[float]]:
        """Pad or truncate to MAX_SEQ_LEN."""
        seq = list(action_sequence)
        if len(seq) > MAX_SEQ_LEN:
            seq = seq[-MAX_SEQ_LEN:]
        while len(seq) < MAX_SEQ_LEN:
            seq.append([0.0] * ACTION_FEATURE_DIM)
        return seq

    def forward(self, action_sequence: list[list[float]]) -> list[float]:
        """Full forward pass returning archetype logits (length NUM_ARCHETYPES)."""
        seq = self._prepare_sequence(action_sequence)

        # Embed
        embedded = [self.embed.forward(x) for x in seq]

        # Self-attention
        attended = self.attention.forward(embedded)

        # Mean pool
        pooled = [0.0] * EMBED_DIM
        for vec in attended:
            pooled = _vec_add(pooled, vec)
        pooled = _vec_scale(pooled, 1.0 / len(attended))

        # FFN + output
        hidden = self.ffn.forward(pooled)
        # ReLU on hidden
        hidden = [max(0.0, x) for x in hidden]
        logits = self.output_layer.forward(hidden)
        return logits

    def predict(self, action_sequence: list[list[float]]) -> dict[str, float]:
        """Predict archetype probabilities from a sequence of encoded actions.

        Args:
            action_sequence: list of 8-dim action feature vectors.

        Returns:
            dict mapping archetype name to probability.
        """
        logits = self.forward(action_sequence)
        probs = _softmax(logits)
        return {name: probs[i] for i, name in enumerate(ARCHETYPE_NAMES)}

    def train_on_batch(
        self,
        sequences: list[list[list[float]]],
        labels: list[int],
        lr: float = 0.001,
    ) -> float:
        """Train on a batch of (sequence, label) pairs with cross-entropy loss.

        Args:
            sequences: list of action sequences (each a list of 8-dim vecs).
            labels: list of archetype indices (0..NUM_ARCHETYPES-1).
            lr: learning rate.

        Returns:
            average cross-entropy loss over the batch.
        """
        total_loss = 0.0
        for seq, label in zip(sequences, labels):
            logits = self.forward(seq)
            probs = _softmax(logits)
            # Cross-entropy loss
            p = max(probs[label], 1e-10)
            loss = -math.log(p)
            total_loss += loss

            # Target: one-hot
            target = [0.0] * NUM_ARCHETYPES
            target[label] = 1.0

            # Backprop through output layer (simplified: train output_layer on
            # the FFN hidden representation)
            seq_prepared = self._prepare_sequence(seq)
            embedded = [self.embed.forward(x) for x in seq_prepared]
            attended = self.attention.forward(embedded)
            pooled = [0.0] * EMBED_DIM
            for vec in attended:
                pooled = _vec_add(pooled, vec)
            pooled = _vec_scale(pooled, 1.0 / len(attended))
            hidden = self.ffn.forward(pooled)
            hidden = [max(0.0, x) for x in hidden]

            # Train output layer with CE gradient approximation via MSE on softmax target
            self.output_layer.train_step(hidden, target, lr=lr)

            # Also nudge the FFN
            self.ffn.train_step(pooled, hidden, lr=lr * 0.1)

        return total_loss / max(len(sequences), 1)


class CurriculumScheduler:
    """Schedule training examples from easy (extreme archetypes) to hard.

    Stages:
        0: Only extreme archetypes (nit, maniac, whale, rock)
        1: Add moderate archetypes (tag, lag, fish)
        2: All archetypes including unknown
    """

    EASY_ARCHETYPES = {"nit", "maniac", "whale", "rock"}
    MEDIUM_ARCHETYPES = {"tag", "lag", "fish"}
    HARD_ARCHETYPES = {"unknown"}

    def __init__(self, total_epochs: int = 100) -> None:
        self.total_epochs = total_epochs
        self.current_epoch = 0

    @property
    def stage(self) -> int:
        """Current curriculum stage (0, 1, or 2)."""
        progress = self.current_epoch / max(self.total_epochs, 1)
        if progress < 0.33:
            return 0
        if progress < 0.67:
            return 1
        return 2

    def allowed_archetypes(self) -> set[str]:
        """Return archetypes allowed at current curriculum stage."""
        s = self.stage
        allowed = set(self.EASY_ARCHETYPES)
        if s >= 1:
            allowed |= self.MEDIUM_ARCHETYPES
        if s >= 2:
            allowed |= self.HARD_ARCHETYPES
        return allowed

    def filter_batch(
        self,
        sequences: list[list[list[float]]],
        labels: list[int],
    ) -> tuple[list[list[list[float]]], list[int]]:
        """Filter a batch to only include allowed archetypes."""
        allowed_indices = {
            i for i, name in enumerate(ARCHETYPE_NAMES)
            if name in self.allowed_archetypes()
        }
        filtered_seqs: list[list[list[float]]] = []
        filtered_labels: list[int] = []
        for seq, label in zip(sequences, labels):
            if label in allowed_indices:
                filtered_seqs.append(seq)
                filtered_labels.append(label)
        return filtered_seqs, filtered_labels

    def advance(self) -> None:
        """Advance to next epoch."""
        self.current_epoch += 1
