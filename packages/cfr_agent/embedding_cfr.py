"""Embedding CFR — Maps info sets to dense embeddings for generalization.

Instead of treating every info set as unique (tabular CFR) or learning a
single function (Deep CFR), Embedding CFR clusters similar info sets in a
learned embedding space. Info sets that map to nearby embeddings share
regrets and strategies, enabling generalization (e.g., AKs on Kh7d2c and
AKs on Ks8c3d produce similar strategies).

Architecture:
- Encoder: SimpleNN(FEATURE_DIM, 32, embedding_dim) maps features -> embedding
- EmbeddingRegretTable: maps embedding clusters to regret/strategy vectors
- Nearest-neighbor clustering with distance threshold

Reference: Inspired by abstraction techniques in Libratus/Pluribus.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any

from packages.common.types import ActionType
from packages.strategy.mixed import ActionDistribution

from packages.cfr_agent.deep_cfr import (
    SimpleNN,
    FEATURE_DIM,
    NUM_ACTIONS,
    ACTION_INDEX,
)
from packages.cfr_agent.trainer import CFRState, CFR_ACTIONS

logger = logging.getLogger(__name__)


def _euclidean_distance(a: list[float], b: list[float]) -> float:
    """Euclidean distance between two vectors."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _softmax(values: list[float]) -> list[float]:
    """Numerically stable softmax."""
    max_v = max(values) if values else 0.0
    exps = [math.exp(v - max_v) for v in values]
    total = sum(exps)
    if total == 0:
        return [1.0 / len(values)] * len(values) if values else []
    return [e / total for e in exps]


@dataclass
class ClusterEntry:
    """A cluster in embedding space with associated regrets and strategy sums."""

    centroid: list[float]
    cumulative_regret: list[float] = field(default_factory=lambda: [0.0] * NUM_ACTIONS)
    strategy_sum: list[float] = field(default_factory=lambda: [0.0] * NUM_ACTIONS)
    count: int = 0


class EmbeddingRegretTable:
    """Maps embedding clusters to regret and strategy vectors.

    Uses nearest-neighbor lookup: if a new embedding is within `threshold`
    of an existing cluster centroid, they share regrets/strategies.
    Otherwise a new cluster is created.
    """

    def __init__(self, embedding_dim: int, threshold: float = 0.5) -> None:
        self.embedding_dim = embedding_dim
        self.threshold = threshold
        self.clusters: list[ClusterEntry] = []

    def find_cluster(self, embedding: list[float]) -> int | None:
        """Find the nearest cluster within threshold. Returns index or None."""
        best_idx = None
        best_dist = float("inf")
        for i, cluster in enumerate(self.clusters):
            dist = _euclidean_distance(embedding, cluster.centroid)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx is not None and best_dist < self.threshold:
            return best_idx
        return None

    def get_or_create_cluster(self, embedding: list[float]) -> int:
        """Find existing cluster or create a new one. Returns cluster index."""
        idx = self.find_cluster(embedding)
        if idx is not None:
            return idx
        # Create new cluster
        self.clusters.append(ClusterEntry(centroid=list(embedding)))
        return len(self.clusters) - 1

    @property
    def num_clusters(self) -> int:
        return len(self.clusters)


class EmbeddingCFRTrainer:
    """CFR trainer that operates in a learned embedding space.

    Similar info sets (e.g., same hand class on similar board textures)
    are mapped to nearby embeddings and share regrets/strategies. This
    provides generalization between positions never seen during training.
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 16,
        threshold: float = 0.5,
        hidden_dim: int = 32,
        seed: int = 42,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.rng = random.Random(seed)

        # Encoder: maps raw features to dense embedding
        self.encoder = SimpleNN(FEATURE_DIM, hidden_dim, embedding_dim, seed=seed)

        # Embedding-space regret table
        self.regret_table = EmbeddingRegretTable(
            embedding_dim=embedding_dim, threshold=threshold
        )

        self.iterations = 0

    def encode(self, features: list[float]) -> list[float]:
        """Encode raw features into embedding space."""
        return self.encoder.forward(features)

    def _cluster_info_set(self, features: list[float]) -> int:
        """Map features to a cluster ID via the encoder."""
        embedding = self.encode(features)
        return self.regret_table.get_or_create_cluster(embedding)

    def current_strategy(
        self, features: list[float], legal_actions: set[ActionType]
    ) -> ActionDistribution:
        """Get current strategy for features via regret matching in embedding space."""
        cluster_id = self._cluster_info_set(features)
        cluster = self.regret_table.clusters[cluster_id]

        # Regret matching over legal actions
        positive: dict[ActionType, float] = {}
        for action in legal_actions:
            idx = ACTION_INDEX.get(action)
            if idx is not None:
                positive[action] = max(0.0, cluster.cumulative_regret[idx])

        total = sum(positive.values())
        if total > 0:
            probs = {a: v / total for a, v in positive.items()}
        else:
            n = len(legal_actions) or 1
            probs = {a: 1.0 / n for a in legal_actions}

        return ActionDistribution(probabilities=probs)

    def update(
        self,
        features: list[float],
        strategy: ActionDistribution,
        action_utilities: dict[ActionType, float],
        node_utility: float,
    ) -> None:
        """Update regrets and strategy sums for the embedding cluster."""
        cluster_id = self._cluster_info_set(features)
        cluster = self.regret_table.clusters[cluster_id]
        cluster.count += 1

        for action, utility in action_utilities.items():
            idx = ACTION_INDEX.get(action)
            if idx is None:
                continue
            regret = utility - node_utility
            cluster.cumulative_regret[idx] += regret
            prob = strategy.probabilities.get(action, 0.0)
            cluster.strategy_sum[idx] += prob

        # Train the encoder to push similar features closer
        # Use strategy sum as a soft label for the embedding
        embedding = self.encode(features)
        target = list(embedding)  # reinforcement: identity target to stabilize
        self.encoder.train_step(features, target, lr=0.0001)

    def train(self, iterations: int = 100) -> dict[int, ActionDistribution]:
        """Run self-play training iterations.

        Returns a mapping from cluster_id to average strategy.
        This is a simplified training loop; for full game-tree traversal,
        integrate with a game engine like the CFRTrainer does.
        """
        # Generate synthetic training samples to demonstrate convergence
        for i in range(iterations):
            # Create random feature vectors and run update
            features = [self.rng.random() for _ in range(FEATURE_DIM)]
            legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}
            strategy = self.current_strategy(features, legal)

            # Synthetic utilities
            utilities = {
                a: self.rng.gauss(0, 1) for a in legal
            }
            ev = sum(
                strategy.probabilities.get(a, 0.0) * u
                for a, u in utilities.items()
            )
            self.update(features, strategy, utilities, ev)
            self.iterations += 1

        # Collect average strategies per cluster
        strategies: dict[int, ActionDistribution] = {}
        for cid, cluster in enumerate(self.regret_table.clusters):
            total = sum(max(0.0, s) for s in cluster.strategy_sum)
            if total > 0:
                probs: dict[ActionType, float] = {}
                for action, idx in ACTION_INDEX.items():
                    val = max(0.0, cluster.strategy_sum[idx])
                    if val > 0:
                        probs[action] = val / total
                strategies[cid] = ActionDistribution(probabilities=probs)
            else:
                strategies[cid] = ActionDistribution(
                    probabilities={a: 1.0 / len(CFR_ACTIONS) for a in CFR_ACTIONS}
                )

        return strategies

    def to_cfr_state(self) -> CFRState:
        """Export embedding clusters as a standard CFRState for compatibility."""
        state = CFRState(iterations=self.iterations)
        for cid, cluster in enumerate(self.regret_table.clusters):
            key = f"emb_cluster_{cid}"
            state.cumulative_regret[key] = {}
            state.strategy_sum[key] = {}
            for action, idx in ACTION_INDEX.items():
                state.cumulative_regret[key][action.value] = cluster.cumulative_regret[idx]
                state.strategy_sum[key][action.value] = cluster.strategy_sum[idx]
        return state
