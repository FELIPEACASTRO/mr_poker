"""CASPER — Case-Based Reasoning for Poker Decisions.

Instead of computing an equilibrium, CASPER stores past game situations
(cases) and retrieves the most similar ones to inform the current decision.
Each case records the context (hand features, board, opponent profile,
pot odds) and the outcome of the action taken.

The case base is searched using weighted k-NN with a domain-specific
similarity metric.  Retrieved cases vote on the best action, weighted
by recency, similarity, and outcome quality.

Reference: Watson & Rubin (2013) "Case-based Reasoning for Poker:
A CASPER approach".
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from packages.common.types import ActionType
from packages.strategy.mixed import ActionDistribution


@dataclass
class PokerCase:
    """A stored poker decision case.

    Attributes:
        features: numeric feature vector describing the situation
        action_taken: the action that was taken
        reward: the outcome (profit/loss in BB)
        opponent_type: archetype of the opponent
        street: game street (0=preflop, 1=flop, 2=turn, 3=river)
        timestamp: case creation order (for recency weighting)
    """
    features: list[float] = field(default_factory=list)
    action_taken: ActionType = ActionType.CHECK
    reward: float = 0.0
    opponent_type: str = "unknown"
    street: int = 0
    timestamp: int = 0


@dataclass
class CaseQuery:
    """Query for retrieving similar cases."""
    features: list[float] = field(default_factory=list)
    opponent_type: str = "unknown"
    street: int = 0


def _euclidean_distance(a: list[float], b: list[float]) -> float:
    """Euclidean distance between two feature vectors."""
    if len(a) != len(b):
        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two feature vectors."""
    if len(a) != len(b):
        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class CaseBase:
    """Storage and retrieval system for poker cases.

    Supports insertion, k-NN retrieval with domain-specific similarity,
    and case pruning to maintain a bounded size.
    """

    def __init__(self, max_cases: int = 10000, seed: int = 42) -> None:
        self.cases: list[PokerCase] = []
        self.max_cases = max_cases
        self._counter = 0
        self.rng = random.Random(seed)

    def add_case(self, case: PokerCase) -> None:
        """Add a case to the case base."""
        case.timestamp = self._counter
        self._counter += 1
        self.cases.append(case)
        if len(self.cases) > self.max_cases:
            self._prune()

    def _prune(self) -> None:
        """Remove lowest-quality cases when base exceeds capacity."""
        # Score = recency * abs(reward) — keep cases that are recent and informative
        scored = [
            (i, c.timestamp / max(1, self._counter) * (0.1 + abs(c.reward)))
            for i, c in enumerate(self.cases)
        ]
        scored.sort(key=lambda x: x[1])
        # Remove bottom 10%
        n_remove = max(1, len(self.cases) // 10)
        indices_to_remove = {scored[i][0] for i in range(n_remove)}
        self.cases = [c for i, c in enumerate(self.cases) if i not in indices_to_remove]

    def retrieve(
        self,
        query: CaseQuery,
        k: int = 10,
        *,
        street_weight: float = 0.3,
        type_weight: float = 0.2,
        feature_weight: float = 0.5,
    ) -> list[tuple[PokerCase, float]]:
        """Retrieve k most similar cases.

        Similarity combines:
        - Feature vector similarity (cosine)
        - Street match bonus
        - Opponent type match bonus

        Returns list of (case, similarity_score) sorted by similarity desc.
        """
        if not self.cases:
            return []

        scored: list[tuple[PokerCase, float]] = []
        for case in self.cases:
            # Feature similarity
            if query.features and case.features:
                feat_sim = max(0.0, _cosine_similarity(query.features, case.features))
            else:
                feat_sim = 0.0

            # Street match
            street_sim = 1.0 if case.street == query.street else 0.0

            # Opponent type match
            type_sim = 1.0 if case.opponent_type == query.opponent_type else 0.0

            total_sim = (
                feature_weight * feat_sim
                + street_weight * street_sim
                + type_weight * type_sim
            )
            scored.append((case, total_sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def size(self) -> int:
        return len(self.cases)


class CASPERAgent:
    """Case-based reasoning agent for poker decisions.

    Retrieves similar past situations and votes on the best action
    based on outcomes of those historical cases.
    """

    def __init__(
        self,
        case_base: CaseBase | None = None,
        k: int = 10,
        recency_decay: float = 0.995,
        seed: int = 42,
    ) -> None:
        self.case_base = case_base or CaseBase(seed=seed)
        self.k = k
        self.recency_decay = recency_decay
        self.rng = random.Random(seed)

    def decide(self, query: CaseQuery) -> ActionDistribution:
        """Decide on an action using case-based reasoning.

        Retrieves k similar cases, weights by similarity and outcome,
        and produces a probability distribution over actions.
        """
        cases = self.case_base.retrieve(query, k=self.k)

        if not cases:
            # No cases — return uniform over common actions
            return ActionDistribution(probabilities={
                ActionType.FOLD: 0.25,
                ActionType.CHECK: 0.25,
                ActionType.CALL: 0.25,
                ActionType.RAISE: 0.25,
            })

        action_scores: dict[ActionType, float] = {}
        max_ts = max(c.timestamp for c, _ in cases) if cases else 1

        for case, similarity in cases:
            # Recency weight
            age = max_ts - case.timestamp
            recency = self.recency_decay ** age

            # Outcome weight: sigmoid of reward to keep bounded
            outcome_w = 1.0 / (1.0 + math.exp(-case.reward * 0.5))

            # Combined weight
            weight = similarity * recency * outcome_w

            action = case.action_taken
            action_scores[action] = action_scores.get(action, 0.0) + weight

        # Normalize to distribution
        total = sum(action_scores.values())
        if total > 0:
            probs = {a: v / total for a, v in action_scores.items()}
        else:
            probs = {ActionType.CHECK: 1.0}

        return ActionDistribution(probabilities=probs)

    def record_outcome(
        self,
        features: list[float],
        action: ActionType,
        reward: float,
        opponent_type: str = "unknown",
        street: int = 0,
    ) -> None:
        """Record a case after observing the outcome."""
        case = PokerCase(
            features=features,
            action_taken=action,
            reward=reward,
            opponent_type=opponent_type,
            street=street,
        )
        self.case_base.add_case(case)

    def case_count(self) -> int:
        return self.case_base.size()
