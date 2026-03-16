"""Bayes-Relational Opponent Model — Relational features with Bayesian tree leaves.

Builds a shallow decision tree (depth 3-4) over *relational features* that
capture action-sequence patterns (e.g., "raised preflop AND bet flop").
Each leaf holds a Bayesian posterior over opponent action distributions,
enabling fast adaptation with <50 hands.

Combines ideas from:
- Bayesian opponent modeling (Southey et al., UAI 2005)
- Relational learning for game agents (Dzeroski et al., 2001)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from packages.common.types import ActionType
from packages.strategy.mixed import ActionDistribution

# Relational feature templates
RELATIONAL_TEMPLATES = [
    ("raised_preflop", "bet_flop"),
    ("raised_preflop", "check_flop"),
    ("called_preflop", "bet_flop"),
    ("called_preflop", "check_flop"),
    ("bet_flop", "bet_turn"),
    ("bet_flop", "check_turn"),
    ("check_flop", "bet_turn"),
    ("check_flop", "check_turn"),
    ("bet_turn", "bet_river"),
    ("bet_turn", "check_river"),
    ("raised_preflop", "bet_flop", "bet_turn"),
    ("called_preflop", "check_flop", "check_turn"),
    ("raised_preflop", "check_flop", "bet_turn"),
    ("called_preflop", "bet_flop", "bet_turn"),
]

# Action labels for prediction
PREDICT_ACTIONS = [
    ActionType.FOLD,
    ActionType.CHECK,
    ActionType.CALL,
    ActionType.BET,
    ActionType.RAISE,
    ActionType.ALL_IN,
]

NUM_PREDICT_ACTIONS = len(PREDICT_ACTIONS)
ACTION_TO_IDX = {a: i for i, a in enumerate(PREDICT_ACTIONS)}


@dataclass
class RelationalFeature:
    """A relational feature capturing a sequence pattern.

    Attributes:
        pattern: tuple of action descriptors (e.g., ("raised_preflop", "bet_flop")).
        name: human-readable name.
    """
    pattern: tuple[str, ...]
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = " AND ".join(self.pattern)

    def matches(self, action_history: list[str]) -> bool:
        """Check if the action history contains this relational pattern."""
        history_set = set(action_history)
        return all(p in history_set for p in self.pattern)


def extract_relational_actions(
    actions: list[tuple[ActionType, str]],
) -> list[str]:
    """Convert a list of (action, street) pairs into relational action descriptors.

    For example: (RAISE, "pre_flop") -> "raised_preflop"
    """
    mapping = {
        (ActionType.RAISE, "pre_flop"): "raised_preflop",
        (ActionType.BET, "pre_flop"): "raised_preflop",  # preflop bet = raise
        (ActionType.CALL, "pre_flop"): "called_preflop",
        (ActionType.CHECK, "pre_flop"): "checked_preflop",
        (ActionType.FOLD, "pre_flop"): "folded_preflop",
        (ActionType.BET, "flop"): "bet_flop",
        (ActionType.RAISE, "flop"): "raised_flop",
        (ActionType.CHECK, "flop"): "check_flop",
        (ActionType.CALL, "flop"): "called_flop",
        (ActionType.FOLD, "flop"): "folded_flop",
        (ActionType.BET, "turn"): "bet_turn",
        (ActionType.RAISE, "turn"): "raised_turn",
        (ActionType.CHECK, "turn"): "check_turn",
        (ActionType.CALL, "turn"): "called_turn",
        (ActionType.FOLD, "turn"): "folded_turn",
        (ActionType.BET, "river"): "bet_river",
        (ActionType.RAISE, "river"): "raised_river",
        (ActionType.CHECK, "river"): "check_river",
        (ActionType.CALL, "river"): "called_river",
        (ActionType.FOLD, "river"): "folded_river",
        (ActionType.ALL_IN, "pre_flop"): "raised_preflop",
        (ActionType.ALL_IN, "flop"): "bet_flop",
        (ActionType.ALL_IN, "turn"): "bet_turn",
        (ActionType.ALL_IN, "river"): "bet_river",
    }
    result: list[str] = []
    for action, street in actions:
        key = (action, street)
        desc = mapping.get(key)
        if desc and desc not in result:
            result.append(desc)
    return result


@dataclass
class BayesianLeaf:
    """Bayesian posterior over action distribution at a tree leaf.

    Uses Dirichlet-Multinomial conjugacy:
        Prior: Dir(alpha)
        Posterior after observations: Dir(alpha + counts)
    """
    alpha: list[float] = field(default_factory=lambda: [1.0] * NUM_PREDICT_ACTIONS)
    counts: list[float] = field(default_factory=lambda: [0.0] * NUM_PREDICT_ACTIONS)
    total_observations: int = 0

    def update(self, action: ActionType) -> None:
        """Update posterior with an observed action."""
        idx = ACTION_TO_IDX.get(action)
        if idx is not None:
            self.counts[idx] += 1.0
            self.total_observations += 1

    def predict(self) -> dict[ActionType, float]:
        """Return MAP action distribution (posterior mean)."""
        posterior = [self.alpha[i] + self.counts[i] for i in range(NUM_PREDICT_ACTIONS)]
        total = sum(posterior)
        return {
            PREDICT_ACTIONS[i]: posterior[i] / total
            for i in range(NUM_PREDICT_ACTIONS)
        }

    def entropy(self) -> float:
        """Shannon entropy of the predictive distribution."""
        probs = self.predict()
        return -sum(p * math.log2(max(p, 1e-10)) for p in probs.values())

    def confidence(self) -> float:
        """Confidence based on observation count (0 to 1)."""
        # Confidence ramps up with observations, saturating around 50
        return 1.0 - 1.0 / (1.0 + self.total_observations / 10.0)


@dataclass
class RelationalNode:
    """Decision tree node with relational feature splitting.

    Internal nodes test a relational feature.
    Leaves hold Bayesian posteriors.
    """
    feature: RelationalFeature | None = None
    true_child: RelationalNode | None = None
    false_child: RelationalNode | None = None
    leaf: BayesianLeaf | None = None
    depth: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.leaf is not None

    def predict(self, relational_actions: list[str]) -> dict[ActionType, float]:
        """Traverse tree and return leaf prediction."""
        if self.is_leaf:
            assert self.leaf is not None
            return self.leaf.predict()
        assert self.feature is not None
        if self.feature.matches(relational_actions):
            assert self.true_child is not None
            return self.true_child.predict(relational_actions)
        assert self.false_child is not None
        return self.false_child.predict(relational_actions)

    def find_leaf(self, relational_actions: list[str]) -> BayesianLeaf:
        """Traverse tree and return the matching leaf."""
        if self.is_leaf:
            assert self.leaf is not None
            return self.leaf
        assert self.feature is not None
        if self.feature.matches(relational_actions):
            assert self.true_child is not None
            return self.true_child.find_leaf(relational_actions)
        assert self.false_child is not None
        return self.false_child.find_leaf(relational_actions)

    def all_leaves(self) -> list[BayesianLeaf]:
        """Collect all leaves in the tree."""
        if self.is_leaf:
            assert self.leaf is not None
            return [self.leaf]
        result: list[BayesianLeaf] = []
        if self.true_child:
            result.extend(self.true_child.all_leaves())
        if self.false_child:
            result.extend(self.false_child.all_leaves())
        return result


def _build_tree(
    features: list[RelationalFeature],
    depth: int,
    max_depth: int,
    rng: random.Random,
) -> RelationalNode:
    """Recursively build a relational decision tree."""
    if depth >= max_depth or not features:
        return RelationalNode(leaf=BayesianLeaf(), depth=depth)

    # Pick a feature for this node
    feat = features[0]
    remaining = features[1:]

    # Split remaining features roughly evenly
    mid = len(remaining) // 2
    true_feats = remaining[:mid]
    false_feats = remaining[mid:]

    true_child = _build_tree(true_feats, depth + 1, max_depth, rng)
    false_child = _build_tree(false_feats, depth + 1, max_depth, rng)

    return RelationalNode(
        feature=feat,
        true_child=true_child,
        false_child=false_child,
        depth=depth,
    )


class BayesRelationalModel:
    """Bayes-Relational opponent model.

    Combines relational features (action sequence patterns) with Bayesian
    leaf posteriors for fast adaptation to opponent tendencies.
    """

    def __init__(self, *, max_depth: int = 4, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.max_depth = max_depth

        # Build relational features
        self.features = [
            RelationalFeature(pattern=t) for t in RELATIONAL_TEMPLATES
        ]

        # Build the decision tree
        self.rng.shuffle(self.features)
        self.tree = _build_tree(
            list(self.features), depth=0, max_depth=max_depth, rng=self.rng
        )

        self._total_observations = 0

    def adapt(self, observations: list[dict[str, Any]]) -> None:
        """Update model with a batch of observations.

        Each observation is a dict with:
            - 'actions': list of (ActionType, street) tuples (the hand history)
            - 'next_action': ActionType (what opponent did next)

        This is the primary learning interface.
        """
        for obs in observations:
            action_history = obs.get("actions", [])
            next_action = obs.get("next_action")
            if next_action is None:
                continue

            relational = extract_relational_actions(action_history)
            leaf = self.tree.find_leaf(relational)
            leaf.update(next_action)
            self._total_observations += 1

    def predict_action(
        self, context: list[tuple[ActionType, str]]
    ) -> ActionDistribution:
        """Predict opponent's next action given hand history context.

        Args:
            context: list of (ActionType, street) tuples for the current hand.

        Returns:
            ActionDistribution over possible next actions.
        """
        relational = extract_relational_actions(context)
        probs = self.tree.predict(relational)
        return ActionDistribution(probabilities=probs)

    def predict_action_dict(
        self, context: list[tuple[ActionType, str]]
    ) -> dict[ActionType, float]:
        """Predict as a plain dict."""
        relational = extract_relational_actions(context)
        return self.tree.predict(relational)

    def confidence(self) -> float:
        """Overall model confidence based on observation count and tree coverage.

        Returns value in [0, 1]. Higher means more reliable predictions.
        """
        leaves = self.tree.all_leaves()
        if not leaves:
            return 0.0

        # Component 1: total observation confidence
        obs_conf = 1.0 - 1.0 / (1.0 + self._total_observations / 20.0)

        # Component 2: tree coverage (fraction of leaves with observations)
        covered = sum(1 for leaf in leaves if leaf.total_observations > 0)
        coverage = covered / len(leaves)

        # Component 3: average leaf confidence
        leaf_confs = [leaf.confidence() for leaf in leaves if leaf.total_observations > 0]
        avg_leaf_conf = sum(leaf_confs) / len(leaf_confs) if leaf_confs else 0.0

        return obs_conf * 0.4 + coverage * 0.3 + avg_leaf_conf * 0.3

    def reset(self) -> None:
        """Reset all leaf posteriors to uniform prior."""
        for leaf in self.tree.all_leaves():
            leaf.counts = [0.0] * NUM_PREDICT_ACTIONS
            leaf.total_observations = 0
        self._total_observations = 0

    @property
    def total_observations(self) -> int:
        return self._total_observations
