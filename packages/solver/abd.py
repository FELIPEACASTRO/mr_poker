"""ABD — Augmented Backward Dynamic programming for depth-limited solving.

Performs backward induction on a game tree with an opponent model at
leaf nodes.  Instead of rolling out to terminal states, the solver uses
the opponent model's probability distribution over hands to weight
showdown outcomes, yielding superior play against sub-rational opponents.

Reference: inspired by depth-limited solving in DeepStack/Libratus.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from packages.common.types import ActionType
from packages.cfr_agent.deep_cfr import SimpleNN, FEATURE_DIM, NUM_ACTIONS, ACTION_INDEX
from packages.strategy.mixed import ActionDistribution


@dataclass
class GameTreeNode:
    """Node in the depth-limited game tree."""
    action: ActionType | None = None     # action taken to reach this node
    children: list[GameTreeNode] = field(default_factory=list)
    value: float = 0.0                    # expected value at this node
    strategy: ActionDistribution | None = None  # strategy at this node
    is_leaf: bool = False
    is_terminal: bool = False
    acting_seat: int | None = None
    depth: int = 0
    # State info for leaf evaluation
    pot: float = 0.0
    street: float = 0.0
    features: list[float] = field(default_factory=list)


class LeafEvaluator:
    """Evaluates leaf nodes using an opponent model.

    Computes P(opponent_hand | actions) from the opponent model and
    weights showdown outcomes accordingly.  This yields a more accurate
    value estimate than a uniform assumption.
    """

    def __init__(self, *, hidden_dim: int = 32, seed: int = 42) -> None:
        # Opponent model: features -> probability of each opponent hand bucket
        # We use NUM_ACTIONS buckets as a proxy for hand-strength categories
        self.opponent_model = SimpleNN(
            FEATURE_DIM, hidden_dim, NUM_ACTIONS, seed=seed,
        )
        self.rng = random.Random(seed)

    def opponent_hand_probs(self, features: list[float]) -> list[float]:
        """Estimate P(opponent_hand_bucket | features) via softmax."""
        raw = self.opponent_model.forward(features)
        max_val = max(raw) if raw else 0.0
        exp_vals = [math.exp(v - max_val) for v in raw]
        total = sum(exp_vals)
        if total <= 0:
            return [1.0 / len(raw)] * len(raw)
        return [v / total for v in exp_vals]

    def evaluate(
        self,
        features: list[float],
        pot: float,
        hero_strength: float = 0.5,
    ) -> float:
        """Evaluate a leaf node using opponent model-weighted outcomes.

        Args:
            features: info-set feature vector at the leaf
            pot: current pot size
            hero_strength: hero's hand strength estimate [0, 1]

        Returns:
            Expected value at this leaf.
        """
        opp_probs = self.opponent_hand_probs(features)

        # Each bucket represents an opponent strength range
        # Bucket i has strength = (i + 0.5) / num_buckets
        num_buckets = len(opp_probs)
        expected_value = 0.0

        for i, prob in enumerate(opp_probs):
            opp_strength = (i + 0.5) / num_buckets
            if hero_strength > opp_strength:
                # Hero wins
                expected_value += prob * pot
            elif hero_strength < opp_strength:
                # Hero loses
                expected_value -= prob * pot
            # Ties contribute 0

        return expected_value

    def train(
        self,
        features_list: list[list[float]],
        target_probs_list: list[list[float]],
        lr: float = 0.001,
    ) -> float:
        """Train opponent model on observed data."""
        if not features_list:
            return 0.0
        total_loss = 0.0
        for features, target in zip(features_list, target_probs_list):
            loss = self.opponent_model.train_step(features, target, lr=lr)
            total_loss += loss
        return total_loss / len(features_list)


class ABDSolver:
    """Augmented Backward Dynamic programming solver.

    Builds a game tree to a configurable depth limit and uses a
    LeafEvaluator (opponent model) at leaf nodes instead of rollout.
    Backward induction computes the optimal strategy from leaves up.
    """

    def __init__(
        self,
        *,
        depth_limit: int = 4,
        hidden_dim: int = 32,
        seed: int = 42,
    ) -> None:
        self.depth_limit = depth_limit
        self.leaf_evaluator = LeafEvaluator(hidden_dim=hidden_dim, seed=seed)
        self.rng = random.Random(seed)

    def _extract_simple_features(self, state: Any, seat: int) -> list[float]:
        """Extract a simple feature vector without requiring full game state."""
        try:
            from packages.cfr_agent.deep_cfr import extract_features
            feats = extract_features(state, seat)
            return feats.to_vector()
        except Exception:
            return [0.0] * FEATURE_DIM

    def _build_tree(
        self,
        runtime: Any,
        engine: Any,
        depth: int,
        hero_seat: int,
    ) -> GameTreeNode:
        """Recursively build the game tree to depth_limit."""
        state = runtime.state
        node = GameTreeNode(depth=depth)

        # Terminal node
        if state.is_terminal or state.acting_seat is None:
            node.is_terminal = True
            node.is_leaf = True
            node.value = state.players[hero_seat].stack
            return node

        # Leaf node at depth limit — use opponent model
        if depth >= self.depth_limit:
            node.is_leaf = True
            features = self._extract_simple_features(state, hero_seat)
            node.features = features
            node.pot = float(state.pot)
            node.street = len(state.board) / 5.0
            # Estimate hero strength from features (high_rank + low_rank avg)
            hero_strength = (features[0] + features[1]) / 2.0 if len(features) >= 2 else 0.5
            node.value = self.leaf_evaluator.evaluate(
                features, float(state.pot), hero_strength,
            )
            return node

        node.acting_seat = state.acting_seat
        legal_actions = engine.legal_actions(runtime)

        for action in legal_actions:
            child_runtime = self._clone_and_act(runtime, engine, action)
            if child_runtime is None:
                child = GameTreeNode(
                    action=action, is_leaf=True, is_terminal=True,
                    value=0.0, depth=depth + 1,
                )
            else:
                child = self._build_tree(child_runtime, engine, depth + 1, hero_seat)
            child.action = action
            node.children.append(child)

        return node

    def backward_induction(self, node: GameTreeNode, hero_seat: int = 0) -> float:
        """Compute optimal strategy from leaves up via backward induction.

        At hero nodes, picks the action maximizing value.
        At opponent nodes, assumes opponent plays proportional to value
        (weighted by opponent model).

        Returns the node's computed value.
        """
        if node.is_leaf or not node.children:
            return node.value

        # Compute child values recursively
        action_values: dict[ActionType, float] = {}
        for child in node.children:
            child_val = self.backward_induction(child, hero_seat)
            if child.action is not None:
                action_values[child.action] = child_val

        if not action_values:
            node.value = 0.0
            return 0.0

        if node.acting_seat == hero_seat:
            # Hero: maximize expected value
            best_action = max(action_values, key=action_values.get)  # type: ignore[arg-type]
            node.value = action_values[best_action]

            # Strategy: put all weight on best action (pure strategy)
            # with small epsilon for exploration
            probs: dict[ActionType, float] = {}
            epsilon = 0.05
            n = len(action_values)
            for action, val in action_values.items():
                if action == best_action:
                    probs[action] = 1.0 - epsilon + epsilon / n
                else:
                    probs[action] = epsilon / n
            node.strategy = ActionDistribution(probabilities=probs)
        else:
            # Opponent: assume mixed strategy proportional to softmax of values
            max_val = max(action_values.values())
            exp_vals = {a: math.exp(v - max_val) for a, v in action_values.items()}
            total = sum(exp_vals.values())
            if total > 0:
                probs = {a: v / total for a, v in exp_vals.items()}
            else:
                n_act = len(action_values) or 1
                probs = {a: 1.0 / n_act for a in action_values}

            node.strategy = ActionDistribution(probabilities=probs)
            # Node value = expected value under opponent's mixed strategy
            node.value = sum(probs[a] * action_values[a] for a in action_values)

        return node.value

    def solve(
        self,
        runtime: Any,
        engine: Any,
        depth_limit: int | None = None,
        hero_seat: int = 0,
    ) -> ActionDistribution:
        """Solve from the current game state using depth-limited ABD.

        Args:
            runtime: HandRuntime from the game engine
            engine: GameEngine instance
            depth_limit: override default depth limit
            hero_seat: seat of the hero player

        Returns:
            ActionDistribution for the hero at the root.
        """
        if depth_limit is not None:
            old_limit = self.depth_limit
            self.depth_limit = depth_limit

        root = self._build_tree(runtime, engine, 0, hero_seat)
        self.backward_induction(root, hero_seat)

        if depth_limit is not None:
            self.depth_limit = old_limit

        if root.strategy is not None:
            return root.strategy

        # Fallback: uniform over legal actions
        legal = engine.legal_actions(runtime)
        n = len(legal) or 1
        return ActionDistribution(probabilities={a: 1.0 / n for a in legal})

    def _clone_and_act(
        self, runtime: Any, engine: Any, action: ActionType
    ) -> Any | None:
        """Clone game state and apply an action."""
        from packages.cfr_agent.trainer import _size_action

        state = runtime.state
        if state.acting_seat is None:
            return None

        player = state.players[state.acting_seat]
        amount = _size_action(action, state, player, engine)

        all_hole_cards: list[str] = []
        for s in sorted(state.players.keys()):
            for c in state.players[s].hole_cards:
                all_hole_cards.append(str(c))
        board_cards = [str(c) for c in state.board]
        burned = [str(c) for c in runtime.burned_cards]
        deck_prefix = all_hole_cards + burned[:len(state.board)] + board_cards

        try:
            new_runtime = engine.start_new_hand(
                stacks=tuple(
                    runtime.initial_stacks[s] for s in sorted(runtime.initial_stacks)
                ),
                button_seat=state.button_seat,
                deck_prefix=deck_prefix,
            )
            for ev in state.actions:
                if new_runtime.state.is_terminal or new_runtime.state.acting_seat is None:
                    break
                engine.apply_action(new_runtime, ev.action_type, ev.amount)
            if new_runtime.state.is_terminal or new_runtime.state.acting_seat is None:
                return None
            engine.apply_action(new_runtime, action, amount)
            return new_runtime
        except Exception:
            return None
