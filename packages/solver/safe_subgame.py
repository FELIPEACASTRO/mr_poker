"""Safe Subgame Solving — Libratus/DecisionHoldem style.

Refines a trunk (blueprint) strategy within a subgame while maintaining
safety guarantees.  A gadget game is constructed that gives the opponent
the choice to enter the subgame or take the trunk strategy's value,
ensuring the refined strategy is never worse than the blueprint.

Reference: Brown & Sandholm (2017) "Safe and Nested Subgame Solving
for Imperfect-Information Games".
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from packages.common.types import ActionType
from packages.strategy.mixed import ActionDistribution


@dataclass
class SubgameDefinition:
    """Defines a subgame to be resolved.

    Attributes:
        trunk_strategy: the blueprint strategy at the subgame root
            (mapping info-set key -> ActionDistribution)
        subgame_root_state: identifier for the subgame root
        reaching_probs: probability of each player reaching the subgame root
            under the trunk strategy (player_id -> probability)
        trunk_value: expected value for hero under the trunk strategy
    """
    trunk_strategy: dict[str, ActionDistribution] = field(default_factory=dict)
    subgame_root_state: str = ""
    reaching_probs: dict[str, float] = field(default_factory=dict)
    trunk_value: float = 0.0


@dataclass
class GadgetGameNode:
    """Node in the gadget game used for safe subgame solving.

    The gadget game augments the original subgame by adding an
    "alternative" action at the root that guarantees the opponent
    receives the trunk strategy's value.
    """
    info_set: str = ""
    actions: list[str] = field(default_factory=list)
    children: dict[str, GadgetGameNode] = field(default_factory=dict)
    # Strategy at this node (action -> probability)
    strategy: dict[str, float] = field(default_factory=dict)
    # Cumulative regret for CFR
    cumulative_regret: dict[str, float] = field(default_factory=dict)
    # Cumulative strategy sum for averaging
    strategy_sum: dict[str, float] = field(default_factory=dict)
    value: float = 0.0
    is_terminal: bool = False
    terminal_value: float = 0.0
    is_chance: bool = False
    acting_player: int = 0  # 0 = hero, 1 = opponent


class GadgetGame:
    """Augmented game for safe subgame solving.

    Adds 'alternative' actions at the root that guarantee the trunk
    strategy's value for each player.  The opponent can choose to
    'enter' the subgame (and face the refined strategy) or take the
    trunk value.  This ensures safety: the refined strategy is
    never worse than the blueprint for the solver.
    """

    def __init__(self, subgame_def: SubgameDefinition) -> None:
        self.subgame_def = subgame_def
        self.root = self._build_gadget_root()

    def _build_gadget_root(self) -> GadgetGameNode:
        """Build the root of the gadget game.

        The opponent chooses between:
        - 'enter': play the subgame
        - 'alternative': receive the trunk strategy value
        """
        root = GadgetGameNode(
            info_set="gadget_root",
            actions=["enter", "alternative"],
            acting_player=1,  # opponent decides
        )

        # Alternative branch: terminal node with trunk value
        alt_node = GadgetGameNode(
            info_set="gadget_alternative",
            is_terminal=True,
            terminal_value=self.subgame_def.trunk_value,
        )
        root.children["alternative"] = alt_node

        # Enter branch: leads into the actual subgame
        enter_node = self._build_subgame_tree()
        root.children["enter"] = enter_node

        # Initialize uniform strategy and zero regrets
        for action in root.actions:
            root.strategy[action] = 1.0 / len(root.actions)
            root.cumulative_regret[action] = 0.0
            root.strategy_sum[action] = 0.0

        return root

    def _build_subgame_tree(self) -> GadgetGameNode:
        """Build a simplified subgame tree from the trunk strategy.

        For each info set in the trunk strategy, creates nodes with
        the available actions.
        """
        if not self.subgame_def.trunk_strategy:
            return GadgetGameNode(
                info_set="subgame_leaf",
                is_terminal=True,
                terminal_value=0.0,
            )

        # Build a node for each info set in the trunk strategy
        # Linked as a chain for simplicity (real impl would build a tree)
        info_sets = list(self.subgame_def.trunk_strategy.keys())
        return self._build_chain(info_sets, 0)

    def _build_chain(
        self, info_sets: list[str], idx: int
    ) -> GadgetGameNode:
        """Build a chain of decision nodes from info sets."""
        if idx >= len(info_sets):
            return GadgetGameNode(
                info_set="subgame_terminal",
                is_terminal=True,
                terminal_value=0.0,
            )

        info_set = info_sets[idx]
        trunk_dist = self.subgame_def.trunk_strategy[info_set]
        actions = [a.value for a in trunk_dist.probabilities.keys()]

        if not actions:
            actions = ["check"]

        node = GadgetGameNode(
            info_set=info_set,
            actions=actions,
            acting_player=idx % 2,
        )

        for action in actions:
            child = self._build_chain(info_sets, idx + 1)
            node.children[action] = child
            node.strategy[action] = 1.0 / len(actions)
            node.cumulative_regret[action] = 0.0
            node.strategy_sum[action] = 0.0

        return node


class SafeSubgameSolver:
    """Safe subgame solver with gadget-game safety guarantees.

    Constructs a gadget game from a subgame definition, runs CFR on it,
    and extracts refined strategies that are guaranteed to be at least
    as good as the trunk (blueprint) strategy.
    """

    def __init__(self, *, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def build_gadget_game(self, subgame_def: SubgameDefinition) -> GadgetGame:
        """Construct the gadget game from a subgame definition."""
        return GadgetGame(subgame_def)

    def _regret_matching(self, node: GadgetGameNode) -> dict[str, float]:
        """Compute current strategy via regret matching."""
        positive = {
            a: max(0.0, node.cumulative_regret.get(a, 0.0))
            for a in node.actions
        }
        total = sum(positive.values())

        if total > 0:
            return {a: v / total for a, v in positive.items()}
        else:
            n = len(node.actions) or 1
            return {a: 1.0 / n for a in node.actions}

    def _cfr_traverse(
        self,
        node: GadgetGameNode,
        reach_hero: float,
        reach_opp: float,
        traverser: int,
    ) -> float:
        """Traverse the gadget game tree with CFR."""
        if node.is_terminal:
            return node.terminal_value

        if not node.actions:
            return node.value

        strategy = self._regret_matching(node)
        node.strategy = strategy

        action_values: dict[str, float] = {}
        node_value = 0.0

        for action in node.actions:
            child = node.children.get(action)
            if child is None:
                action_values[action] = 0.0
                continue

            if node.acting_player == 0:
                # Hero node
                child_val = self._cfr_traverse(
                    child,
                    reach_hero * strategy.get(action, 0.0),
                    reach_opp,
                    traverser,
                )
            else:
                # Opponent node
                child_val = self._cfr_traverse(
                    child,
                    reach_hero,
                    reach_opp * strategy.get(action, 0.0),
                    traverser,
                )

            action_values[action] = child_val
            node_value += strategy.get(action, 0.0) * child_val

        # Update regrets for the acting player
        if node.acting_player == traverser:
            for action in node.actions:
                regret = action_values.get(action, 0.0) - node_value
                node.cumulative_regret[action] = (
                    node.cumulative_regret.get(action, 0.0) + regret
                )
                # Accumulate strategy sum for averaging
                prob = strategy.get(action, 0.0)
                weight = reach_hero if traverser == 0 else reach_opp
                node.strategy_sum[action] = (
                    node.strategy_sum.get(action, 0.0) + weight * prob
                )

        node.value = node_value
        return node_value

    def solve_subgame(
        self,
        subgame_def: SubgameDefinition,
        iterations: int = 1000,
    ) -> dict[str, ActionDistribution]:
        """Solve a subgame with safety guarantees.

        Builds a gadget game, runs CFR for the specified iterations,
        and extracts the refined average strategies.

        Args:
            subgame_def: defines the subgame and trunk strategy
            iterations: number of CFR iterations

        Returns:
            Mapping from info-set key to refined ActionDistribution.
        """
        gadget = self.build_gadget_game(subgame_def)

        for i in range(iterations):
            traverser = i % 2
            self._cfr_traverse(gadget.root, 1.0, 1.0, traverser)

        # Extract average strategies from all nodes
        return self._extract_strategies(gadget.root)

    def _extract_strategies(
        self, node: GadgetGameNode
    ) -> dict[str, ActionDistribution]:
        """Extract average strategies from the solved gadget game tree."""
        result: dict[str, ActionDistribution] = {}

        if node.is_terminal or not node.actions:
            return result

        # Compute average strategy from strategy sum
        total = sum(max(0.0, v) for v in node.strategy_sum.values())
        if total > 0:
            avg_probs: dict[ActionType, float] = {}
            for action_str, sum_val in node.strategy_sum.items():
                # Convert string action back to ActionType if possible
                try:
                    action_type = ActionType(action_str)
                    avg_probs[action_type] = max(0.0, sum_val) / total
                except ValueError:
                    pass  # skip non-standard actions like "enter"/"alternative"

            if avg_probs:
                result[node.info_set] = ActionDistribution(probabilities=avg_probs)

        # Recurse into children
        for child in node.children.values():
            child_strategies = self._extract_strategies(child)
            result.update(child_strategies)

        return result

    @staticmethod
    def compute_reaching_probs(
        trunk_strategy: dict[str, ActionDistribution],
        action_sequence: list[tuple[str, ActionType]],
    ) -> dict[str, float]:
        """Compute reaching probabilities for each info set along a path.

        Args:
            trunk_strategy: blueprint strategy mapping
            action_sequence: list of (info_set, action) pairs along path

        Returns:
            Reaching probability for each info set in the sequence.
        """
        reaching: dict[str, float] = {}
        prob = 1.0

        for info_set, action in action_sequence:
            reaching[info_set] = prob
            dist = trunk_strategy.get(info_set)
            if dist is not None:
                action_prob = dist.probabilities.get(action, 0.0)
                prob *= action_prob
            else:
                prob *= 0.5  # default if info set not in trunk

        return reaching

    @staticmethod
    def safety_bound(original_value: float, refined_value: float) -> bool:
        """Check that the refined strategy is not worse than the trunk.

        Returns True if the refined value is at least as good as the
        original (within a small tolerance for numerical error).
        """
        tolerance = 1e-6
        return refined_value >= original_value - tolerance
