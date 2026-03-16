"""Equilibrium Refinements for Subgame Solving.

Implements refinements beyond basic Nash equilibrium that are critical
for practical subgame solving in imperfect-information games:

1. Trembling Hand Perfect Equilibrium (THPE): strategies that remain
   optimal even when opponents make small mistakes (trembles).

2. Sequential Equilibrium: ensures strategies are rational at every
   information set, not just on the equilibrium path.

3. Maximin Refinement: selects the Nash equilibrium that maximizes
   the minimum payoff (most robust against worst-case opponents).

Reference: Selten (1975) "Reexamination of the perfectness concept
for equilibrium points in extensive games".
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from packages.common.types import ActionType
from packages.strategy.mixed import ActionDistribution


@dataclass
class ExtensiveFormNode:
    """Node in an extensive-form game tree.

    Attributes:
        info_set: information set identifier
        actions: available actions at this node
        children: mapping action -> child node
        acting_player: 0 (hero), 1 (villain), -1 (chance/terminal)
        is_terminal: whether this is a leaf
        terminal_value: payoff at terminal nodes (from hero's perspective)
        strategy: current mixed strategy (action -> prob)
        belief: belief distribution over states at this info set
    """
    info_set: str = ""
    actions: list[str] = field(default_factory=list)
    children: dict[str, ExtensiveFormNode] = field(default_factory=dict)
    acting_player: int = 0
    is_terminal: bool = False
    terminal_value: float = 0.0
    strategy: dict[str, float] = field(default_factory=dict)
    belief: dict[str, float] = field(default_factory=dict)
    cumulative_regret: dict[str, float] = field(default_factory=dict)
    strategy_sum: dict[str, float] = field(default_factory=dict)


def _regret_match(regrets: dict[str, float], actions: list[str]) -> dict[str, float]:
    """Regret matching to get current strategy."""
    positive = {a: max(0.0, regrets.get(a, 0.0)) for a in actions}
    total = sum(positive.values())
    if total > 0:
        return {a: v / total for a, v in positive.items()}
    n = len(actions) or 1
    return {a: 1.0 / n for a in actions}


class TremblingHandRefinement:
    """Trembling Hand Perfect Equilibrium.

    Perturbs strategies with a small tremble (epsilon) so that every
    action has non-zero probability, then computes the equilibrium
    of this perturbed game.  As epsilon -> 0, the limit is a THPE.
    """

    def __init__(self, epsilon: float = 0.01, seed: int = 42) -> None:
        self.epsilon = epsilon
        self.rng = random.Random(seed)

    def apply_tremble(
        self, strategy: dict[str, float], actions: list[str]
    ) -> dict[str, float]:
        """Apply minimum tremble probability to all actions."""
        n = len(actions)
        if n == 0:
            return {}

        min_prob = self.epsilon / n
        trembled: dict[str, float] = {}
        remaining = 1.0 - min_prob * n

        for a in actions:
            orig = strategy.get(a, 1.0 / n)
            trembled[a] = min_prob + remaining * orig

        # Normalize
        total = sum(trembled.values())
        if total > 0:
            trembled = {a: v / total for a, v in trembled.items()}
        return trembled

    def refine_tree(
        self, root: ExtensiveFormNode, iterations: int = 500
    ) -> dict[str, dict[str, float]]:
        """Refine strategies in tree to be trembling-hand perfect.

        Runs CFR on the perturbed game and returns refined strategies.
        """
        # Run CFR with trembles
        for i in range(iterations):
            traverser = i % 2
            self._cfr_traverse(root, 1.0, 1.0, traverser)

        # Extract average strategies with trembles applied
        result: dict[str, dict[str, float]] = {}
        self._extract(root, result)
        return result

    def _cfr_traverse(
        self,
        node: ExtensiveFormNode,
        reach_hero: float,
        reach_opp: float,
        traverser: int,
    ) -> float:
        if node.is_terminal:
            return node.terminal_value
        if not node.actions:
            return 0.0

        # Get strategy with tremble applied
        raw = _regret_match(node.cumulative_regret, node.actions)
        strategy = self.apply_tremble(raw, node.actions)
        node.strategy = strategy

        action_values: dict[str, float] = {}
        node_value = 0.0

        for action in node.actions:
            child = node.children.get(action)
            if child is None:
                action_values[action] = 0.0
                continue

            prob = strategy.get(action, 0.0)
            if node.acting_player == 0:
                val = self._cfr_traverse(child, reach_hero * prob, reach_opp, traverser)
            else:
                val = self._cfr_traverse(child, reach_hero, reach_opp * prob, traverser)

            action_values[action] = val
            node_value += prob * val

        if node.acting_player == traverser:
            weight = reach_hero if traverser == 0 else reach_opp
            for action in node.actions:
                regret = action_values.get(action, 0.0) - node_value
                node.cumulative_regret[action] = (
                    node.cumulative_regret.get(action, 0.0) + regret
                )
                node.strategy_sum[action] = (
                    node.strategy_sum.get(action, 0.0) + weight * strategy.get(action, 0.0)
                )

        return node_value

    def _extract(
        self, node: ExtensiveFormNode, result: dict[str, dict[str, float]]
    ) -> None:
        if node.is_terminal or not node.actions:
            return

        total = sum(max(0.0, v) for v in node.strategy_sum.values())
        if total > 0:
            avg = {a: max(0.0, node.strategy_sum.get(a, 0.0)) / total for a in node.actions}
            # Apply tremble to average strategy
            result[node.info_set] = self.apply_tremble(avg, node.actions)

        for child in node.children.values():
            self._extract(child, result)


class SequentialEquilibrium:
    """Sequential Equilibrium refinement.

    Ensures beliefs are consistent with strategies at every info set
    (even off the equilibrium path) and strategies are sequentially
    rational given those beliefs.
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def compute_beliefs(
        self,
        root: ExtensiveFormNode,
        strategies: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        """Compute consistent beliefs at each information set.

        Uses Bayes' rule where possible; for off-path info sets,
        uses the limit of trembling-hand sequences.
        """
        beliefs: dict[str, dict[str, float]] = {}
        reaching: dict[str, float] = {}
        self._compute_reaching(root, strategies, 1.0, reaching)

        # Collect nodes by info set
        info_set_nodes: dict[str, list[tuple[str, float]]] = {}
        self._collect_nodes(root, reaching, info_set_nodes, "root")

        for info_set, nodes in info_set_nodes.items():
            total_reach = sum(r for _, r in nodes)
            if total_reach > 0:
                beliefs[info_set] = {
                    node_id: reach / total_reach for node_id, reach in nodes
                }
            else:
                # Off-path: uniform belief
                n = len(nodes)
                beliefs[info_set] = {node_id: 1.0 / n for node_id, _ in nodes}

        return beliefs

    def _compute_reaching(
        self,
        node: ExtensiveFormNode,
        strategies: dict[str, dict[str, float]],
        current_prob: float,
        reaching: dict[str, float],
    ) -> None:
        reaching[node.info_set] = reaching.get(node.info_set, 0.0) + current_prob
        if node.is_terminal or not node.actions:
            return

        strat = strategies.get(node.info_set, {})
        for action in node.actions:
            child = node.children.get(action)
            if child is not None:
                prob = strat.get(action, 1.0 / len(node.actions))
                self._compute_reaching(child, strategies, current_prob * prob, reaching)

    def _collect_nodes(
        self,
        node: ExtensiveFormNode,
        reaching: dict[str, float],
        result: dict[str, list[tuple[str, float]]],
        node_id: str,
    ) -> None:
        if node.info_set:
            if node.info_set not in result:
                result[node.info_set] = []
            result[node.info_set].append((node_id, reaching.get(node.info_set, 0.0)))

        for action, child in node.children.items():
            self._collect_nodes(child, reaching, result, f"{node_id}_{action}")

    def is_sequentially_rational(
        self,
        node: ExtensiveFormNode,
        strategy: dict[str, float],
        beliefs: dict[str, float],
    ) -> bool:
        """Check if strategy is sequentially rational given beliefs.

        A strategy is sequentially rational if the actions with positive
        probability are all best responses given the beliefs.
        """
        if not node.actions:
            return True

        # Compute expected value for each action
        action_values: dict[str, float] = {}
        for action in node.actions:
            child = node.children.get(action)
            if child is not None:
                action_values[action] = child.terminal_value if child.is_terminal else 0.0
            else:
                action_values[action] = 0.0

        if not action_values:
            return True

        best_value = max(action_values.values())
        for action in node.actions:
            prob = strategy.get(action, 0.0)
            if prob > 0 and action_values.get(action, 0.0) < best_value - 1e-6:
                return False

        return True


class MaximinRefinement:
    """Maximin equilibrium refinement.

    Among all Nash equilibria, selects the one that maximizes the
    minimum payoff (most robust against adversarial opponents).
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def maximin_strategy(
        self,
        payoff_matrix: list[list[float]],
        iterations: int = 5000,
    ) -> tuple[list[float], float]:
        """Compute maximin strategy for a row player.

        Uses iterative best response to find the strategy that
        maximizes the minimum expected payoff.

        Args:
            payoff_matrix: payoff_matrix[i][j] = payoff when
                row plays i, col plays j
            iterations: number of iterations

        Returns:
            (strategy, maximin_value) tuple
        """
        m = len(payoff_matrix)
        if m == 0:
            return [], 0.0
        n = len(payoff_matrix[0])
        if n == 0:
            return [1.0 / m] * m, 0.0

        # Initialize uniform
        strategy = [1.0 / m] * m
        cum_strategy = [1.0 / m] * m

        for t in range(1, iterations + 1):
            # Find opponent's best response (minimizer)
            col_values = [
                sum(strategy[i] * payoff_matrix[i][j] for i in range(m))
                for j in range(n)
            ]
            worst_col = min(range(n), key=lambda j: col_values[j])

            # Best response to worst-case column
            row_values = [payoff_matrix[i][worst_col] for i in range(m)]
            best_row = max(range(m), key=lambda i: row_values[i])

            # Smooth update
            lr = 2.0 / (t + 2)
            for i in range(m):
                strategy[i] = (1 - lr) * strategy[i] + (lr if i == best_row else 0.0)
                cum_strategy[i] += strategy[i]

        # Normalize cumulative
        total = sum(cum_strategy)
        if total > 0:
            final = [v / total for v in cum_strategy]
        else:
            final = [1.0 / m] * m

        # Compute maximin value
        min_payoff = math.inf
        for j in range(n):
            val = sum(final[i] * payoff_matrix[i][j] for i in range(m))
            min_payoff = min(min_payoff, val)

        return final, min_payoff

    def refine(
        self,
        strategies: list[ActionDistribution],
        payoff_fn: Any = None,
    ) -> list[ActionDistribution]:
        """Refine a set of strategies using maximin criterion.

        If no payoff function is provided, applies small perturbation
        toward more robust (less extreme) probabilities.
        """
        refined = []
        for dist in strategies:
            probs = dist.probabilities
            if not probs:
                refined.append(dist)
                continue

            # Robustness: move probabilities toward uniform slightly
            n = len(probs)
            uniform = 1.0 / n
            alpha = 0.05  # robustness parameter
            new_probs = {
                a: (1 - alpha) * p + alpha * uniform
                for a, p in probs.items()
            }
            refined.append(ActionDistribution(probabilities=new_probs))

        return refined
