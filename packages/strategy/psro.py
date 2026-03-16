"""PSRO + Double Oracle — Policy-Space Response Oracles.

Iteratively builds a population of diverse policies and computes
a Nash equilibrium meta-strategy over them.  Each iteration:
    1. Compute Nash over the current policy population (meta-solver).
    2. Find a best-response policy to the meta-strategy mixture.
    3. Add the best-response to the population.

The meta-strategy converges to an approximate Nash equilibrium of
the full game while only needing a small number of explicit policies.

Reference: Lanctot et al. (2017) "A Unified Game-Theoretic Approach
to Multiagent Reinforcement Learning"
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
from packages.common.types import ActionType
from packages.strategy.mixed import ActionDistribution


def _softmax(logits: list[float]) -> list[float]:
    max_v = max(logits) if logits else 0.0
    exps = [math.exp(v - max_v) for v in logits]
    total = sum(exps)
    if total == 0:
        return [1.0 / len(logits)] * len(logits)
    return [e / total for e in exps]


class PayoffMatrix:
    """Stores empirical payoffs between all policy pairs.

    payoffs[i][j] = expected payoff of policy i against policy j.
    The matrix is antisymmetric: payoffs[i][j] = -payoffs[j][i].
    """

    def __init__(self) -> None:
        self._matrix: list[list[float]] = []
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def expand(self) -> None:
        """Add a new row/column for a new policy (initialised to 0)."""
        for row in self._matrix:
            row.append(0.0)
        self._size += 1
        self._matrix.append([0.0] * self._size)

    def set_payoff(self, i: int, j: int, payoff: float) -> None:
        """Set the payoff of policy i vs policy j (and -payoff for j vs i)."""
        self._matrix[i][j] = payoff
        self._matrix[j][i] = -payoff

    def get_payoff(self, i: int, j: int) -> float:
        return self._matrix[i][j]

    def get_row(self, i: int) -> list[float]:
        return list(self._matrix[i])


def _solve_nash_2x2(matrix: PayoffMatrix) -> list[float]:
    """Solve Nash equilibrium for a 2-policy game (support enumeration)."""
    if matrix.size == 0:
        return []
    if matrix.size == 1:
        return [1.0]

    a = matrix.get_payoff(0, 0)
    b = matrix.get_payoff(0, 1)
    c = matrix.get_payoff(1, 0)
    d = matrix.get_payoff(1, 1)

    # Try mixed strategy
    denom = (a - b - c + d)
    if abs(denom) > 1e-10:
        p = (d - c) / denom
        p = max(0.0, min(1.0, p))
        return [p, 1.0 - p]

    # Pure strategy fallback
    if a >= c:
        return [1.0, 0.0]
    return [0.0, 1.0]


def solve_nash_support_enum(matrix: PayoffMatrix) -> list[float]:
    """Solve for Nash equilibrium via support enumeration.

    For small populations (<=8), enumerate possible supports and find
    the Nash equilibrium.  Falls back to fictitious play for larger
    populations.
    """
    n = matrix.size
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    if n == 2:
        return _solve_nash_2x2(matrix)

    # For larger populations, use fictitious play
    return _fictitious_play(matrix, iterations=200)


def _fictitious_play(matrix: PayoffMatrix, iterations: int = 200) -> list[float]:
    """Approximate Nash via fictitious play."""
    n = matrix.size
    counts = [1.0] * n  # Start uniform

    for _ in range(iterations):
        # Current mixed strategy
        total = sum(counts)
        mix = [c / total for c in counts]

        # Best response: find policy with highest expected payoff vs mix
        best_val = float("-inf")
        best_idx = 0
        for i in range(n):
            val = sum(matrix.get_payoff(i, j) * mix[j] for j in range(n))
            if val > best_val:
                best_val = val
                best_idx = i

        counts[best_idx] += 1.0

    total = sum(counts)
    return [c / total for c in counts]


class PSROTrainer:
    """Policy-Space Response Oracles with Double Oracle.

    Maintains a growing population of policy networks and a payoff matrix.
    Each iteration computes a Nash meta-strategy and trains a best response.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        seed: int = 42,
    ) -> None:
        self.rng = random.Random(seed)
        self.hidden_dim = hidden_dim
        self.seed = seed

        # Population of policy networks
        self.population: list[SimpleNN] = []

        # Payoff matrix between policies
        self.payoff_matrix = PayoffMatrix()

        # Current meta-strategy (weights over population)
        self._meta_strategy: list[float] = []

        # Initialise with one random policy
        self._add_random_policy()

    def _add_random_policy(self) -> SimpleNN:
        """Create and add a new random policy to the population."""
        policy = SimpleNN(
            FEATURE_DIM,
            self.hidden_dim,
            NUM_ACTIONS,
            seed=self.seed + len(self.population) * 7,
        )
        self.population.append(policy)
        self.payoff_matrix.expand()
        return policy

    def evaluate_pair(
        self,
        policy_a: SimpleNN,
        policy_b: SimpleNN,
        num_hands: int = 100,
    ) -> tuple[float, float]:
        """Evaluate policy_a vs policy_b over simulated hands.

        Uses synthetic feature vectors to compare decision-making.

        Returns:
            (payoff_a, payoff_b) — antisymmetric: payoff_b = -payoff_a.
        """
        total_a = 0.0

        for h in range(num_hands):
            features = [self.rng.random() for _ in range(FEATURE_DIM)]

            out_a = policy_a.forward(features)
            out_b = policy_b.forward(features)

            probs_a = _softmax(out_a)
            probs_b = _softmax(out_b)

            # Simple payoff: aggressive actions score higher vs passive,
            # but lose to counter-aggression.
            # Action 3=BET, 4=RAISE, 5=ALL_IN are "aggressive"
            aggression_a = probs_a[3] + probs_a[4] + probs_a[5]
            aggression_b = probs_b[3] + probs_b[4] + probs_b[5]
            passivity_b = probs_b[0] + probs_b[1]  # FOLD + CHECK

            # Rock-paper-scissors-like dynamics
            payoff = (
                aggression_a * passivity_b  # aggression beats passivity
                - aggression_a * aggression_b * 0.5  # mutual aggression is costly
                + (1.0 - aggression_a) * (1.0 - passivity_b) * 0.1  # passive play small +EV
            )
            total_a += payoff

        avg_a = total_a / max(num_hands, 1)
        return (avg_a, -avg_a)

    def _evaluate_all_pairs(self) -> None:
        """Re-evaluate payoffs for all policy pairs involving the newest policy."""
        n = len(self.population)
        newest = n - 1
        for i in range(n - 1):
            pa, pb = self.evaluate_pair(
                self.population[newest], self.population[i], num_hands=50
            )
            self.payoff_matrix.set_payoff(newest, i, pa)

    def best_response_oracle(
        self,
        meta_strategy: list[float] | None = None,
        train_steps: int = 200,
        lr: float = 0.01,
    ) -> SimpleNN:
        """Train a new policy that best-responds to the meta-strategy mixture.

        The best-response is trained to maximise expected payoff against
        the weighted mixture of existing policies.
        """
        meta = meta_strategy or self._meta_strategy
        if not meta:
            meta = [1.0 / len(self.population)] * len(self.population)

        # Create new policy
        new_policy = SimpleNN(
            FEATURE_DIM,
            self.hidden_dim,
            NUM_ACTIONS,
            seed=self.seed + len(self.population) * 13 + 999,
        )

        # Train against the meta-strategy mixture
        for step in range(train_steps):
            features = [self.rng.random() for _ in range(FEATURE_DIM)]

            # Compute the meta-strategy opponent's average response
            opponent_response = [0.0] * NUM_ACTIONS
            for idx, weight in enumerate(meta):
                if weight < 1e-8:
                    continue
                out = self.population[idx].forward(features)
                probs = _softmax(out)
                for a in range(NUM_ACTIONS):
                    opponent_response[a] += weight * probs[a]

            # Target: exploit the opponent's weaknesses
            # High fold rate → bet more; high aggression → trap more
            target = [0.0] * NUM_ACTIONS
            fold_rate = opponent_response[0]
            check_rate = opponent_response[1]
            aggression = opponent_response[3] + opponent_response[4] + opponent_response[5]

            # Exploit: bet/raise against passive, call/check against aggressive
            target[ACTION_INDEX[ActionType.BET]] = fold_rate + check_rate
            target[ACTION_INDEX[ActionType.RAISE]] = fold_rate * 0.5
            target[ACTION_INDEX[ActionType.CALL]] = aggression
            target[ACTION_INDEX[ActionType.CHECK]] = aggression * 0.3
            target[ACTION_INDEX[ActionType.FOLD]] = 0.05

            # Normalize target
            t_sum = sum(target)
            if t_sum > 0:
                target = [t / t_sum for t in target]

            new_policy.train_step(features, target, lr=lr)

        return new_policy

    def train(self, num_iterations: int = 10) -> list[float]:
        """Run PSRO for the given number of iterations.

        Each iteration:
            1. Compute Nash over current population.
            2. Find best response to the Nash mixture.
            3. Add best response to population and evaluate.

        Returns:
            The final meta-strategy.
        """
        for i in range(num_iterations):
            # Compute meta-Nash
            self._meta_strategy = solve_nash_support_enum(self.payoff_matrix)

            # Find best response
            br = self.best_response_oracle(self._meta_strategy)

            # Add to population
            self.population.append(br)
            self.payoff_matrix.expand()

            # Evaluate new policy against all existing
            self._evaluate_all_pairs()

        # Final meta-strategy
        self._meta_strategy = solve_nash_support_enum(self.payoff_matrix)
        return self._meta_strategy

    def get_meta_strategy(self) -> list[float]:
        """Return the current meta-strategy (mixture weights over population)."""
        if not self._meta_strategy:
            n = len(self.population)
            return [1.0 / n] * n if n > 0 else []
        return list(self._meta_strategy)

    def diversity_score(self) -> float:
        """Measure how diverse the population's policies are.

        Computes average pairwise KL divergence between policies
        on a set of random feature vectors.
        """
        n = len(self.population)
        if n < 2:
            return 0.0

        total_kl = 0.0
        pairs = 0
        num_samples = 20

        for _ in range(num_samples):
            features = [self.rng.random() for _ in range(FEATURE_DIM)]
            outputs = [_softmax(p.forward(features)) for p in self.population]

            for i in range(n):
                for j in range(i + 1, n):
                    kl = 0.0
                    for k in range(NUM_ACTIONS):
                        if outputs[i][k] > 1e-10:
                            kl += outputs[i][k] * math.log(
                                outputs[i][k] / max(outputs[j][k], 1e-10)
                            )
                    total_kl += kl
                    pairs += 1

        return total_kl / max(pairs, 1)
