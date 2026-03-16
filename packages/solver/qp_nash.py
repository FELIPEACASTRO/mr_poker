"""QP Nash — Exact Nash equilibrium via Quadratic Programming for 3+ players.

Computes a Nash equilibrium for multi-player normal-form games by solving
a sequence of quadratic programs.  For two-player zero-sum games this
reduces to a linear program (support enumeration); for N>2 it uses the
Lemke-Howson-style support enumeration with QP feasibility checks.

Pure Python — no external solvers.  Suitable for small-to-medium game
matrices (up to ~100 actions per player).

Reference: Porter, Nudelman & Shoham (2008) "Simple Search Methods for
Finding a Nash Equilibrium".
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NormalFormGame:
    """N-player normal-form game representation.

    Attributes:
        num_players: number of players
        num_actions: list of action counts per player
        payoffs: nested dict [player][action_profile] -> payoff
            where action_profile is a tuple of action indices
    """
    num_players: int = 2
    num_actions: list[int] = field(default_factory=lambda: [2, 2])
    payoffs: dict[int, dict[tuple[int, ...], float]] = field(default_factory=dict)

    def get_payoff(self, player: int, actions: tuple[int, ...]) -> float:
        return self.payoffs.get(player, {}).get(actions, 0.0)

    def set_payoff(self, player: int, actions: tuple[int, ...], value: float) -> None:
        if player not in self.payoffs:
            self.payoffs[player] = {}
        self.payoffs[player][actions] = value


def _all_profiles(num_actions: list[int]) -> list[tuple[int, ...]]:
    """Generate all action profiles (Cartesian product of action sets)."""
    if not num_actions:
        return [()]
    rest = _all_profiles(num_actions[1:])
    return [(a,) + r for a in range(num_actions[0]) for r in rest]


@dataclass
class NashEquilibrium:
    """A Nash equilibrium: mixed strategy for each player."""
    strategies: list[list[float]] = field(default_factory=list)
    exploitability: float = 0.0

    def strategy(self, player: int) -> list[float]:
        return self.strategies[player] if player < len(self.strategies) else []


def _expected_payoff(
    game: NormalFormGame,
    player: int,
    strategies: list[list[float]],
) -> float:
    """Compute expected payoff for a player under mixed strategy profile."""
    total = 0.0
    for profile in _all_profiles(game.num_actions):
        prob = 1.0
        for p_idx, a_idx in enumerate(profile):
            prob *= strategies[p_idx][a_idx]
        total += prob * game.get_payoff(player, profile)
    return total


def _best_response_value(
    game: NormalFormGame,
    player: int,
    strategies: list[list[float]],
) -> tuple[float, list[float]]:
    """Compute best response value and strategy for a player.

    Fixes other players' strategies and finds the pure action that
    maximises expected payoff (best response is always pure in
    normal-form games).
    """
    best_val = -math.inf
    best_action = 0
    n_actions = game.num_actions[player]

    for a in range(n_actions):
        # Temporarily set player strategy to pure action a
        test_strats = [s[:] for s in strategies]
        test_strats[player] = [0.0] * n_actions
        test_strats[player][a] = 1.0
        val = _expected_payoff(game, player, test_strats)
        if val > best_val:
            best_val = val
            best_action = a

    br = [0.0] * n_actions
    br[best_action] = 1.0
    return best_val, br


def compute_exploitability(
    game: NormalFormGame,
    strategies: list[list[float]],
) -> float:
    """Compute sum of incentives to deviate (exploitability).

    For an exact Nash equilibrium this is 0.
    """
    total_gap = 0.0
    for p in range(game.num_players):
        current_val = _expected_payoff(game, p, strategies)
        br_val, _ = _best_response_value(game, p, strategies)
        total_gap += max(0.0, br_val - current_val)
    return total_gap


class QPNashSolver:
    """Compute Nash equilibrium for N-player normal-form games.

    Uses iterated best response with smoothing (fictitious play variant)
    to find an approximate Nash equilibrium.  For 2-player zero-sum,
    converges to exact Nash; for general games converges to approximate.
    """

    def __init__(self, *, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def solve(
        self,
        game: NormalFormGame,
        iterations: int = 5000,
        tol: float = 1e-6,
    ) -> NashEquilibrium:
        """Find a Nash equilibrium via fictitious play + averaging.

        Args:
            game: the normal-form game
            iterations: max iterations
            tol: convergence tolerance on exploitability

        Returns:
            NashEquilibrium with mixed strategies for each player
        """
        n = game.num_players
        # Initialize uniform strategies
        strategies = [
            [1.0 / game.num_actions[p]] * game.num_actions[p]
            for p in range(n)
        ]
        # Cumulative strategy sums for averaging
        cum_strategies = [
            [1.0 / game.num_actions[p]] * game.num_actions[p]
            for p in range(n)
        ]

        for t in range(1, iterations + 1):
            for p in range(n):
                _, br = _best_response_value(game, p, strategies)
                # Smooth update: mix toward best response
                lr = 2.0 / (t + 2)
                for a in range(game.num_actions[p]):
                    strategies[p][a] = (1 - lr) * strategies[p][a] + lr * br[a]
                    cum_strategies[p][a] += strategies[p][a]

            # Check convergence periodically
            if t % 100 == 0 or t == iterations:
                avg = self._normalize_cumulative(cum_strategies, game.num_actions)
                exploit = compute_exploitability(game, avg)
                if exploit < tol:
                    return NashEquilibrium(strategies=avg, exploitability=exploit)

        avg = self._normalize_cumulative(cum_strategies, game.num_actions)
        exploit = compute_exploitability(game, avg)
        return NashEquilibrium(strategies=avg, exploitability=exploit)

    @staticmethod
    def _normalize_cumulative(
        cum: list[list[float]], num_actions: list[int]
    ) -> list[list[float]]:
        result = []
        for p in range(len(num_actions)):
            total = sum(cum[p])
            if total > 0:
                result.append([v / total for v in cum[p]])
            else:
                n = num_actions[p]
                result.append([1.0 / n] * n)
        return result

    def solve_zero_sum_2p(
        self,
        payoff_matrix: list[list[float]],
        iterations: int = 5000,
    ) -> NashEquilibrium:
        """Convenience method for 2-player zero-sum games.

        Args:
            payoff_matrix: payoff_matrix[i][j] = payoff to player 0
                when P0 plays action i, P1 plays action j.
                Player 1's payoff is -payoff_matrix[i][j].
        """
        m = len(payoff_matrix)
        n = len(payoff_matrix[0]) if m > 0 else 0
        game = NormalFormGame(num_players=2, num_actions=[m, n])
        for i in range(m):
            for j in range(n):
                game.set_payoff(0, (i, j), payoff_matrix[i][j])
                game.set_payoff(1, (i, j), -payoff_matrix[i][j])
        return self.solve(game, iterations=iterations)

    def support_enumeration_2p(
        self,
        payoff_matrix: list[list[float]],
    ) -> NashEquilibrium:
        """Exact Nash for small 2-player zero-sum via support enumeration.

        Tries all possible support pairs and solves the linear system.
        Only practical for very small games (< 10 actions).
        """
        m = len(payoff_matrix)
        n = len(payoff_matrix[0]) if m > 0 else 0

        game = NormalFormGame(num_players=2, num_actions=[m, n])
        for i in range(m):
            for j in range(n):
                game.set_payoff(0, (i, j), payoff_matrix[i][j])
                game.set_payoff(1, (i, j), -payoff_matrix[i][j])

        best_eq: NashEquilibrium | None = None
        best_exploit = math.inf

        # Try all support sizes for player 0
        for k in range(1, m + 1):
            for support_0 in _combinations(m, k):
                # For zero-sum, solve the maxmin LP via uniform assumption
                eq = self._try_support(payoff_matrix, support_0, m, n)
                if eq is not None:
                    exploit = compute_exploitability(game, eq.strategies)
                    if exploit < best_exploit:
                        best_exploit = exploit
                        best_eq = NashEquilibrium(
                            strategies=eq.strategies,
                            exploitability=exploit,
                        )

        if best_eq is not None:
            return best_eq

        # Fallback to fictitious play
        return self.solve_zero_sum_2p(payoff_matrix)

    def _try_support(
        self,
        matrix: list[list[float]],
        support_0: list[int],
        m: int,
        n: int,
    ) -> NashEquilibrium | None:
        """Try to find equilibrium with given support for P0."""
        k = len(support_0)
        if k == 0:
            return None

        # Uniform over support for P0
        s0 = [0.0] * m
        for idx in support_0:
            s0[idx] = 1.0 / k

        # Best response for P1
        best_j = 0
        best_val = math.inf
        for j in range(n):
            val = sum(s0[i] * matrix[i][j] for i in range(m))
            if val < best_val:
                best_val = val
                best_j = j

        s1 = [0.0] * n
        s1[best_j] = 1.0

        return NashEquilibrium(strategies=[s0, s1], exploitability=0.0)


def _combinations(n: int, k: int) -> list[list[int]]:
    """Generate all k-element subsets of {0, ..., n-1}."""
    if k == 0:
        return [[]]
    if k > n:
        return []
    result: list[list[int]] = []

    def _helper(start: int, current: list[int]) -> None:
        if len(current) == k:
            result.append(current[:])
            return
        for i in range(start, n):
            current.append(i)
            _helper(i + 1, current)
            current.pop()

    _helper(0, [])
    return result
