"""OpenSkill-inspired Bayesian Rating System for poker players.

Implements Weng-Lin (2011) Bayesian online rating adapted for poker:
- Handles pairwise outcomes (heads-up) and multiplayer rankings
- Accounts for poker variance (high stochasticity)
- Supports partial information (not all hands go to showdown)
- Tracks rating uncertainty (sigma) that shrinks with more data

Reference: arXiv:2401.05451
Reference: Weng & Lin (2011) "A Bayesian Approximation Method for Online Ranking"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class PlayerRating:
    """Bayesian rating for a poker player.

    Represents skill as a Gaussian N(mu, sigma^2).
    The ordinal rating (mu - 3*sigma) is a conservative lower bound
    used for leaderboards.
    """

    mu: float = 25.0          # Mean skill estimate (default: 25)
    sigma: float = 25.0 / 3   # Uncertainty (default: 25/3 ~ 8.33)
    games_played: int = 0
    wins: int = 0
    last_updated: float = 0.0

    @property
    def ordinal(self) -> float:
        """Conservative skill estimate (mu - 3*sigma)."""
        return self.mu - 3 * self.sigma

    @property
    def confidence(self) -> float:
        """Confidence in rating (0-1).

        1.0 when sigma is 0 (perfect knowledge),
        0.0 when sigma >= initial sigma (no knowledge).
        """
        return max(0.0, 1.0 - self.sigma / (25.0 / 3))

    @property
    def skill_label(self) -> str:
        """Human-readable skill label based on ordinal rating."""
        ordinal = self.ordinal
        if ordinal < 0:
            return "fish"
        if ordinal < 10:
            return "recreational"
        if ordinal < 20:
            return "regular"
        if ordinal < 30:
            return "skilled"
        return "expert"


@dataclass
class MatchResult:
    """Result of a heads-up match or session between two players.

    Attributes:
        player_a_id: Identifier for player A.
        player_b_id: Identifier for player B.
        winner_id: ID of the winner, or None for a draw.
        profit_bb: Profit in big blinds (positive = A won, negative = B won).
        hands_played: Number of hands in the session (for confidence weighting).
    """

    player_a_id: str
    player_b_id: str
    winner_id: str | None  # None for draw
    profit_bb: float       # for margin-weighted updates
    hands_played: int      # for confidence weighting


class OpenSkillRating:
    """Bayesian rating system for poker players.

    Key adaptations for poker:
    1. High variance: sigma_dynamics is larger than chess (poker luck factor)
    2. Profit-weighted: big wins count more than small wins
    3. Session-based: update after session, not per-hand (reduces noise)
    4. Ordinal ranking: conservative estimate (mu - 3*sigma) for leaderboards

    Attributes:
        beta: Performance variance (half a sigma). Controls how much
              randomness is expected in outcomes.
        tau: Dynamics factor. How much ratings can drift between games
             (models skill change over time).
        kappa: Draw margin. How close ratings must be to consider a draw likely.
        initial_mu: Default mu for new players.
        initial_sigma: Default sigma for new players.
    """

    def __init__(
        self,
        beta: float = 25.0 / 6,
        tau: float = 25.0 / 300,
        kappa: float = 0.0001,
        initial_mu: float = 25.0,
        initial_sigma: float = 25.0 / 3,
    ) -> None:
        self.beta = beta
        self.tau = tau
        self.kappa = kappa
        self.initial_mu = initial_mu
        self.initial_sigma = initial_sigma
        self._ratings: dict[str, PlayerRating] = {}
        self._history: dict[str, list[dict]] = {}

    # -- Gaussian CDF / PDF helpers ------------------------------------------

    @staticmethod
    def _phi(x: float) -> float:
        """Standard normal CDF (Phi)."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def _pdf(x: float) -> float:
        """Standard normal PDF."""
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    @classmethod
    def _v_function(cls, t: float, epsilon: float) -> float:
        """Truncated Gaussian correction factor v (win case).

        v = pdf(t - epsilon) / Phi(t - epsilon)
        """
        denom = cls._phi(t - epsilon)
        if denom < 1e-15:
            return -t + epsilon
        return cls._pdf(t - epsilon) / denom

    @classmethod
    def _w_function(cls, t: float, epsilon: float) -> float:
        """Truncated Gaussian variance correction w (win case).

        w = v * (v + t - epsilon)
        """
        v = cls._v_function(t, epsilon)
        return v * (v + t - epsilon)

    @classmethod
    def _v_draw(cls, t: float, epsilon: float) -> float:
        """Truncated Gaussian correction v for draw case."""
        denom = cls._phi(epsilon - t) - cls._phi(-epsilon - t)
        if denom < 1e-15:
            return 0.0
        return (cls._pdf(-epsilon - t) - cls._pdf(epsilon - t)) / denom

    @classmethod
    def _w_draw(cls, t: float, epsilon: float) -> float:
        """Truncated Gaussian variance correction w for draw case."""
        denom = cls._phi(epsilon - t) - cls._phi(-epsilon - t)
        if denom < 1e-15:
            return 1.0
        v = cls._v_draw(t, epsilon)
        num = (
            (epsilon - t) * cls._pdf(epsilon - t)
            + (epsilon + t) * cls._pdf(-epsilon - t)
        )
        return v * v + num / denom

    # -- Public API -----------------------------------------------------------

    def get_or_create_rating(self, player_id: str) -> PlayerRating:
        """Get existing rating or create a default one."""
        if player_id not in self._ratings:
            self._ratings[player_id] = PlayerRating(
                mu=self.initial_mu, sigma=self.initial_sigma
            )
            self._history[player_id] = []
        return self._ratings[player_id]

    def rate_match(self, result: MatchResult) -> tuple[PlayerRating, PlayerRating]:
        """Update ratings after a heads-up match.

        Uses Weng-Lin Bayesian update:
            c = sqrt(2*beta^2 + sigma_a^2 + sigma_b^2)
            t = (mu_a - mu_b) / c
            epsilon = kappa / c  (draw margin)

        If A wins:
            mu_a  += sigma_a^2 / c * v(t, epsilon)
            sigma_a^2 *= 1 - sigma_a^2 / c^2 * w(t, epsilon)
        If B wins: symmetric with negated t.
        If draw: use draw-specific v/w functions.

        Profit weighting: scale the update magnitude by a factor derived
        from profit_bb to give larger updates for decisive victories.
        """
        ra = self.get_or_create_rating(result.player_a_id)
        rb = self.get_or_create_rating(result.player_b_id)

        # Apply dynamics (time-based variance increase)
        ra.sigma = math.sqrt(ra.sigma ** 2 + self.tau ** 2)
        rb.sigma = math.sqrt(rb.sigma ** 2 + self.tau ** 2)

        # Combined variance
        c = math.sqrt(2 * self.beta ** 2 + ra.sigma ** 2 + rb.sigma ** 2)

        # Rating difference normalized by combined variance
        t = (ra.mu - rb.mu) / c
        epsilon = self.kappa / c

        # Profit-weight: scale update by how decisive the result was.
        # A 100bb win scales ~ 1.7x, a 1bb win scales ~ 1.0x.
        profit_scale = 1.0 + 0.1 * math.log1p(abs(result.profit_bb))

        # Hands-played confidence: more hands -> more trustworthy result.
        # Asymptotic at 1.0 for many hands; 0.5 for a single hand.
        hands_conf = 1.0 - math.exp(-result.hands_played / 50.0)
        scale = profit_scale * max(0.3, hands_conf)

        is_draw = result.winner_id is None
        a_won = result.winner_id == result.player_a_id

        if is_draw:
            v = self._v_draw(t, epsilon)
            w = self._w_draw(t, epsilon)

            delta_a = ra.sigma ** 2 / c * v * scale
            delta_b = rb.sigma ** 2 / c * (-v) * scale

            new_sigma_a_sq = ra.sigma ** 2 * (1 - ra.sigma ** 2 / c ** 2 * w)
            new_sigma_b_sq = rb.sigma ** 2 * (1 - rb.sigma ** 2 / c ** 2 * w)
        elif a_won:
            v = self._v_function(t, epsilon)
            w = self._w_function(t, epsilon)

            delta_a = ra.sigma ** 2 / c * v * scale
            delta_b = -(rb.sigma ** 2 / c * v * scale)

            new_sigma_a_sq = ra.sigma ** 2 * (1 - ra.sigma ** 2 / c ** 2 * w)
            new_sigma_b_sq = rb.sigma ** 2 * (1 - rb.sigma ** 2 / c ** 2 * w)
        else:
            # B won: negate t
            v = self._v_function(-t, epsilon)
            w = self._w_function(-t, epsilon)

            delta_a = -(ra.sigma ** 2 / c * v * scale)
            delta_b = rb.sigma ** 2 / c * v * scale

            new_sigma_a_sq = ra.sigma ** 2 * (1 - ra.sigma ** 2 / c ** 2 * w)
            new_sigma_b_sq = rb.sigma ** 2 * (1 - rb.sigma ** 2 / c ** 2 * w)

        ra.mu += delta_a
        rb.mu += delta_b
        ra.sigma = math.sqrt(max(1e-6, new_sigma_a_sq))
        rb.sigma = math.sqrt(max(1e-6, new_sigma_b_sq))

        ra.games_played += 1
        rb.games_played += 1
        if a_won:
            ra.wins += 1
        elif not is_draw:
            rb.wins += 1

        ra.last_updated = result.hands_played + ra.last_updated
        rb.last_updated = result.hands_played + rb.last_updated

        # Record history
        self._history.setdefault(result.player_a_id, []).append(
            {"mu": ra.mu, "sigma": ra.sigma, "games_played": ra.games_played}
        )
        self._history.setdefault(result.player_b_id, []).append(
            {"mu": rb.mu, "sigma": rb.sigma, "games_played": rb.games_played}
        )

        return ra, rb

    def rate_tournament(
        self, rankings: list[tuple[str, float]]
    ) -> dict[str, PlayerRating]:
        """Update ratings after a tournament.

        Args:
            rankings: Ordered list of (player_id, profit). First entry is the
                      winner (highest profit). Must have at least 2 entries.

        Decomposes the tournament into pairwise comparisons: each player
        "beats" all players ranked below them. Updates are averaged across
        all pairwise comparisons to avoid over-counting.

        Returns:
            Dict mapping player_id -> updated PlayerRating.
        """
        if len(rankings) < 2:
            if rankings:
                pid = rankings[0][0]
                return {pid: self.get_or_create_rating(pid)}
            return {}

        # Sort by profit descending (index 0 = best)
        sorted_rankings = sorted(rankings, key=lambda x: x[1], reverse=True)

        # Collect all pairwise match results
        n = len(sorted_rankings)
        n_pairs = n * (n - 1) // 2

        # Snapshot ratings before tournament to avoid sequential bias
        snapshots: dict[str, tuple[float, float]] = {}
        for pid, _ in sorted_rankings:
            r = self.get_or_create_rating(pid)
            snapshots[pid] = (r.mu, r.sigma)

        # Accumulate deltas for each player
        mu_deltas: dict[str, float] = {pid: 0.0 for pid, _ in sorted_rankings}
        sigma_sq_factors: dict[str, list[float]] = {
            pid: [] for pid, _ in sorted_rankings
        }

        for i in range(n):
            for j in range(i + 1, n):
                pid_i, profit_i = sorted_rankings[i]
                pid_j, profit_j = sorted_rankings[j]

                ri = self.get_or_create_rating(pid_i)
                rj = self.get_or_create_rating(pid_j)

                # Restore snapshot to avoid sequential bias
                ri.mu, ri.sigma = snapshots[pid_i]
                rj.mu, rj.sigma = snapshots[pid_j]

                # Apply dynamics
                sigma_i = math.sqrt(ri.sigma ** 2 + self.tau ** 2)
                sigma_j = math.sqrt(rj.sigma ** 2 + self.tau ** 2)

                c = math.sqrt(2 * self.beta ** 2 + sigma_i ** 2 + sigma_j ** 2)
                t = (ri.mu - rj.mu) / c
                epsilon = self.kappa / c

                profit_diff = abs(profit_i - profit_j)
                profit_scale = 1.0 + 0.1 * math.log1p(profit_diff)

                if abs(profit_i - profit_j) < 0.01:
                    # Draw
                    v = self._v_draw(t, epsilon)
                    w = self._w_draw(t, epsilon)
                    mu_deltas[pid_i] += sigma_i ** 2 / c * v * profit_scale
                    mu_deltas[pid_j] += sigma_j ** 2 / c * (-v) * profit_scale
                else:
                    # i beat j
                    v = self._v_function(t, epsilon)
                    w = self._w_function(t, epsilon)
                    mu_deltas[pid_i] += sigma_i ** 2 / c * v * profit_scale
                    mu_deltas[pid_j] -= sigma_j ** 2 / c * v * profit_scale

                sigma_sq_factors[pid_i].append(sigma_i ** 2 / c ** 2 * w)
                sigma_sq_factors[pid_j].append(sigma_j ** 2 / c ** 2 * w)

        # Apply averaged deltas
        results: dict[str, PlayerRating] = {}
        for pid, _ in sorted_rankings:
            r = self.get_or_create_rating(pid)
            orig_mu, orig_sigma = snapshots[pid]

            # Average delta across pairs
            avg_mu_delta = mu_deltas[pid] / max(1, n - 1)
            r.mu = orig_mu + avg_mu_delta

            # Sigma update: combine all pairwise variance reductions
            sigma_sq = (orig_sigma ** 2 + self.tau ** 2)
            for factor in sigma_sq_factors[pid]:
                sigma_sq *= (1 - factor / max(1, n - 1))
            r.sigma = math.sqrt(max(1e-6, sigma_sq))

            r.games_played += 1

            # Determine if this player "won" (top rank)
            if pid == sorted_rankings[0][0]:
                r.wins += 1

            self._history.setdefault(pid, []).append(
                {"mu": r.mu, "sigma": r.sigma, "games_played": r.games_played}
            )
            results[pid] = r

        return results

    def expected_win_probability(self, player_a: str, player_b: str) -> float:
        """Predicted probability that player A beats player B.

        P(A > B) = Phi((mu_a - mu_b) / sqrt(2*beta^2 + sigma_a^2 + sigma_b^2))
        """
        ra = self.get_or_create_rating(player_a)
        rb = self.get_or_create_rating(player_b)

        c = math.sqrt(2 * self.beta ** 2 + ra.sigma ** 2 + rb.sigma ** 2)
        return self._phi((ra.mu - rb.mu) / c)

    def leaderboard(self, min_games: int = 10) -> list[tuple[str, PlayerRating]]:
        """Return ranked list of players by ordinal rating.

        Args:
            min_games: Minimum games played to appear on leaderboard.

        Returns:
            List of (player_id, PlayerRating) sorted by ordinal (descending).
        """
        eligible = [
            (pid, rating)
            for pid, rating in self._ratings.items()
            if rating.games_played >= min_games
        ]
        eligible.sort(key=lambda x: x[1].ordinal, reverse=True)
        return eligible

    def rating_history(self, player_id: str) -> list[dict]:
        """Return rating history for a player.

        Each entry is a dict with 'mu', 'sigma', 'games_played'.
        """
        return list(self._history.get(player_id, []))

    def reset(self) -> None:
        """Reset all ratings and history."""
        self._ratings.clear()
        self._history.clear()
