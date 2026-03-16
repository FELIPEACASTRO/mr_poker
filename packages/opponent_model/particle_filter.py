"""Particle Filtering for Non-Stationary Opponent Modeling.

Extends Bayesian opponent modeling with Sequential Monte Carlo (particle
filter) to track opponents who change their playing style during a session
(e.g., transitioning from nit -> LAG when tilted).

Each particle represents a hypothesis about the opponent's current style
(archetype + parameters).  Particles are weighted by how well they predict
observed actions, resampled when effective sample size drops, and can
transition between styles via a Markov transition model.

Reference: Bard & Bowling (2007) "Particle Filtering for Dynamic Agent
Modelling in Simplified Poker", University of Alberta, AAAI-07.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from packages.common.types import ActionType
from packages.opponent_model.classifier import (
    ARCHETYPES,
    PlayerStats,
    classify_player,
)


@dataclass
class Particle:
    """A hypothesis about opponent's current playing style.

    Attributes:
        archetype: current style hypothesis (nit, tag, lag, etc.)
        vpip: hypothesized VPIP (voluntary put in pot)
        pfr: hypothesized PFR (preflop raise frequency)
        aggression: hypothesized aggression factor
        weight: particle weight (likelihood of observations)
    """
    archetype: str = "unknown"
    vpip: float = 0.3
    pfr: float = 0.2
    aggression: float = 2.0
    weight: float = 1.0


# Archetype parameter priors (mean vpip, pfr, aggression)
ARCHETYPE_PARAMS: dict[str, tuple[float, float, float]] = {
    "nit": (0.12, 0.10, 1.5),
    "rock": (0.10, 0.08, 1.0),
    "tag": (0.22, 0.18, 2.8),
    "lag": (0.35, 0.25, 3.5),
    "maniac": (0.55, 0.35, 4.5),
    "fish": (0.45, 0.10, 1.2),
    "whale": (0.60, 0.08, 0.8),
    "unknown": (0.30, 0.20, 2.0),
}

# Transition matrix: P(new_type | old_type)
# Most likely to stay the same; some drift possible
TRANSITION_MATRIX: dict[str, dict[str, float]] = {
    "nit": {"nit": 0.80, "rock": 0.05, "tag": 0.10, "lag": 0.03, "maniac": 0.01, "fish": 0.01},
    "rock": {"rock": 0.80, "nit": 0.10, "tag": 0.05, "lag": 0.03, "fish": 0.02},
    "tag": {"tag": 0.75, "lag": 0.10, "nit": 0.08, "maniac": 0.05, "fish": 0.02},
    "lag": {"lag": 0.70, "tag": 0.10, "maniac": 0.12, "fish": 0.05, "nit": 0.03},
    "maniac": {"maniac": 0.65, "lag": 0.20, "tag": 0.05, "fish": 0.08, "nit": 0.02},
    "fish": {"fish": 0.75, "whale": 0.10, "lag": 0.05, "nit": 0.05, "tag": 0.05},
    "whale": {"whale": 0.80, "fish": 0.10, "maniac": 0.05, "lag": 0.05},
    "unknown": {"unknown": 0.40, "tag": 0.15, "lag": 0.10, "nit": 0.10, "fish": 0.10, "maniac": 0.05, "rock": 0.05, "whale": 0.05},
}


def _action_likelihood(
    action: ActionType,
    vpip: float,
    pfr: float,
    aggression: float,
    street: str = "preflop",
) -> float:
    """P(action | opponent_params).

    Simple generative model based on opponent parameters.
    """
    agg_norm = aggression / (1.0 + aggression)  # normalized to [0, 1)

    if street == "preflop":
        if action in (ActionType.RAISE, ActionType.ALL_IN):
            return pfr * 0.8 + agg_norm * 0.2
        if action == ActionType.CALL:
            return max(0.01, vpip - pfr)
        if action == ActionType.FOLD:
            return max(0.01, 1.0 - vpip)
        return 0.1  # CHECK / BET
    else:
        if action in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
            return agg_norm * 0.7 + 0.1
        if action == ActionType.CALL:
            return 0.3 * (1 - agg_norm) + 0.1
        if action == ActionType.FOLD:
            return max(0.01, 0.5 * (1.0 - vpip))
        if action == ActionType.CHECK:
            return max(0.01, 0.4 * (1.0 - agg_norm))
        return 0.1


class ParticleFilterOpponentModel:
    """Particle filter for tracking non-stationary opponents.

    Maintains a population of particles, each representing a hypothesis
    about the opponent's current style.  Updates particles as new actions
    are observed using Sequential Monte Carlo.
    """

    def __init__(
        self,
        num_particles: int = 200,
        resample_threshold: float = 0.5,
        noise_std: float = 0.03,
        seed: int = 42,
    ) -> None:
        self.num_particles = num_particles
        self.resample_threshold = resample_threshold
        self.noise_std = noise_std
        self.rng = random.Random(seed)
        self.particles: list[Particle] = []
        self._init_particles()
        self._observations = 0

    def _init_particles(self) -> None:
        """Initialize particles uniformly across archetypes."""
        self.particles = []
        types = list(ARCHETYPE_PARAMS.keys())
        per_type = max(1, self.num_particles // len(types))

        for archetype in types:
            mean_vpip, mean_pfr, mean_agg = ARCHETYPE_PARAMS[archetype]
            for _ in range(per_type):
                p = Particle(
                    archetype=archetype,
                    vpip=max(0.01, min(0.99, mean_vpip + self.rng.gauss(0, 0.05))),
                    pfr=max(0.01, min(0.99, mean_pfr + self.rng.gauss(0, 0.05))),
                    aggression=max(0.1, mean_agg + self.rng.gauss(0, 0.5)),
                    weight=1.0 / self.num_particles,
                )
                self.particles.append(p)

        # Trim or pad to exact count
        while len(self.particles) > self.num_particles:
            self.particles.pop()
        while len(self.particles) < self.num_particles:
            base = self.rng.choice(self.particles)
            self.particles.append(Particle(
                archetype=base.archetype,
                vpip=base.vpip,
                pfr=base.pfr,
                aggression=base.aggression,
                weight=1.0 / self.num_particles,
            ))

    def update(self, action: ActionType, street: str = "preflop") -> None:
        """Update particle weights given an observed action.

        Steps:
        1. Transition: evolve particles (style drift)
        2. Weight: reweight by action likelihood
        3. Resample: if effective sample size too low
        """
        self._transition()
        self._reweight(action, street)
        self._observations += 1

        ess = self._effective_sample_size()
        if ess < self.resample_threshold * self.num_particles:
            self._resample()

    def _transition(self) -> None:
        """Apply Markov transition to each particle's archetype."""
        for p in self.particles:
            transitions = TRANSITION_MATRIX.get(p.archetype, TRANSITION_MATRIX["unknown"])
            types = list(transitions.keys())
            probs = [transitions[t] for t in types]
            new_type = self.rng.choices(types, weights=probs, k=1)[0]

            if new_type != p.archetype:
                # Transition: shift parameters toward new archetype
                mean_vpip, mean_pfr, mean_agg = ARCHETYPE_PARAMS[new_type]
                blend = 0.3  # partial shift
                p.archetype = new_type
                p.vpip = (1 - blend) * p.vpip + blend * mean_vpip
                p.pfr = (1 - blend) * p.pfr + blend * mean_pfr
                p.aggression = (1 - blend) * p.aggression + blend * mean_agg

            # Add noise (drift)
            p.vpip = max(0.01, min(0.99, p.vpip + self.rng.gauss(0, self.noise_std)))
            p.pfr = max(0.01, min(0.99, p.pfr + self.rng.gauss(0, self.noise_std)))
            p.aggression = max(0.1, p.aggression + self.rng.gauss(0, self.noise_std * 5))

    def _reweight(self, action: ActionType, street: str) -> None:
        """Reweight particles by action likelihood."""
        for p in self.particles:
            likelihood = _action_likelihood(action, p.vpip, p.pfr, p.aggression, street)
            p.weight *= max(likelihood, 1e-10)

        # Normalize
        total = sum(p.weight for p in self.particles)
        if total > 0:
            for p in self.particles:
                p.weight /= total

    def _effective_sample_size(self) -> float:
        """Compute effective sample size (ESS)."""
        sum_sq = sum(p.weight ** 2 for p in self.particles)
        if sum_sq == 0:
            return 0.0
        return 1.0 / sum_sq

    def _resample(self) -> None:
        """Systematic resampling of particles."""
        n = self.num_particles
        weights = [p.weight for p in self.particles]
        cum = []
        s = 0.0
        for w in weights:
            s += w
            cum.append(s)

        new_particles: list[Particle] = []
        u = self.rng.random() / n
        idx = 0

        for _ in range(n):
            while idx < len(cum) - 1 and u > cum[idx]:
                idx += 1
            old = self.particles[idx]
            new_particles.append(Particle(
                archetype=old.archetype,
                vpip=old.vpip,
                pfr=old.pfr,
                aggression=old.aggression,
                weight=1.0 / n,
            ))
            u += 1.0 / n

        self.particles = new_particles

    def estimate_archetype(self) -> dict[str, float]:
        """Estimate current opponent archetype distribution.

        Returns a probability distribution over archetypes,
        weighted by particle weights.
        """
        dist: dict[str, float] = {}
        for p in self.particles:
            dist[p.archetype] = dist.get(p.archetype, 0.0) + p.weight

        # Normalize
        total = sum(dist.values())
        if total > 0:
            dist = {k: v / total for k, v in dist.items()}
        return dist

    def most_likely_archetype(self) -> str:
        """Return the most likely current archetype."""
        dist = self.estimate_archetype()
        if not dist:
            return "unknown"
        return max(dist, key=dist.get)  # type: ignore[arg-type]

    def estimate_params(self) -> tuple[float, float, float]:
        """Estimate weighted-average opponent parameters.

        Returns (vpip, pfr, aggression) weighted by particle weights.
        """
        vpip = sum(p.vpip * p.weight for p in self.particles)
        pfr = sum(p.pfr * p.weight for p in self.particles)
        agg = sum(p.aggression * p.weight for p in self.particles)
        total = sum(p.weight for p in self.particles)
        if total > 0:
            return vpip / total, pfr / total, agg / total
        return 0.3, 0.2, 2.0

    def confidence(self) -> float:
        """Confidence in current estimate (0 to 1).

        Based on ESS and number of observations.
        """
        ess = self._effective_sample_size()
        ess_ratio = ess / self.num_particles  # 1.0 = all particles agree
        obs_factor = min(1.0, self._observations / 30.0)
        return (1.0 - ess_ratio) * obs_factor

    def reset(self) -> None:
        """Reset to uniform prior."""
        self._init_particles()
        self._observations = 0
