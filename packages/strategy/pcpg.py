"""Parameterized Play-style Conditioned Policy Generation (PCPG).

Maintains a population of agents with diverse play styles, parameterized
by continuous behavioral knobs (aggression, tightness, bluff frequency, etc.).
Uses evolutionary methods to evolve a robust training population.

Reference: Li et al. (2023) "Parameterized Policy Generation"
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from packages.cfr_agent.deep_cfr import ACTION_INDEX, FEATURE_DIM, extract_features
from packages.cfr_agent.trainer import _size_action
from packages.common.types import ActionType
from packages.engine.engine import GameEngine, HandRuntime
from packages.strategy.mixed import ActionDistribution


@dataclass
class PlayStyleParams:
    """Continuous parameters defining an agent's play style.

    Each parameter ranges from 0.0 to 1.0:
    - aggression: tendency to bet/raise vs check/call
    - tightness: hand selection strictness (1.0 = very tight)
    - bluff_freq: frequency of bluffing with weak hands
    - value_bet_freq: frequency of value betting with strong hands
    - fold_to_3bet: tendency to fold when facing a 3-bet
    - cbet_freq: continuation bet frequency
    """

    aggression: float = 0.5
    tightness: float = 0.5
    bluff_freq: float = 0.3
    value_bet_freq: float = 0.7
    fold_to_3bet: float = 0.4
    cbet_freq: float = 0.6

    def to_vector(self) -> list[float]:
        return [
            self.aggression, self.tightness, self.bluff_freq,
            self.value_bet_freq, self.fold_to_3bet, self.cbet_freq,
        ]

    @classmethod
    def from_vector(cls, vec: list[float]) -> PlayStyleParams:
        return cls(
            aggression=max(0.0, min(1.0, vec[0])),
            tightness=max(0.0, min(1.0, vec[1])),
            bluff_freq=max(0.0, min(1.0, vec[2])),
            value_bet_freq=max(0.0, min(1.0, vec[3])),
            fold_to_3bet=max(0.0, min(1.0, vec[4])),
            cbet_freq=max(0.0, min(1.0, vec[5])),
        )

    def distance(self, other: PlayStyleParams) -> float:
        """Euclidean distance between two play styles."""
        v1 = self.to_vector()
        v2 = other.to_vector()
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

    def mutate(self, rng: random.Random, sigma: float = 0.1) -> PlayStyleParams:
        """Return a mutated copy with Gaussian noise."""
        vec = self.to_vector()
        mutated = [max(0.0, min(1.0, v + rng.gauss(0, sigma))) for v in vec]
        return PlayStyleParams.from_vector(mutated)

    def crossover(self, other: PlayStyleParams, rng: random.Random) -> PlayStyleParams:
        """Uniform crossover with another PlayStyleParams."""
        v1 = self.to_vector()
        v2 = other.to_vector()
        child = [v1[i] if rng.random() < 0.5 else v2[i] for i in range(len(v1))]
        return PlayStyleParams.from_vector(child)


NUM_STYLE_PARAMS = 6


def _hand_strength_estimate(state: Any, seat: int) -> float:
    """Quick estimate of hand strength from features (0-1 scale)."""
    features = extract_features(state, seat)
    # Combine card quality signals
    card_quality = (features.high_rank + features.low_rank) / 2.0
    if features.is_pair > 0.5:
        card_quality = min(1.0, card_quality + 0.25)
    if features.is_suited > 0.5:
        card_quality = min(1.0, card_quality + 0.08)
    return card_quality


class ParameterizedAgent:
    """Agent that makes decisions based on PlayStyleParams + hand strength."""

    def __init__(self, params: PlayStyleParams, *, seed: int = 42) -> None:
        self.params = params
        self.rng = random.Random(seed)

    def decide(
        self, runtime: HandRuntime, engine: GameEngine
    ) -> tuple[ActionType, int]:
        """Select action based on play style parameters and hand strength."""
        state = runtime.state
        if state.acting_seat is None:
            return ActionType.FOLD, 0

        seat = state.acting_seat
        legal = set(engine.legal_actions(runtime))
        if not legal:
            return ActionType.FOLD, 0

        dist = self.get_action_distribution(state, seat, legal)
        action = dist.sample(self.rng)
        if action not in legal:
            action = self.rng.choice(list(legal))

        player = state.players[seat]
        amount = _size_action(action, state, player, engine)
        return action, amount

    def get_action_distribution(
        self, state: Any, seat: int, legal: set[ActionType]
    ) -> ActionDistribution:
        """Compute action probabilities from play style + hand strength."""
        strength = _hand_strength_estimate(state, seat)
        p = self.params

        # Base probabilities
        probs: dict[ActionType, float] = {}

        # Strong hand: value bet
        if strength > (1.0 - p.value_bet_freq * 0.5):
            # Value bet region
            if ActionType.BET in legal:
                probs[ActionType.BET] = 0.4 + 0.4 * p.aggression
            if ActionType.RAISE in legal:
                probs[ActionType.RAISE] = 0.4 + 0.4 * p.aggression
            if ActionType.CALL in legal:
                probs[ActionType.CALL] = 0.3
            if ActionType.CHECK in legal:
                probs[ActionType.CHECK] = 0.1
            if ActionType.ALL_IN in legal:
                probs[ActionType.ALL_IN] = 0.1 * p.aggression

        # Weak hand: potential bluff or fold
        elif strength < p.tightness * 0.4:
            # Weak hand region
            if ActionType.FOLD in legal:
                probs[ActionType.FOLD] = 0.5 + 0.3 * (1 - p.bluff_freq)
            if ActionType.CHECK in legal:
                probs[ActionType.CHECK] = 0.3
            # Bluff
            if ActionType.BET in legal:
                probs[ActionType.BET] = 0.2 * p.bluff_freq * p.aggression
            if ActionType.RAISE in legal:
                probs[ActionType.RAISE] = 0.1 * p.bluff_freq * p.aggression
            if ActionType.CALL in legal:
                probs[ActionType.CALL] = 0.1 * (1 - p.tightness)

        # Medium hand: balanced play
        else:
            if ActionType.CHECK in legal:
                probs[ActionType.CHECK] = 0.3 * (1 - p.aggression)
            if ActionType.CALL in legal:
                probs[ActionType.CALL] = 0.4
            if ActionType.BET in legal:
                probs[ActionType.BET] = 0.3 * p.aggression * p.cbet_freq
            if ActionType.RAISE in legal:
                probs[ActionType.RAISE] = 0.2 * p.aggression
            if ActionType.FOLD in legal:
                probs[ActionType.FOLD] = 0.15 * p.tightness

        # Ensure only legal actions and normalize
        probs = {a: max(v, 0.001) for a, v in probs.items() if a in legal}
        # Add any legal actions not yet included with minimal weight
        for a in legal:
            if a not in probs:
                probs[a] = 0.01

        total = sum(probs.values())
        if total > 0:
            probs = {a: v / total for a, v in probs.items()}
        else:
            n = len(legal)
            probs = {a: 1.0 / n for a in legal}

        return ActionDistribution(probabilities=probs)


def _latin_hypercube_sample(n: int, dim: int, rng: random.Random) -> list[list[float]]:
    """Latin Hypercube Sampling for diverse parameter coverage.

    Divides each dimension into n equal intervals and ensures exactly
    one sample per interval per dimension.
    """
    samples: list[list[float]] = []
    # Create permutations for each dimension
    perms = []
    for _d in range(dim):
        perm = list(range(n))
        rng.shuffle(perm)
        perms.append(perm)

    for i in range(n):
        point = []
        for d in range(dim):
            low = perms[d][i] / n
            high = (perms[d][i] + 1) / n
            point.append(rng.uniform(low, high))
        samples.append(point)

    return samples


class PCPGPopulation:
    """Population of parameterized agents with diverse play styles.

    Uses evolutionary methods to maintain and evolve a population
    that covers the strategy space, providing robust training opponents.
    """

    def __init__(self, *, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.agents: list[ParameterizedAgent] = []
        self.fitness_scores: list[float] = []

    def generate_diverse_population(self, size: int = 20) -> list[ParameterizedAgent]:
        """Create a diverse population using Latin Hypercube Sampling."""
        samples = _latin_hypercube_sample(size, NUM_STYLE_PARAMS, self.rng)
        self.agents = []
        self.fitness_scores = [0.0] * size

        for i, sample in enumerate(samples):
            params = PlayStyleParams.from_vector(sample)
            agent = ParameterizedAgent(params, seed=self.rng.randint(0, 2**31))
            self.agents.append(agent)

        return self.agents

    def evolve(self, fitness_scores: list[float]) -> list[ParameterizedAgent]:
        """Evolutionary step: keep top 50%, mutate/crossover the rest.

        Args:
            fitness_scores: fitness value for each agent (higher is better)

        Returns:
            Updated population of agents
        """
        if len(fitness_scores) != len(self.agents):
            raise ValueError(
                f"fitness_scores length {len(fitness_scores)} != population size {len(self.agents)}"
            )

        self.fitness_scores = fitness_scores[:]
        n = len(self.agents)
        half = max(1, n // 2)

        # Sort by fitness (descending)
        ranked = sorted(
            range(n), key=lambda i: fitness_scores[i], reverse=True
        )

        # Keep top half (elite)
        elites = [self.agents[i] for i in ranked[:half]]

        # Generate replacements via mutation and crossover
        new_agents = list(elites)
        while len(new_agents) < n:
            parent1 = self.rng.choice(elites)
            parent2 = self.rng.choice(elites)
            child_params = parent1.params.crossover(parent2.params, self.rng)
            child_params = child_params.mutate(self.rng, sigma=0.1)
            child = ParameterizedAgent(child_params, seed=self.rng.randint(0, 2**31))
            new_agents.append(child)

        self.agents = new_agents[:n]
        self.fitness_scores = [0.0] * n
        return self.agents

    def get_best_response_training_set(self) -> list[ParameterizedAgent]:
        """Return the population as training opponents for best-response computation.

        These diverse opponents ensure robust training that covers
        different play styles rather than overfitting to one opponent type.
        """
        if not self.agents:
            self.generate_diverse_population()
        return list(self.agents)

    def diversity_score(self) -> float:
        """Measure population diversity as mean pairwise distance in parameter space.

        Returns:
            Mean pairwise Euclidean distance. Higher = more diverse.
            Returns 0.0 for populations of size < 2.
        """
        n = len(self.agents)
        if n < 2:
            return 0.0

        total_dist = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += self.agents[i].params.distance(self.agents[j].params)
                count += 1

        return total_dist / count if count > 0 else 0.0

    def get_agent(self, index: int) -> ParameterizedAgent:
        """Get agent at given index."""
        return self.agents[index]

    def best_agent(self) -> ParameterizedAgent:
        """Return the agent with highest fitness score."""
        if not self.agents:
            raise ValueError("empty population")
        best_idx = max(range(len(self.agents)), key=lambda i: self.fitness_scores[i])
        return self.agents[best_idx]
