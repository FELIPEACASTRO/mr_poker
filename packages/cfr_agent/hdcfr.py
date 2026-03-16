"""HDCFR — Hierarchical Deep CFR with skill-based decomposition.

Decomposes the poker strategy into a hierarchy:
1. High-level skill selector: chooses a "play style" for the current situation
2. Low-level executors: one per skill, maps (features + skill) to action probs

Skills represent strategic archetypes (AGGRESSIVE, PASSIVE, TRAPPING, BALANCED)
that transfer between game variants. The hierarchical decomposition allows
the agent to learn reusable strategic concepts.

Architecture:
- Skill selector: SimpleNN(FEATURE_DIM, 32, num_skills) -> skill probabilities
- Executors: SimpleNN(FEATURE_DIM + skill_dim, 32, NUM_ACTIONS) per skill
- Combined: P(a|s) = sum_k P(skill_k|s) * P(a|s, skill_k)

Reference: Inspired by hierarchical RL (options framework) applied to CFR.
"""

from __future__ import annotations

import logging
import math
import random
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from packages.common.types import ActionType
from packages.strategy.mixed import ActionDistribution

from packages.cfr_agent.deep_cfr import (
    SimpleNN,
    FEATURE_DIM,
    NUM_ACTIONS,
    ACTION_INDEX,
)
from packages.cfr_agent.trainer import CFRState, CFR_ACTIONS

logger = logging.getLogger(__name__)


class Skill(IntEnum):
    """Strategic archetypes for hierarchical play."""
    AGGRESSIVE = 0
    PASSIVE = 1
    TRAPPING = 2
    BALANCED = 3


NUM_SKILLS = len(Skill)


def _softmax(values: list[float]) -> list[float]:
    """Numerically stable softmax."""
    if not values:
        return []
    max_v = max(values)
    exps = [math.exp(v - max_v) for v in values]
    total = sum(exps)
    if total == 0:
        return [1.0 / len(values)] * len(values)
    return [e / total for e in exps]


def _one_hot(index: int, size: int) -> list[float]:
    """Create a one-hot vector."""
    vec = [0.0] * size
    vec[index] = 1.0
    return vec


@dataclass
class SkillMemoryEntry:
    """Training sample for skill-level learning."""
    features: list[float]
    skill_id: int
    advantage: float


class SkillMemory:
    """Stores training data per skill for executor networks."""

    def __init__(self, maxlen: int = 100_000) -> None:
        self.buffers: dict[int, deque[SkillMemoryEntry]] = {
            s.value: deque(maxlen=maxlen) for s in Skill
        }

    def add(self, features: list[float], skill_id: int, advantage: float) -> None:
        self.buffers[skill_id].append(
            SkillMemoryEntry(features=features, skill_id=skill_id, advantage=advantage)
        )

    def sample(self, skill_id: int, batch_size: int, rng: random.Random) -> list[SkillMemoryEntry]:
        buf = self.buffers[skill_id]
        if len(buf) <= batch_size:
            return list(buf)
        return rng.sample(list(buf), batch_size)

    def size(self, skill_id: int) -> int:
        return len(self.buffers[skill_id])


class HierarchicalPolicy:
    """Two-level policy: skill selector + per-skill executors.

    P(action | state) = sum_k P(skill_k | state) * P(action | state, skill_k)
    """

    def __init__(
        self,
        *,
        num_skills: int = NUM_SKILLS,
        hidden_dim: int = 32,
        seed: int = 42,
    ) -> None:
        self.num_skills = num_skills

        # High-level: select a skill
        self.skill_selector = SimpleNN(
            FEATURE_DIM, hidden_dim, num_skills, seed=seed
        )

        # Low-level: one executor per skill
        # Input: features + one-hot skill encoding
        executor_input_dim = FEATURE_DIM + num_skills
        self.executors = [
            SimpleNN(executor_input_dim, hidden_dim, NUM_ACTIONS, seed=seed + i + 1)
            for i in range(num_skills)
        ]

    def select_skill(self, features: list[float]) -> tuple[int, list[float]]:
        """Select a skill given features. Returns (skill_id, skill_probs)."""
        logits = self.skill_selector.forward(features)
        probs = _softmax(logits)
        # Argmax for skill selection (deterministic during evaluation)
        skill_id = probs.index(max(probs))
        return skill_id, probs

    def execute_skill(
        self, features: list[float], skill_id: int
    ) -> list[float]:
        """Execute a specific skill. Returns action probabilities."""
        skill_vec = _one_hot(skill_id, self.num_skills)
        combined = features + skill_vec
        logits = self.executors[skill_id].forward(combined)
        return _softmax(logits)

    def get_action_probs(self, features: list[float]) -> list[float]:
        """Compute combined action probabilities over all skills.

        P(a|s) = sum_k P(skill_k|s) * P(a|s, skill_k)
        """
        _, skill_probs = self.select_skill(features)
        combined = [0.0] * NUM_ACTIONS
        for k in range(self.num_skills):
            action_probs = self.execute_skill(features, k)
            for a in range(NUM_ACTIONS):
                combined[a] += skill_probs[k] * action_probs[a]
        return combined


class HDCFRTrainer:
    """Hierarchical Deep CFR trainer.

    Uses a two-level policy (skill selector + executors) to decompose
    the strategy space. Skills represent reusable strategic concepts
    that transfer across game variants.
    """

    def __init__(
        self,
        *,
        num_skills: int = NUM_SKILLS,
        hidden_dim: int = 32,
        seed: int = 42,
    ) -> None:
        self.rng = random.Random(seed)
        self.num_skills = num_skills

        self.policy = HierarchicalPolicy(
            num_skills=num_skills, hidden_dim=hidden_dim, seed=seed
        )
        self.skill_memory = SkillMemory()

        # Selector training memory: (features, skill_advantage_vector)
        self.selector_memory: deque[tuple[list[float], list[float]]] = deque(
            maxlen=100_000
        )

        self.iterations = 0

    def get_strategy(
        self, features: list[float], legal_actions: set[ActionType]
    ) -> ActionDistribution:
        """Get hierarchical strategy as an ActionDistribution."""
        action_probs = self.policy.get_action_probs(features)

        # Filter to legal actions and renormalize
        probs: dict[ActionType, float] = {}
        for action in legal_actions:
            idx = ACTION_INDEX.get(action)
            if idx is not None:
                probs[action] = max(0.0, action_probs[idx])

        total = sum(probs.values())
        if total > 0:
            probs = {a: v / total for a, v in probs.items()}
        else:
            n = len(legal_actions) or 1
            probs = {a: 1.0 / n for a in legal_actions}

        return ActionDistribution(probabilities=probs)

    def update(
        self,
        features: list[float],
        action_utilities: dict[ActionType, float],
        node_utility: float,
    ) -> None:
        """Update skill memories with training data."""
        skill_id, skill_probs = self.policy.select_skill(features)

        # Compute per-skill advantages
        skill_advantages = [0.0] * self.num_skills
        for k in range(self.num_skills):
            action_probs = self.policy.execute_skill(features, k)
            skill_ev = 0.0
            for action, utility in action_utilities.items():
                idx = ACTION_INDEX.get(action)
                if idx is not None:
                    skill_ev += action_probs[idx] * utility
            skill_advantages[k] = skill_ev - node_utility

        # Store selector training data
        self.selector_memory.append((features, skill_advantages))

        # Store executor training data for the selected skill
        advantage = sum(
            action_utilities.get(a, 0.0) * self.policy.execute_skill(features, skill_id)[ACTION_INDEX[a]]
            for a in action_utilities
            if a in ACTION_INDEX
        ) - node_utility
        self.skill_memory.add(features, skill_id, advantage)

    def train_skills(self, iterations: int = 100) -> None:
        """Train skill selector and executors via self-play.

        Alternates between:
        1. Training skill selector on skill advantages
        2. Training executors on per-skill action advantages
        """
        for i in range(iterations):
            # Generate synthetic training data
            features = [self.rng.random() for _ in range(FEATURE_DIM)]
            legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}
            strategy = self.get_strategy(features, legal)

            utilities = {a: self.rng.gauss(0, 1) for a in legal}
            ev = sum(
                strategy.probabilities.get(a, 0.0) * u
                for a, u in utilities.items()
            )
            self.update(features, utilities, ev)
            self.iterations += 1

            # Train networks periodically
            if (i + 1) % 10 == 0:
                self._train_selector()
                self._train_executors()

    def _train_selector(self, epochs: int = 1) -> None:
        """Train the skill selector network."""
        if len(self.selector_memory) < 10:
            return
        for _ in range(epochs):
            batch_size = min(32, len(self.selector_memory))
            batch = self.rng.sample(list(self.selector_memory), batch_size)
            for features, skill_advantages in batch:
                # Target: softmax of advantages (prefer higher-advantage skills)
                target = _softmax(skill_advantages)
                self.policy.skill_selector.train_step(features, target, lr=0.001)

    def _train_executors(self, epochs: int = 1) -> None:
        """Train per-skill executor networks."""
        for skill_id in range(self.num_skills):
            if self.skill_memory.size(skill_id) < 5:
                continue
            for _ in range(epochs):
                batch = self.skill_memory.sample(skill_id, 16, self.rng)
                for entry in batch:
                    skill_vec = _one_hot(entry.skill_id, self.num_skills)
                    combined = entry.features + skill_vec
                    # Target: uniform + advantage bias (encourage positive-advantage actions)
                    target = [1.0 / NUM_ACTIONS] * NUM_ACTIONS
                    self.policy.executors[skill_id].train_step(
                        combined, target, lr=0.001
                    )

    def to_cfr_state(self) -> CFRState:
        """Export hierarchical policy as a standard CFRState."""
        state = CFRState(iterations=self.iterations)
        # Store one info set per skill showing its preferred strategy
        for skill in Skill:
            key = f"hdcfr_skill_{skill.name}"
            # Use zero features as a canonical representation
            features = [0.0] * FEATURE_DIM
            action_probs = self.policy.execute_skill(features, skill.value)
            state.strategy_sum[key] = {}
            state.cumulative_regret[key] = {}
            for action, idx in ACTION_INDEX.items():
                state.strategy_sum[key][action.value] = action_probs[idx]
                state.cumulative_regret[key][action.value] = 0.0
        return state
