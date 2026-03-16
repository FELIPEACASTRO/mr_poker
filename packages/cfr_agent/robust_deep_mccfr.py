"""Robust Deep MCCFR — Deep MCCFR with target networks and variance-aware training.

Extends Deep CFR with:
- Target networks for stable value estimation (soft Polyak updates)
- Prioritized replay buffer (higher TD-error = more frequent sampling)
- Variance-aware loss weighting to reduce gradient noise

Reference: Brown et al. (2019) + Lillicrap et al. (2016) target network trick.
"""

from __future__ import annotations

import logging
import math
import random
from collections import deque
from dataclasses import dataclass, field

from packages.cfr_agent.deep_cfr import (
    ACTION_INDEX,
    FEATURE_DIM,
    NUM_ACTIONS,
    AdvantageMemory,
    SimpleNN,
)
from packages.common.types import ActionType
from packages.strategy.mixed import ActionDistribution

logger = logging.getLogger(__name__)


class TargetNetwork:
    """Wraps a SimpleNN with Polyak (soft) target updates.

    The target network tracks an online network with exponential moving
    average, providing stable regression targets and reducing oscillation.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, seed: int = 42):
        self.network = SimpleNN(input_dim, hidden_dim, output_dim, seed=seed)

    def forward(self, x: list[float]) -> list[float]:
        return self.network.forward(x)

    def soft_update(self, source: SimpleNN, tau: float = 0.01) -> None:
        """Polyak averaging: target = tau * source + (1 - tau) * target."""
        for i in range(len(self.network.w1)):
            for j in range(len(self.network.w1[i])):
                self.network.w1[i][j] = (
                    tau * source.w1[i][j] + (1.0 - tau) * self.network.w1[i][j]
                )
            self.network.b1[i] = tau * source.b1[i] + (1.0 - tau) * self.network.b1[i]

        for i in range(len(self.network.w2)):
            for j in range(len(self.network.w2[i])):
                self.network.w2[i][j] = (
                    tau * source.w2[i][j] + (1.0 - tau) * self.network.w2[i][j]
                )
            self.network.b2[i] = tau * source.b2[i] + (1.0 - tau) * self.network.b2[i]


@dataclass
class PrioritizedSample:
    """A single sample in the prioritized replay buffer."""
    features: list[float]
    target: list[float]
    priority: float = 1.0


class PrioritizedReplayBuffer:
    """Replay buffer that samples proportional to priority (TD-error based).

    Higher priority samples are drawn more frequently, focusing training
    on the most informative transitions.
    """

    def __init__(self, capacity: int = 500_000, epsilon: float = 1e-4):
        self.capacity = capacity
        self.buffer: deque[PrioritizedSample] = deque(maxlen=capacity)
        self.epsilon = epsilon

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, features: list[float], target: list[float], td_error: float = 1.0) -> None:
        """Add a sample with priority = |td_error| + epsilon."""
        priority = abs(td_error) + self.epsilon
        self.buffer.append(PrioritizedSample(features=features, target=target, priority=priority))

    def sample(self, batch_size: int, rng: random.Random | None = None) -> list[PrioritizedSample]:
        """Sample a batch weighted by priority."""
        rng = rng or random.Random()
        if len(self.buffer) <= batch_size:
            return list(self.buffer)

        priorities = [s.priority for s in self.buffer]
        total = sum(priorities)
        if total <= 0:
            return rng.sample(list(self.buffer), batch_size)

        weights = [p / total for p in priorities]
        # Weighted sampling with replacement
        chosen: list[PrioritizedSample] = []
        cumulative = []
        running = 0.0
        for w in weights:
            running += w
            cumulative.append(running)

        for _ in range(batch_size):
            r = rng.random()
            # Binary search in cumulative weights
            lo, hi = 0, len(cumulative) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if cumulative[mid] < r:
                    lo = mid + 1
                else:
                    hi = mid
            chosen.append(self.buffer[lo])

        return chosen

    def update_priorities(self, indices: list[int], td_errors: list[float]) -> None:
        """Update priorities for previously sampled items."""
        for idx, td_err in zip(indices, td_errors):
            if 0 <= idx < len(self.buffer):
                self.buffer[idx].priority = abs(td_err) + self.epsilon


class RobustDeepMCCFR:
    """Deep MCCFR with target networks and variance-aware training.

    Key improvements over vanilla Deep CFR:
    1. Target networks: smooth advantage targets, reduce oscillation
    2. Prioritized replay: focus on high-error samples
    3. Variance-aware loss: down-weight high-variance gradients
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        tau: float = 0.01,
        lr: float = 0.001,
        variance_beta: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.tau = tau
        self.lr = lr
        self.variance_beta = variance_beta
        self.rng = random.Random(seed)

        # Online advantage networks (one per player)
        self.online_nets = [
            SimpleNN(FEATURE_DIM, hidden_dim, NUM_ACTIONS, seed=seed),
            SimpleNN(FEATURE_DIM, hidden_dim, NUM_ACTIONS, seed=seed + 1),
        ]
        # Target advantage networks (one per player)
        self.target_nets = [
            TargetNetwork(FEATURE_DIM, hidden_dim, NUM_ACTIONS, seed=seed + 10),
            TargetNetwork(FEATURE_DIM, hidden_dim, NUM_ACTIONS, seed=seed + 11),
        ]
        # Prioritized replay buffers (one per player)
        self.replay_buffers = [
            PrioritizedReplayBuffer(),
            PrioritizedReplayBuffer(),
        ]
        # Running variance tracker for loss weighting
        self._loss_ema: float = 1.0
        self._loss_var_ema: float = 1.0
        self.iterations: int = 0

    def add_sample(
        self,
        player: int,
        features: list[float],
        advantages: list[float],
    ) -> None:
        """Add a training sample to the replay buffer with computed priority."""
        # Use target network prediction to compute TD-error
        target_pred = self.target_nets[player].forward(features)
        td_error = sum(
            (advantages[i] - target_pred[i]) ** 2 for i in range(NUM_ACTIONS)
        ) / NUM_ACTIONS
        self.replay_buffers[player].add(features, advantages, td_error=math.sqrt(td_error))

    def train_step(self, player: int, batch_size: int = 64) -> float:
        """One training step on prioritized samples with variance-aware weighting."""
        buf = self.replay_buffers[player]
        if len(buf) < 10:
            return 0.0

        batch = buf.sample(min(batch_size, len(buf)), self.rng)
        total_loss = 0.0

        for sample in batch:
            # Variance-aware weight: down-weight when variance is high
            pred = self.online_nets[player].forward(sample.features)
            sample_loss = sum(
                (pred[i] - sample.target[i]) ** 2 for i in range(len(sample.target))
            ) / len(sample.target)

            # Exponential moving average of loss for variance estimation
            self._loss_ema = 0.99 * self._loss_ema + 0.01 * sample_loss
            diff = sample_loss - self._loss_ema
            self._loss_var_ema = 0.99 * self._loss_var_ema + 0.01 * diff * diff

            # Variance-aware learning rate
            variance_scale = 1.0 / (1.0 + self.variance_beta * math.sqrt(max(self._loss_var_ema, 1e-8)))
            effective_lr = self.lr * variance_scale

            loss = self.online_nets[player].train_step(
                sample.features, sample.target, lr=effective_lr
            )
            total_loss += loss

        # Soft-update target network
        self.target_nets[player].soft_update(self.online_nets[player], self.tau)
        self.iterations += 1

        return total_loss / max(len(batch), 1)

    def get_strategy(
        self,
        player: int,
        features: list[float],
        legal_actions: set[ActionType],
    ) -> ActionDistribution:
        """Get strategy from online advantage network via regret matching."""
        raw = self.online_nets[player].forward(features)
        positive = {}
        for action in legal_actions:
            idx = ACTION_INDEX.get(action)
            if idx is not None:
                positive[action] = max(0.0, raw[idx])

        total = sum(positive.values())
        if total > 0:
            probs = {a: v / total for a, v in positive.items()}
        else:
            n = len(legal_actions) or 1
            probs = {a: 1.0 / n for a in legal_actions}

        return ActionDistribution(probabilities=probs)

    def get_target_strategy(
        self,
        player: int,
        features: list[float],
        legal_actions: set[ActionType],
    ) -> ActionDistribution:
        """Get strategy from the (more stable) target network."""
        raw = self.target_nets[player].forward(features)
        positive = {}
        for action in legal_actions:
            idx = ACTION_INDEX.get(action)
            if idx is not None:
                positive[action] = max(0.0, raw[idx])

        total = sum(positive.values())
        if total > 0:
            probs = {a: v / total for a, v in positive.items()}
        else:
            n = len(legal_actions) or 1
            probs = {a: 1.0 / n for a in legal_actions}

        return ActionDistribution(probabilities=probs)
