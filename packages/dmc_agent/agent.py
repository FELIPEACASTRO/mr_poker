"""Deep Monte-Carlo (DMC) Agent — DouZero-style deep RL for poker.

Learns Q(s,a) directly via self-play without CFR or tree search.
Uses epsilon-greedy exploration, experience replay, and a target network
with soft updates for stable training.

Reference: Zha et al. (2021) "DouZero: Mastering DouDiZhu with Self-Play Deep RL"
"""

from __future__ import annotations

import copy
import logging
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from packages.cfr_agent.deep_cfr import (
    ACTION_INDEX,
    FEATURE_DIM,
    NUM_ACTIONS,
    SimpleNN,
    extract_features,
)
from packages.cfr_agent.trainer import _size_action
from packages.common.types import ActionType
from packages.engine.engine import GameEngine, HandRuntime
from packages.strategy.mixed import ActionDistribution

logger = logging.getLogger(__name__)


@dataclass
class Transition:
    """Single experience tuple (s, a, r, s', done)."""

    state: list[float]
    action_idx: int
    reward: float
    next_state: list[float]
    done: bool


class ExperienceReplay:
    """Fixed-capacity replay buffer with uniform sampling."""

    def __init__(self, capacity: int = 10_000) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int, rng: random.Random) -> list[Transition]:
        if len(self.buffer) <= batch_size:
            return list(self.buffer)
        return rng.sample(list(self.buffer), batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


def _copy_nn(source: SimpleNN) -> SimpleNN:
    """Create a deep copy of a SimpleNN."""
    target = SimpleNN(len(source.w1[0]), len(source.w1), len(source.w2), seed=0)
    target.w1 = [row[:] for row in source.w1]
    target.b1 = source.b1[:]
    target.w2 = [row[:] for row in source.w2]
    target.b2 = source.b2[:]
    return target


def _soft_update(target: SimpleNN, source: SimpleNN, tau: float) -> None:
    """Polyak averaging: target = tau * source + (1 - tau) * target."""
    for i in range(len(target.w1)):
        for j in range(len(target.w1[i])):
            target.w1[i][j] = tau * source.w1[i][j] + (1 - tau) * target.w1[i][j]
        target.b1[i] = tau * source.b1[i] + (1 - tau) * target.b1[i]
    for i in range(len(target.w2)):
        for j in range(len(target.w2[i])):
            target.w2[i][j] = tau * source.w2[i][j] + (1 - tau) * target.w2[i][j]
        target.b2[i] = tau * source.b2[i] + (1 - tau) * target.b2[i]


# Reverse lookup: index -> ActionType
INDEX_TO_ACTION = {v: k for k, v in ACTION_INDEX.items()}


class DMCAgent:
    """Deep Monte-Carlo agent using Q-learning from self-play.

    Architecture:
    - Q-network: SimpleNN that maps state features to Q-values per action
    - Target network: slow-moving copy of Q-network for stable TD targets
    - Replay buffer: stores transitions for experience replay
    - Epsilon-greedy: decaying exploration from 1.0 to 0.05
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        capacity: int = 10_000,
        tau: float = 0.01,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        gamma: float = 0.99,
        lr: float = 0.001,
        seed: int = 42,
    ) -> None:
        self.q_network = SimpleNN(FEATURE_DIM, hidden_dim, NUM_ACTIONS, seed=seed)
        self.target_network = _copy_nn(self.q_network)
        self.replay = ExperienceReplay(capacity=capacity)
        self.tau = tau
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.lr = lr
        self.rng = random.Random(seed)
        self.steps = 0

    def select_action(
        self, feat_vec: list[float], legal_actions: set[ActionType]
    ) -> int:
        """Epsilon-greedy action selection. Returns action index."""
        if not legal_actions:
            return ACTION_INDEX[ActionType.FOLD]

        if self.rng.random() < self.epsilon:
            # Explore: random legal action
            action = self.rng.choice(list(legal_actions))
            return ACTION_INDEX.get(action, 0)

        # Exploit: pick legal action with highest Q-value
        q_values = self.q_network.forward(feat_vec)
        best_idx = -1
        best_q = float("-inf")
        for action in legal_actions:
            idx = ACTION_INDEX.get(action)
            if idx is not None and q_values[idx] > best_q:
                best_q = q_values[idx]
                best_idx = idx
        return best_idx if best_idx >= 0 else 0

    def train_step(self, batch_size: int = 32) -> float:
        """Sample from replay and perform one gradient step. Returns mean loss."""
        if len(self.replay) < batch_size:
            return 0.0

        batch = self.replay.sample(batch_size, self.rng)
        total_loss = 0.0

        for transition in batch:
            # Compute TD target: r + gamma * max_a' Q_target(s', a') * (1 - done)
            q_values = self.q_network.forward(transition.state)
            if transition.done:
                td_target = transition.reward
            else:
                next_q = self.target_network.forward(transition.next_state)
                td_target = transition.reward + self.gamma * max(next_q)

            # Build target vector: keep current Q-values, override chosen action
            target = q_values[:]
            target[transition.action_idx] = td_target

            loss = self.q_network.train_step(transition.state, target, lr=self.lr)
            total_loss += loss

        # Soft-update target network
        _soft_update(self.target_network, self.q_network, self.tau)

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.steps += 1

        return total_loss / len(batch)

    def self_play(self, engine: GameEngine, num_episodes: int = 100) -> list[float]:
        """Play against self, collecting experience. Returns per-episode rewards."""
        episode_rewards: list[float] = []

        for ep in range(num_episodes):
            seed = self.rng.randint(0, 2**31)
            runtime = engine.start_new_hand(
                stacks=(100, 100),
                button_seat=ep % 2,
                seed=seed,
            )
            state = runtime.state
            trajectory: list[tuple[list[float], int, int]] = []  # (feat, action_idx, seat)

            while not state.is_terminal and state.acting_seat is not None:
                seat = state.acting_seat
                features = extract_features(state, seat)
                feat_vec = features.to_vector()
                legal = set(engine.legal_actions(runtime))

                action_idx = self.select_action(feat_vec, legal)
                action = INDEX_TO_ACTION.get(action_idx, ActionType.FOLD)
                if action not in legal:
                    action = self.rng.choice(list(legal)) if legal else ActionType.FOLD
                    action_idx = ACTION_INDEX.get(action, 0)

                trajectory.append((feat_vec, action_idx, seat))

                player = state.players[seat]
                amount = _size_action(action, state, player, engine)
                try:
                    engine.apply_action(runtime, action, amount)
                except Exception:
                    break

            # Compute rewards from terminal stacks
            reward_p0 = state.players[0].stack - 100
            reward_p1 = state.players[1].stack - 100

            # Store transitions
            for i, (feat, aidx, seat) in enumerate(trajectory):
                reward = reward_p0 if seat == 0 else reward_p1
                # Next state: next entry in trajectory for same seat, or terminal
                next_feat = [0.0] * FEATURE_DIM
                done = True
                for j in range(i + 1, len(trajectory)):
                    if trajectory[j][2] == seat:
                        next_feat = trajectory[j][0]
                        done = False
                        break
                self.replay.add(Transition(
                    state=feat,
                    action_idx=aidx,
                    reward=reward,
                    next_state=next_feat,
                    done=done,
                ))

            # Train after each episode
            if len(self.replay) >= 32:
                self.train_step(batch_size=32)

            episode_rewards.append(reward_p0)

        return episode_rewards

    def decide(
        self, runtime: HandRuntime, engine: GameEngine
    ) -> tuple[ActionType, int]:
        """Select action for the current acting player."""
        state = runtime.state
        if state.acting_seat is None:
            return ActionType.FOLD, 0

        seat = state.acting_seat
        features = extract_features(state, seat)
        feat_vec = features.to_vector()
        legal = set(engine.legal_actions(runtime))

        # Use greedy (no exploration) for actual play
        q_values = self.q_network.forward(feat_vec)
        best_action = ActionType.FOLD
        best_q = float("-inf")
        for action in legal:
            idx = ACTION_INDEX.get(action)
            if idx is not None and q_values[idx] > best_q:
                best_q = q_values[idx]
                best_action = action

        player = state.players[seat]
        amount = _size_action(best_action, state, player, engine)
        return best_action, amount

    def get_q_values(self, feat_vec: list[float]) -> dict[ActionType, float]:
        """Return Q-values for all actions given features."""
        raw = self.q_network.forward(feat_vec)
        return {INDEX_TO_ACTION[i]: raw[i] for i in range(NUM_ACTIONS)}

    def get_action_distribution(
        self, feat_vec: list[float], legal_actions: set[ActionType], temperature: float = 1.0
    ) -> ActionDistribution:
        """Convert Q-values to a softmax action distribution."""
        q_values = self.q_network.forward(feat_vec)
        legal_q = {}
        for action in legal_actions:
            idx = ACTION_INDEX.get(action)
            if idx is not None:
                legal_q[action] = q_values[idx]

        if not legal_q:
            n = len(legal_actions) or 1
            return ActionDistribution(probabilities={a: 1.0 / n for a in legal_actions})

        # Softmax with temperature
        max_q = max(legal_q.values())
        exp_q = {a: math.exp((v - max_q) / max(temperature, 1e-8)) for a, v in legal_q.items()}
        total = sum(exp_q.values())
        probs = {a: v / total for a, v in exp_q.items()}
        return ActionDistribution(probabilities=probs)
