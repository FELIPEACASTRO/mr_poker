"""Neural Fictitious Self-Play (NFSP) Agent.

NFSP combines reinforcement learning (best response) with supervised
learning (average strategy) to approximate Nash equilibria in imperfect
information games. The agent maintains two networks:

1. Best Response (BR/RL): A DQN that learns to exploit the current
   opponent mix via Q-learning.
2. Average Strategy (SL): A supervised network trained on the agent's
   own past actions, approximating the average strategy.

The agent plays an eta-greedy mixture: with probability eta it uses
the best response (exploration), and with probability (1-eta) it uses
the average strategy (exploitation/Nash approximation).

Reference: Heinrich & Silver (2016) "Deep Reinforcement Learning
from Self-Play in Imperfect-Information Games", NIPS 2016.
"""

from __future__ import annotations

import math
import random
from collections import deque

from packages.common.types import ActionType
from packages.cfr_agent.deep_cfr import SimpleNN, ACTION_INDEX, NUM_ACTIONS
from packages.strategy.mixed import ActionDistribution


class NFSPAgent:
    """Neural Fictitious Self-Play agent.

    Combines two components:
    1. Best Response (RL): DQN that learns to exploit current opponent
    2. Average Strategy (SL): Supervised learning on own past play

    The agent plays a mixture: eta fraction from BR, (1-eta) from SL.
    """

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 64,
        num_actions: int = 6,
        eta: float = 0.1,
        sl_lr: float = 0.001,
        rl_lr: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.eta = eta
        self.sl_lr = sl_lr
        self.rl_lr = rl_lr
        self.rng = random.Random(seed)

        # Best Response network (DQN): predicts Q-values
        self.br_network = SimpleNN(input_dim, hidden_dim, num_actions, seed=seed)
        # Average Strategy network (SL): predicts action probabilities
        self.sl_network = SimpleNN(input_dim, hidden_dim, num_actions, seed=seed + 1)

        # RL replay memory: (state, action_idx, reward, next_state, done)
        self.rl_memory: deque[tuple[list[float], int, float, list[float], bool]] = deque(maxlen=100000)
        # SL reservoir memory: (state, action_idx)
        self.sl_memory: deque[tuple[list[float], int]] = deque(maxlen=200000)

        # DQN parameters
        self.gamma = 0.99  # discount factor
        self.epsilon = 0.06  # exploration for DQN

    def choose_action(
        self,
        features: list[float],
        legal_actions: set[ActionType],
        rng: random.Random | None = None,
    ) -> ActionType:
        """Choose action using eta-greedy between BR and SL.

        With probability eta, play best response (exploit).
        With probability (1-eta), play average strategy (Nash approx).
        """
        if not legal_actions:
            return ActionType.FOLD

        r = (rng or self.rng).random()

        if r < self.eta:
            # Best response (epsilon-greedy DQN)
            action = self._br_action(features, legal_actions, rng or self.rng)
            # Store in SL memory (only BR actions contribute to average)
            action_idx = ACTION_INDEX.get(action, 0)
            self.sl_memory.append((list(features), action_idx))
        else:
            # Average strategy
            dist = self.get_average_strategy(features, legal_actions)
            action = dist.sample(rng or self.rng)

        if action not in legal_actions:
            action = (rng or self.rng).choice(list(legal_actions))

        return action

    def _br_action(
        self,
        features: list[float],
        legal_actions: set[ActionType],
        rng: random.Random,
    ) -> ActionType:
        """Best response action via epsilon-greedy DQN."""
        if rng.random() < self.epsilon:
            return rng.choice(list(legal_actions))

        q_values = self.br_network.forward(features)

        best_action = None
        best_q = float("-inf")
        for action in legal_actions:
            idx = ACTION_INDEX.get(action)
            if idx is not None and q_values[idx] > best_q:
                best_q = q_values[idx]
                best_action = action

        return best_action if best_action is not None else rng.choice(list(legal_actions))

    def store_rl_transition(
        self,
        state: list[float],
        action: ActionType,
        reward: float,
        next_state: list[float],
        done: bool,
    ) -> None:
        """Store transition for RL training."""
        action_idx = ACTION_INDEX.get(action, 0)
        self.rl_memory.append((list(state), action_idx, reward, list(next_state), done))

    def store_sl_sample(self, state: list[float], action: ActionType) -> None:
        """Store sample for supervised learning."""
        action_idx = ACTION_INDEX.get(action, 0)
        self.sl_memory.append((list(state), action_idx))

    def train_step(self, batch_size: int = 64) -> tuple[float, float]:
        """Train both BR (DQN) and SL networks.

        Returns:
            Tuple of (rl_loss, sl_loss).
        """
        rl_loss = self._train_rl(batch_size)
        sl_loss = self._train_sl(batch_size)
        return rl_loss, sl_loss

    def _train_rl(self, batch_size: int) -> float:
        """Train best response network via DQN (Q-learning)."""
        if len(self.rl_memory) < batch_size:
            return 0.0

        batch = self.rng.sample(list(self.rl_memory), batch_size)
        total_loss = 0.0

        for state, action_idx, reward, next_state, done in batch:
            # Compute target Q-value
            if done:
                target_q = reward
            else:
                next_q = self.br_network.forward(next_state)
                target_q = reward + self.gamma * max(next_q)

            # Current Q-values
            current_q = self.br_network.forward(state)
            target = list(current_q)
            target[action_idx] = target_q

            loss = self.br_network.train_step(state, target, lr=self.rl_lr)
            total_loss += loss

        return total_loss / batch_size

    def _train_sl(self, batch_size: int) -> float:
        """Train average strategy network via supervised learning."""
        if len(self.sl_memory) < batch_size:
            return 0.0

        batch = self.rng.sample(list(self.sl_memory), batch_size)
        total_loss = 0.0

        for state, action_idx in batch:
            # Target is one-hot encoding of the chosen action
            target = [0.0] * self.num_actions
            target[action_idx] = 1.0

            loss = self.sl_network.train_step(state, target, lr=self.sl_lr)
            total_loss += loss

        return total_loss / batch_size

    def get_average_strategy(
        self, features: list[float], legal_actions: set[ActionType]
    ) -> ActionDistribution:
        """Get the average strategy (Nash approximation).

        Uses softmax over the SL network outputs for legal actions.
        """
        raw = self.sl_network.forward(features)

        # Collect values for legal actions
        legal_vals: dict[ActionType, float] = {}
        for action in legal_actions:
            idx = ACTION_INDEX.get(action)
            if idx is not None:
                legal_vals[action] = raw[idx]

        if not legal_vals:
            n = len(legal_actions) or 1
            return ActionDistribution(probabilities={a: 1.0 / n for a in legal_actions})

        # Softmax
        max_val = max(legal_vals.values())
        exp_vals = {a: math.exp(v - max_val) for a, v in legal_vals.items()}
        total = sum(exp_vals.values())

        if total > 0:
            probs = {a: v / total for a, v in exp_vals.items()}
        else:
            n = len(legal_actions) or 1
            probs = {a: 1.0 / n for a in legal_actions}

        return ActionDistribution(probabilities=probs)
