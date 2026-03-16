"""PPO Agent — Proximal Policy Optimization for poker.

Uses SimpleNN from deep_cfr for both policy and value networks.
Trains via self-play episodes with PPO-clip objective and GAE.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any

from packages.common.types import ActionType
from packages.engine.engine import GameEngine, HandRuntime
from packages.strategy.mixed import ActionDistribution
from packages.baseline_agent.contracts import AgentDecision
from packages.cfr_agent.deep_cfr import (
    SimpleNN,
    extract_features,
    FEATURE_DIM,
    NUM_ACTIONS,
    ACTION_INDEX,
)
from packages.cfr_agent.trainer import _size_action

logger = logging.getLogger(__name__)

# Reverse mapping: index -> ActionType
INDEX_ACTION = {v: k for k, v in ACTION_INDEX.items()}


@dataclass
class Experience:
    """Single step of experience from self-play."""
    state_features: list[float]
    action_index: int
    reward: float
    value: float
    log_prob: float
    done: bool = False


@dataclass
class ExperienceBuffer:
    """Buffer storing experience tuples for PPO training."""
    experiences: list[Experience] = field(default_factory=list)
    max_size: int = 10_000

    def add(self, exp: Experience) -> None:
        if len(self.experiences) >= self.max_size:
            self.experiences.pop(0)
        self.experiences.append(exp)

    def clear(self) -> None:
        self.experiences.clear()

    def __len__(self) -> int:
        return len(self.experiences)


def _softmax(logits: list[float]) -> list[float]:
    """Numerically stable softmax."""
    max_val = max(logits) if logits else 0.0
    exp_vals = [math.exp(v - max_val) for v in logits]
    total = sum(exp_vals)
    if total == 0:
        return [1.0 / len(logits)] * len(logits)
    return [v / total for v in exp_vals]


def _log_softmax(logits: list[float]) -> list[float]:
    """Numerically stable log-softmax."""
    max_val = max(logits) if logits else 0.0
    shifted = [v - max_val for v in logits]
    log_sum_exp = math.log(sum(math.exp(v) for v in shifted))
    return [v - log_sum_exp for v in shifted]


class PPOAgent:
    """PPO-based poker agent with policy and value networks.

    Uses PPO-clip objective for stable policy updates and GAE
    for low-variance advantage estimation.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        epsilon: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        lr_policy: float = 0.001,
        lr_value: float = 0.001,
        seed: int = 42,
    ) -> None:
        self.epsilon = epsilon
        self.gamma = gamma
        self.lam = lam
        self.lr_policy = lr_policy
        self.lr_value = lr_value
        self.rng = random.Random(seed)

        # Policy network: outputs logits over actions, then softmax
        self.policy_net = SimpleNN(FEATURE_DIM, hidden_dim, NUM_ACTIONS, seed=seed)
        # Value network: outputs single scalar state value
        self.value_net = SimpleNN(FEATURE_DIM, hidden_dim, 1, seed=seed + 1)

        self.buffer = ExperienceBuffer()

        # Training stats
        self.total_episodes = 0
        self.total_train_steps = 0

    def _get_policy(self, features: list[float], legal: set[ActionType]) -> list[float]:
        """Get action probabilities from policy network, masked to legal actions."""
        logits = self.policy_net.forward(features)
        # Mask illegal actions with large negative value
        masked = []
        for i in range(NUM_ACTIONS):
            action = INDEX_ACTION.get(i)
            if action is not None and action in legal:
                masked.append(logits[i])
            else:
                masked.append(-1e9)
        return _softmax(masked)

    def _get_log_probs(self, features: list[float], legal: set[ActionType]) -> list[float]:
        """Get log probabilities from policy network, masked to legal actions."""
        logits = self.policy_net.forward(features)
        masked = []
        for i in range(NUM_ACTIONS):
            action = INDEX_ACTION.get(i)
            if action is not None and action in legal:
                masked.append(logits[i])
            else:
                masked.append(-1e9)
        return _log_softmax(masked)

    def _get_value(self, features: list[float]) -> float:
        """Get state value estimate from value network."""
        return self.value_net.forward(features)[0]

    def _sample_action(
        self, probs: list[float], legal: set[ActionType]
    ) -> tuple[ActionType, int]:
        """Sample an action from the probability distribution."""
        r = self.rng.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                action = INDEX_ACTION.get(i)
                if action is not None and action in legal:
                    return action, i
        # Fallback: pick the highest-probability legal action
        best_i = -1
        best_p = -1.0
        for i, p in enumerate(probs):
            action = INDEX_ACTION.get(i)
            if action is not None and action in legal and p > best_p:
                best_p = p
                best_i = i
        if best_i >= 0:
            return INDEX_ACTION[best_i], best_i
        # Last resort: random legal action
        action = self.rng.choice(list(legal))
        return action, ACTION_INDEX.get(action, 0)

    def compute_gae(
        self,
        rewards: list[float],
        values: list[float],
        dones: list[bool],
    ) -> list[float]:
        """Compute Generalized Advantage Estimation.

        GAE(gamma, lambda) = sum_{l=0}^{inf} (gamma*lambda)^l * delta_{t+l}
        where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        """
        n = len(rewards)
        if n == 0:
            return []

        advantages = [0.0] * n
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1 or dones[t]:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1.0 - float(dones[t])) - values[t]
            last_gae = delta + self.gamma * self.lam * (1.0 - float(dones[t])) * last_gae
            advantages[t] = last_gae

        return advantages

    def train_step(self, epochs: int = 4, batch_size: int = 64) -> float:
        """Run PPO update on buffered experience. Returns average policy loss."""
        if len(self.buffer) < 2:
            return 0.0

        experiences = self.buffer.experiences

        # Extract arrays from buffer
        rewards = [e.reward for e in experiences]
        values = [e.value for e in experiences]
        dones = [e.done for e in experiences]

        # Compute GAE advantages
        advantages = self.compute_gae(rewards, values, dones)

        # Compute returns (advantage + value baseline)
        returns = [adv + val for adv, val in zip(advantages, values)]

        # Normalize advantages
        if len(advantages) > 1:
            mean_adv = sum(advantages) / len(advantages)
            var_adv = sum((a - mean_adv) ** 2 for a in advantages) / len(advantages)
            std_adv = math.sqrt(var_adv + 1e-8)
            advantages = [(a - mean_adv) / std_adv for a in advantages]

        total_policy_loss = 0.0
        num_updates = 0

        for _epoch in range(epochs):
            # Shuffle indices
            indices = list(range(len(experiences)))
            self.rng.shuffle(indices)

            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start:start + batch_size]

                for idx in batch_idx:
                    exp = experiences[idx]
                    adv = advantages[idx]
                    ret = returns[idx]

                    # --- Policy update (PPO-clip) ---
                    # Recompute current log prob
                    logits = self.policy_net.forward(exp.state_features)
                    log_probs = _log_softmax(logits)
                    new_log_prob = log_probs[exp.action_index]
                    old_log_prob = exp.log_prob

                    # Importance ratio
                    ratio = math.exp(new_log_prob - old_log_prob)
                    # Clipped ratio
                    clipped_ratio = max(1.0 - self.epsilon, min(1.0 + self.epsilon, ratio))
                    # PPO objective (we want to maximize, so negate for loss)
                    policy_loss = -min(ratio * adv, clipped_ratio * adv)

                    # Compute policy gradient manually via softmax + cross-entropy
                    probs = _softmax(logits)
                    # Gradient of loss w.r.t. logits: for PPO, approximate with
                    # policy gradient: d_logit[a] = -advantage * (1(a==chosen) - pi(a))
                    # scaled by clip factor
                    clip_scale = 1.0
                    if (ratio > 1.0 + self.epsilon and adv > 0) or \
                       (ratio < 1.0 - self.epsilon and adv < 0):
                        clip_scale = 0.0  # Clipped: no gradient

                    target = list(probs)  # start from current probs
                    for a_idx in range(NUM_ACTIONS):
                        indicator = 1.0 if a_idx == exp.action_index else 0.0
                        grad = -adv * clip_scale * (indicator - probs[a_idx])
                        target[a_idx] = logits[a_idx] - self.lr_policy * grad

                    # Apply the update by using train_step with constructed target
                    self.policy_net.train_step(
                        exp.state_features, target, lr=self.lr_policy
                    )

                    # --- Value update (MSE on returns) ---
                    self.value_net.train_step(
                        exp.state_features, [ret], lr=self.lr_value
                    )

                    total_policy_loss += abs(policy_loss)
                    num_updates += 1

        self.total_train_steps += 1
        avg_loss = total_policy_loss / max(num_updates, 1)

        # Clear buffer after training
        self.buffer.clear()

        return avg_loss

    def self_play_episode(
        self,
        engine: GameEngine,
        *,
        starting_stack: int = 100,
        seat: int = 0,
    ) -> float:
        """Play one hand of self-play and collect experience.

        The PPO agent plays as `seat`, opponent plays randomly.
        Returns the reward (profit/loss in chips) for the PPO agent.
        """
        seed = self.rng.randint(0, 2**31)
        runtime = engine.start_new_hand(
            stacks=(starting_stack, starting_stack),
            button_seat=self.rng.randint(0, 1),
            seed=seed,
        )

        episode_experiences: list[Experience] = []
        state = runtime.state

        while not state.is_terminal and state.acting_seat is not None:
            acting = state.acting_seat
            legal = set(engine.legal_actions(runtime))

            if not legal:
                break

            if acting == seat:
                # PPO agent's turn
                features = extract_features(state, acting)
                feat_vec = features.to_vector()

                probs = self._get_policy(feat_vec, legal)
                log_probs = self._get_log_probs(feat_vec, legal)
                value = self._get_value(feat_vec)

                action, action_idx = self._sample_action(probs, legal)
                log_prob = log_probs[action_idx]

                episode_experiences.append(Experience(
                    state_features=feat_vec,
                    action_index=action_idx,
                    reward=0.0,  # Will be filled at end
                    value=value,
                    log_prob=log_prob,
                    done=False,
                ))
            else:
                # Opponent plays randomly
                action = self.rng.choice(list(legal))

            player = state.players[acting]
            amount = _size_action(action, state, player, engine)

            try:
                engine.apply_action(runtime, action, amount)
            except Exception:
                break

            state = runtime.state

        # Compute final reward
        reward = state.players[seat].stack - starting_stack

        # Assign reward to the last experience, mark as done
        if episode_experiences:
            episode_experiences[-1].reward = float(reward)
            episode_experiences[-1].done = True

            # Add all experiences to buffer
            for exp in episode_experiences:
                self.buffer.add(exp)

        self.total_episodes += 1
        return float(reward)

    def decide(self, runtime: HandRuntime, engine: GameEngine) -> AgentDecision:
        """Make a decision compatible with BaselineAgent interface."""
        state = runtime.state
        acting_seat = state.acting_seat
        if acting_seat is None or state.is_terminal:
            raise ValueError("no action available")

        legal = set(engine.legal_actions(runtime))
        features = extract_features(state, acting_seat)
        feat_vec = features.to_vector()

        probs = self._get_policy(feat_vec, legal)
        action, action_idx = self._sample_action(probs, legal)

        player = state.players[acting_seat]
        amount = _size_action(action, state, player, engine)

        # Build rationale
        probs_str = " ".join(
            f"{INDEX_ACTION[i].value}={p:.0%}"
            for i, p in enumerate(probs)
            if p > 0.01 and i in INDEX_ACTION
        )
        value = self._get_value(feat_vec)

        return AgentDecision(
            action_type=action,
            amount=amount,
            rationale=f"ppo:{probs_str} V={value:.2f}",
        )

    def get_action_distribution(
        self, runtime: HandRuntime, engine: GameEngine
    ) -> ActionDistribution:
        """Return the current policy as an ActionDistribution."""
        state = runtime.state
        acting_seat = state.acting_seat
        if acting_seat is None or state.is_terminal:
            return ActionDistribution(probabilities={})

        legal = set(engine.legal_actions(runtime))
        features = extract_features(state, acting_seat)
        feat_vec = features.to_vector()

        probs = self._get_policy(feat_vec, legal)

        distribution: dict[ActionType, float] = {}
        for i, p in enumerate(probs):
            action = INDEX_ACTION.get(i)
            if action is not None and action in legal and p > 1e-6:
                distribution[action] = p

        return ActionDistribution(probabilities=distribution)
