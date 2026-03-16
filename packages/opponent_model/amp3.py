"""AMP3 — Actor-critic with Multi-Player opponent style modeling.

Implements an Actor-Critic agent that conditions its policy and value
estimates on opponent style embeddings. Supports 2-6 players by
processing each opponent's PlayerStats independently through a shared
OpponentStyleNetwork, then aggregating the embeddings.

Pure Python implementation using SimpleNN from deep_cfr.

Reference: He et al. (2016) "Opponent Modeling in Deep RL" (adapted)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from packages.cfr_agent.deep_cfr import SimpleNN
from packages.common.types import ActionType
from packages.opponent_model.classifier import PlayerStats, ARCHETYPES

# Dimensions
GAME_FEATURE_DIM = 15  # matches InfoSetFeatures
STYLE_EMBED_DIM = 8
COMBINED_DIM = GAME_FEATURE_DIM + STYLE_EMBED_DIM  # 23
ACTOR_HIDDEN = 32
CRITIC_HIDDEN = 32
NUM_ACTIONS = 6
STATS_FEATURE_DIM = 12  # features extracted from PlayerStats

# Map actions to indices for output
ACTION_LIST = [
    ActionType.FOLD,
    ActionType.CHECK,
    ActionType.CALL,
    ActionType.BET,
    ActionType.RAISE,
    ActionType.ALL_IN,
]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_LIST)}


def _softmax(values: list[float]) -> list[float]:
    """Numerically stable softmax."""
    m = max(values) if values else 0.0
    exps = [math.exp(v - m) for v in values]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def stats_to_features(stats: PlayerStats) -> list[float]:
    """Convert PlayerStats into a fixed-length feature vector (12-dim)."""
    return [
        stats.vpip,
        stats.pfr,
        stats.three_bet_pct,
        stats.fold_to_3bet_pct,
        stats.cbet_pct,
        stats.fold_to_cbet_pct,
        stats.aggression_factor / 10.0,  # normalize
        stats.wtsd,
        stats.wsd,
        stats.fold_pct,
        min(stats.total_hands / 500.0, 1.0),  # sample size confidence
        stats.overbet_count / max(stats.total_hands, 1),
    ]


class OpponentStyleNetwork:
    """Maps PlayerStats features to an 8-dim style embedding."""

    def __init__(self, *, seed: int = 42) -> None:
        self.net = SimpleNN(STATS_FEATURE_DIM, 16, STYLE_EMBED_DIM, seed=seed)

    def forward(self, stats_features: list[float]) -> list[float]:
        """Produce style embedding from stats features."""
        raw = self.net.forward(stats_features)
        # Normalize to unit-ish scale with tanh approximation
        return [math.tanh(x) for x in raw]

    def train_step(self, x: list[float], target: list[float], lr: float = 0.001) -> float:
        return self.net.train_step(x, target, lr=lr)


class ActorNetwork:
    """Policy network: game_features + style_embedding -> action probabilities."""

    def __init__(self, *, seed: int = 42) -> None:
        self.net = SimpleNN(COMBINED_DIM, ACTOR_HIDDEN, NUM_ACTIONS, seed=seed)

    def forward(self, combined: list[float]) -> list[float]:
        """Return action logits."""
        return self.net.forward(combined)

    def action_probs(self, combined: list[float]) -> list[float]:
        """Return softmax action probabilities."""
        logits = self.forward(combined)
        return _softmax(logits)

    def train_step(self, x: list[float], target: list[float], lr: float = 0.001) -> float:
        return self.net.train_step(x, target, lr=lr)


class CriticNetwork:
    """Value network: game_features + style_embedding -> scalar value."""

    def __init__(self, *, seed: int = 42) -> None:
        self.net = SimpleNN(COMBINED_DIM, CRITIC_HIDDEN, 1, seed=seed)

    def forward(self, combined: list[float]) -> float:
        """Return state value estimate (scalar)."""
        return self.net.forward(combined)[0]

    def train_step(self, x: list[float], target: list[float], lr: float = 0.001) -> float:
        return self.net.train_step(x, target, lr=lr)


class AMP3Agent:
    """Actor-Critic agent conditioned on opponent style embeddings.

    For multi-player (2-6), each opponent's style is computed independently
    and then aggregated (mean) into a single style embedding that is
    concatenated with game features for both actor and critic.
    """

    def __init__(self, *, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.style_net = OpponentStyleNetwork(seed=seed)
        self.actor = ActorNetwork(seed=seed + 100)
        self.critic = CriticNetwork(seed=seed + 200)
        self.gamma = 0.99  # discount factor
        self.entropy_coef = 0.01

    def _aggregate_styles(
        self, opponent_stats_list: list[PlayerStats]
    ) -> list[float]:
        """Compute aggregated style embedding from multiple opponents."""
        if not opponent_stats_list:
            return [0.0] * STYLE_EMBED_DIM

        embeddings: list[list[float]] = []
        for stats in opponent_stats_list:
            features = stats_to_features(stats)
            emb = self.style_net.forward(features)
            embeddings.append(emb)

        # Mean aggregation
        agg = [0.0] * STYLE_EMBED_DIM
        for emb in embeddings:
            for i in range(STYLE_EMBED_DIM):
                agg[i] += emb[i]
        n = len(embeddings)
        return [x / n for x in agg]

    def _combine(
        self, game_features: list[float], style_embedding: list[float]
    ) -> list[float]:
        """Concatenate game features and style embedding."""
        return game_features + style_embedding

    def get_action(
        self,
        game_features: list[float],
        opponent_stats: list[PlayerStats] | PlayerStats,
    ) -> ActionType:
        """Select an action given game state and opponent info.

        Args:
            game_features: 15-dim game state features.
            opponent_stats: single PlayerStats or list for multi-player.

        Returns:
            Selected ActionType.
        """
        if isinstance(opponent_stats, PlayerStats):
            opponent_stats = [opponent_stats]

        style_emb = self._aggregate_styles(opponent_stats)
        combined = self._combine(game_features, style_emb)
        probs = self.actor.action_probs(combined)

        # Sample from distribution
        r = self.rng.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return ACTION_LIST[i]
        return ACTION_LIST[-1]

    def get_action_probs(
        self,
        game_features: list[float],
        opponent_stats: list[PlayerStats] | PlayerStats,
    ) -> dict[ActionType, float]:
        """Return full action probability distribution."""
        if isinstance(opponent_stats, PlayerStats):
            opponent_stats = [opponent_stats]

        style_emb = self._aggregate_styles(opponent_stats)
        combined = self._combine(game_features, style_emb)
        probs = self.actor.action_probs(combined)
        return {ACTION_LIST[i]: probs[i] for i in range(NUM_ACTIONS)}

    def get_value(
        self,
        game_features: list[float],
        opponent_stats: list[PlayerStats] | PlayerStats,
    ) -> float:
        """Estimate state value."""
        if isinstance(opponent_stats, PlayerStats):
            opponent_stats = [opponent_stats]

        style_emb = self._aggregate_styles(opponent_stats)
        combined = self._combine(game_features, style_emb)
        return self.critic.forward(combined)

    def update(
        self,
        game_features: list[float],
        opponent_stats: list[PlayerStats] | PlayerStats,
        action: ActionType,
        reward: float,
        next_game_features: list[float] | None = None,
        next_opponent_stats: list[PlayerStats] | PlayerStats | None = None,
        lr: float = 0.001,
    ) -> dict[str, float]:
        """Single-step actor-critic update.

        Args:
            game_features: current state features (15-dim).
            opponent_stats: opponent stats for current state.
            action: action taken.
            reward: observed reward.
            next_game_features: next state features (None if terminal).
            next_opponent_stats: next state opponent stats.
            lr: learning rate.

        Returns:
            dict with 'actor_loss' and 'critic_loss'.
        """
        if isinstance(opponent_stats, PlayerStats):
            opponent_stats = [opponent_stats]

        style_emb = self._aggregate_styles(opponent_stats)
        combined = self._combine(game_features, style_emb)

        # Current value
        current_value = self.critic.forward(combined)

        # Next value (0 if terminal)
        next_value = 0.0
        if next_game_features is not None:
            if next_opponent_stats is None:
                next_opponent_stats = opponent_stats
            if isinstance(next_opponent_stats, PlayerStats):
                next_opponent_stats = [next_opponent_stats]
            next_style = self._aggregate_styles(next_opponent_stats)
            next_combined = self._combine(next_game_features, next_style)
            next_value = self.critic.forward(next_combined)

        # TD target and advantage
        td_target = reward + self.gamma * next_value
        advantage = td_target - current_value

        # Critic update: minimize (value - td_target)^2
        critic_loss = self.critic.train_step(combined, [td_target], lr=lr)

        # Actor update: policy gradient with advantage
        probs = self.actor.action_probs(combined)
        action_idx = ACTION_TO_IDX.get(action, 0)

        # Construct target that nudges probability toward good actions
        target_probs = list(probs)
        if advantage > 0:
            # Increase probability of this action
            target_probs[action_idx] = min(1.0, probs[action_idx] + 0.1 * advantage)
        else:
            # Decrease probability of this action
            target_probs[action_idx] = max(0.0, probs[action_idx] + 0.1 * advantage)

        # Renormalize
        total = sum(target_probs) or 1.0
        target_probs = [p / total for p in target_probs]

        actor_loss = self.actor.train_step(combined, target_probs, lr=lr)

        return {"actor_loss": actor_loss, "critic_loss": critic_loss}
