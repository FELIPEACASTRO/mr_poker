"""Strategic Behavior Prediction — Sequential action prediction.

Goes beyond archetype classification to predict the *specific* next action
an opponent will take, given their full action history.  Uses a sliding-window
feature encoder and a simple neural network to produce calibrated action
probabilities.

This enables more precise exploitation: knowing an opponent will fold 70%
of the time in a spot allows targeted bluffing, rather than relying on
coarse archetype labels.

Reference: Waugh (2022) "Strategic Behavior Prediction in Imperfect-
Information Games", Carnegie Mellon University (PhD Thesis).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from packages.common.types import ActionType
from packages.cfr_agent.deep_cfr import SimpleNN


# Action encoding indices
ACTION_INDICES: dict[ActionType, int] = {
    ActionType.FOLD: 0,
    ActionType.CHECK: 1,
    ActionType.CALL: 2,
    ActionType.BET: 3,
    ActionType.RAISE: 4,
    ActionType.ALL_IN: 5,
}
NUM_ACTION_TYPES = 6

# Street encoding
STREET_MAP: dict[str, float] = {
    "preflop": 0.0,
    "pre_flop": 0.0,
    "flop": 0.33,
    "turn": 0.67,
    "river": 1.0,
}


@dataclass
class ActionEvent:
    """A single observed action in the history."""
    action: ActionType = ActionType.CHECK
    street: str = "preflop"
    bet_fraction: float = 0.0  # bet size as fraction of pot
    position: int = 0  # 0=OOP, 1=IP
    facing_bet: bool = False


def encode_event(event: ActionEvent) -> list[float]:
    """Encode a single action event as a feature vector.

    Output: 10-dim vector:
        [0:6] one-hot action type
        [6] street (normalized)
        [7] bet fraction (capped at 3.0)
        [8] position
        [9] facing bet
    """
    vec = [0.0] * 10
    idx = ACTION_INDICES.get(event.action, 1)
    vec[idx] = 1.0
    vec[6] = STREET_MAP.get(event.street, 0.0)
    vec[7] = min(event.bet_fraction, 3.0) / 3.0
    vec[8] = float(event.position)
    vec[9] = 1.0 if event.facing_bet else 0.0
    return vec


class BehaviorPredictor:
    """Predicts the next action an opponent will take.

    Maintains a sliding window of recent actions and uses a neural
    network to predict action probabilities for the next decision.
    """

    def __init__(
        self,
        window_size: int = 8,
        hidden_dim: int = 32,
        learning_rate: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.window_size = window_size
        self.rng = random.Random(seed)

        # Input: window_size * 10 features per event + 4 context features
        self.input_dim = window_size * 10 + 4
        self.hidden_dim = hidden_dim
        self.output_dim = NUM_ACTION_TYPES

        # Neural network: input -> hidden -> output
        self.network = SimpleNN(
            self.input_dim, hidden_dim, self.output_dim,
            seed=seed,
        )
        self.lr = learning_rate

        # History buffer
        self.history: list[ActionEvent] = []

        # Training data
        self._train_inputs: list[list[float]] = []
        self._train_targets: list[list[float]] = []
        self._max_train_size = 5000

    def observe(self, event: ActionEvent) -> None:
        """Record an observed action."""
        self.history.append(event)

        # Create training example from the previous window -> this action
        if len(self.history) >= 2:
            features = self._build_features(self.history[:-1])
            target = [0.0] * NUM_ACTION_TYPES
            idx = ACTION_INDICES.get(event.action, 1)
            target[idx] = 1.0

            self._train_inputs.append(features)
            self._train_targets.append(target)

            # Cap training buffer
            if len(self._train_inputs) > self._max_train_size:
                self._train_inputs = self._train_inputs[-self._max_train_size:]
                self._train_targets = self._train_targets[-self._max_train_size:]

    def _build_features(
        self, history: list[ActionEvent], context: dict[str, float] | None = None
    ) -> list[float]:
        """Build feature vector from action history.

        Takes the last `window_size` events, pads with zeros if shorter,
        and appends context features.
        """
        # Get last window_size events
        window = history[-self.window_size:]

        # Encode and flatten
        features: list[float] = []
        for event in window:
            features.extend(encode_event(event))

        # Pad if window is short
        pad_needed = self.window_size * 10 - len(features)
        features = [0.0] * pad_needed + features

        # Context features (4 dims)
        ctx = context or {}
        features.append(ctx.get("pot_odds", 0.0))
        features.append(ctx.get("stack_depth", 0.5))
        features.append(min(len(history) / 100.0, 1.0))  # history length
        features.append(ctx.get("street", 0.0))

        return features

    def predict(
        self, context: dict[str, float] | None = None
    ) -> dict[ActionType, float]:
        """Predict the next action probabilities.

        Returns a probability distribution over action types.
        """
        features = self._build_features(self.history, context)
        logits = self.network.forward(features)

        # Softmax
        max_logit = max(logits) if logits else 0.0
        exp_vals = [math.exp(l - max_logit) for l in logits]
        total = sum(exp_vals)

        if total == 0:
            total = 1.0

        probs: dict[ActionType, float] = {}
        for action, idx in ACTION_INDICES.items():
            if idx < len(exp_vals):
                probs[action] = exp_vals[idx] / total

        return probs

    def train_step(self, epochs: int = 1) -> float:
        """Train the network on accumulated observations.

        Returns average loss.
        """
        if len(self._train_inputs) < 2:
            return 0.0

        total_loss = 0.0
        n = 0

        for _ in range(epochs):
            indices = list(range(len(self._train_inputs)))
            self.rng.shuffle(indices)

            for i in indices:
                features = self._train_inputs[i]
                target = self._train_targets[i]

                # Forward pass
                logits = self.network.forward(features)

                # Softmax + cross-entropy loss
                max_logit = max(logits) if logits else 0.0
                exp_vals = [math.exp(l - max_logit) for l in logits]
                total_exp = sum(exp_vals)
                probs = [e / total_exp for e in exp_vals]

                loss = -sum(
                    t * math.log(max(p, 1e-10))
                    for t, p in zip(target, probs)
                )
                total_loss += loss
                n += 1

                # Use SimpleNN.train_step with cross-entropy target
                # Convert softmax probs to MSE-friendly target
                self.network.train_step(features, target, self.lr)

        return total_loss / max(n, 1)

    def predict_specific_action(
        self,
        action: ActionType,
        context: dict[str, float] | None = None,
    ) -> float:
        """Get predicted probability for a specific action."""
        probs = self.predict(context)
        return probs.get(action, 0.0)

    def most_likely_action(
        self, context: dict[str, float] | None = None
    ) -> tuple[ActionType, float]:
        """Return the most likely next action and its probability."""
        probs = self.predict(context)
        if not probs:
            return ActionType.CHECK, 0.0
        best = max(probs, key=probs.get)  # type: ignore[arg-type]
        return best, probs[best]

    def calibration_error(self) -> float:
        """Compute expected calibration error on training data.

        Lower = better calibrated predictions.
        """
        if len(self._train_inputs) < 10:
            return 1.0

        n_bins = 5
        bins: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]

        for features, target in zip(self._train_inputs, self._train_targets):
            logits = self.network.forward(features)
            max_logit = max(logits) if logits else 0.0
            exp_vals = [math.exp(l - max_logit) for l in logits]
            total_exp = sum(exp_vals)
            probs = [e / total_exp for e in exp_vals]

            for j in range(len(probs)):
                p = probs[j]
                t = target[j] if j < len(target) else 0.0
                bin_idx = min(int(p * n_bins), n_bins - 1)
                bins[bin_idx].append((p, t))

        ece = 0.0
        total = sum(len(b) for b in bins)
        for b in bins:
            if b:
                avg_conf = sum(p for p, _ in b) / len(b)
                avg_acc = sum(t for _, t in b) / len(b)
                ece += len(b) / total * abs(avg_conf - avg_acc)

        return ece

    def reset(self) -> None:
        """Reset history and training data."""
        self.history.clear()
        self._train_inputs.clear()
        self._train_targets.clear()

    @property
    def observation_count(self) -> int:
        return len(self.history)
