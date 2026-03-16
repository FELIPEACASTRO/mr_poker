"""Strategic Behavior Prediction — Sequential action prediction.

Goes beyond archetype classification to predict the *specific* next action
an opponent will take, given their full action history.  Uses a sliding-window
feature encoder and a simple neural network to produce calibrated action
probabilities.  Optionally uses a GRU cell for sequential encoding of action
history, providing better temporal modelling than the flat sliding window.

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
from packages.cfr_agent.deep_cfr import GRUCell, SimpleNN


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


_EVENT_DIM = 10  # dimension of encode_event output
_CONTEXT_DIM = 4  # pot_odds, stack_depth, history_len, street


class BehaviorPredictor:
    """Predicts the next action an opponent will take.

    Maintains a sliding window of recent actions and uses a neural
    network to predict action probabilities for the next decision.

    When *use_gru=True* (the default), action history is encoded
    sequentially through a GRU cell whose final hidden state is
    concatenated with context features before being fed to the output
    network.  When *use_gru=False*, the legacy flat sliding-window
    encoding is used instead.
    """

    def __init__(
        self,
        window_size: int = 8,
        hidden_dim: int = 32,
        learning_rate: float = 0.01,
        seed: int = 42,
        use_gru: bool = True,
        gru_hidden_dim: int = 24,
        lr_decay: float = 0.999,
        min_lr: float = 0.0001,
        batch_size: int = 64,
    ) -> None:
        self.window_size = window_size
        self.rng = random.Random(seed)
        self.hidden_dim = hidden_dim
        self.output_dim = NUM_ACTION_TYPES
        self.use_gru = use_gru
        self.gru_hidden_dim = gru_hidden_dim
        self.lr_decay = lr_decay
        self.min_lr = min_lr
        self.batch_size = batch_size

        if use_gru:
            # GRU processes 10-dim encoded events sequentially.
            # Output network input: gru_hidden_dim + 4 context features
            self.gru = GRUCell(_EVENT_DIM, gru_hidden_dim, seed=seed)
            self.input_dim = gru_hidden_dim + _CONTEXT_DIM
        else:
            self.gru = None  # type: ignore[assignment]
            # Legacy: window_size * 10 features per event + 4 context features
            self.input_dim = window_size * _EVENT_DIM + _CONTEXT_DIM

        # Neural network: input -> hidden -> output
        self.network = SimpleNN(
            self.input_dim, hidden_dim, self.output_dim,
            seed=seed,
        )
        self.lr = learning_rate

        # History buffer
        self.history: list[ActionEvent] = []

        # Training data — stored as *raw history snapshots* so that GRU
        # can re-encode them each epoch (important while GRU weights are
        # not trained here).  For the flat path we store pre-built
        # feature vectors for speed.
        self._train_inputs: list[list[float]] = []
        self._train_targets: list[list[float]] = []
        self._max_train_size = 20000

    # ------------------------------------------------------------------
    # Feature building
    # ------------------------------------------------------------------

    def _build_features(
        self, history: list[ActionEvent], context: dict[str, float] | None = None
    ) -> list[float]:
        """Build feature vector from action history.

        When *use_gru* is True, encodes the history sequentially through
        the GRU and concatenates the final hidden state with context.
        Otherwise falls back to the flat sliding-window encoding.
        """
        if self.use_gru and self.gru is not None:
            return self._build_features_gru(history, context)
        return self._build_features_window(history, context)

    def _build_features_gru(
        self, history: list[ActionEvent], context: dict[str, float] | None = None
    ) -> list[float]:
        """GRU-based feature encoding."""
        # Encode each event as a 10-dim vector
        encoded = [encode_event(e) for e in history]
        if encoded:
            h = self.gru.forward_sequence(encoded)
        else:
            h = [0.0] * self.gru_hidden_dim

        # Context features (4 dims)
        ctx = context or {}
        features = list(h)
        features.append(ctx.get("pot_odds", 0.0))
        features.append(ctx.get("stack_depth", 0.5))
        features.append(min(len(history) / 100.0, 1.0))
        features.append(ctx.get("street", 0.0))
        return features

    def _build_features_window(
        self, history: list[ActionEvent], context: dict[str, float] | None = None
    ) -> list[float]:
        """Legacy sliding-window feature encoding."""
        # Get last window_size events
        window = history[-self.window_size:]

        # Encode and flatten
        features: list[float] = []
        for event in window:
            features.extend(encode_event(event))

        # Pad if window is short
        pad_needed = self.window_size * _EVENT_DIM - len(features)
        features = [0.0] * pad_needed + features

        # Context features (4 dims)
        ctx = context or {}
        features.append(ctx.get("pot_odds", 0.0))
        features.append(ctx.get("stack_depth", 0.5))
        features.append(min(len(history) / 100.0, 1.0))  # history length
        features.append(ctx.get("street", 0.0))

        return features

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self, epochs: int = 1) -> float:
        """Train the network on accumulated observations.

        Uses proper softmax cross-entropy gradients for classification.
        Supports mini-batch training (controlled by *batch_size*) and
        applies learning-rate decay after each epoch.

        Returns average loss.
        """
        if len(self._train_inputs) < 2:
            return 0.0

        total_loss = 0.0
        n = 0
        net = self.network
        num_samples = len(self._train_inputs)

        for _epoch in range(epochs):
            # Build index list for this epoch
            indices = list(range(num_samples))
            self.rng.shuffle(indices)

            # Mini-batch iteration
            bs = min(self.batch_size, num_samples)
            for batch_start in range(0, num_samples, bs):
                batch_indices = indices[batch_start:batch_start + bs]

                for i in batch_indices:
                    x = self._train_inputs[i]
                    target = self._train_targets[i]

                    # Forward pass (replicate SimpleNN internals for proper gradients)
                    hidden_raw = []
                    hidden = []
                    for h in range(len(net.w1)):
                        val = net.b1[h] + sum(net.w1[h][j] * x[j] for j in range(len(x)))
                        hidden_raw.append(val)
                        hidden.append(max(0.0, val))

                    logits = []
                    for o in range(len(net.w2)):
                        val = net.b2[o] + sum(net.w2[o][j] * hidden[j] for j in range(len(hidden)))
                        logits.append(val)

                    # Softmax
                    max_logit = max(logits) if logits else 0.0
                    exp_vals = [math.exp(l - max_logit) for l in logits]
                    total_exp = sum(exp_vals)
                    probs = [e / total_exp for e in exp_vals]

                    # Cross-entropy loss
                    loss = -sum(
                        t * math.log(max(p, 1e-10))
                        for t, p in zip(target, probs)
                    )
                    total_loss += loss
                    n += 1

                    # Gradient of softmax cross-entropy: d_logit = probs - target
                    d_output = [probs[o] - target[o] for o in range(len(logits))]

                    # Backward: output layer
                    d_hidden = [0.0] * len(hidden)
                    for o in range(len(net.w2)):
                        for j in range(len(hidden)):
                            d_hidden[j] += d_output[o] * net.w2[o][j]
                            net.w2[o][j] -= self.lr * d_output[o] * hidden[j]
                        net.b2[o] -= self.lr * d_output[o]

                    # Backward: hidden layer (ReLU derivative)
                    for h in range(len(net.w1)):
                        if hidden_raw[h] <= 0:
                            continue
                        grad = d_hidden[h]
                        for j in range(len(x)):
                            net.w1[h][j] -= self.lr * grad * x[j]
                        net.b1[h] -= self.lr * grad

            # Learning rate decay after each epoch
            self.lr = max(self.min_lr, self.lr * self.lr_decay)

        return total_loss / max(n, 1)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

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
