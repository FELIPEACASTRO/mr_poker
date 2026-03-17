"""Skill Estimator — CNN-LSTM architecture for opponent skill rating.

Inspired by chess rating estimation via CNN-LSTM + clock times
(arxiv.org/abs/2409.11506): uses a sequence of decision features
(action type, timing, bet sizing, position) to estimate opponent
skill level without manual feature engineering.

Architecture:
    Input sequence: T timesteps × D features per decision
    CNN layer: 1D convolution for local pattern extraction
    LSTM layer: sequential dependency modeling
    Output head: FC → skill rating (0-1 normalized)

The model processes a window of the opponent's recent decisions
and outputs an estimated skill level.

Reference: arxiv.org/abs/2409.11506 (Chess Rating Estimation)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class DecisionFeature:
    """Features for a single opponent decision.

    All values normalized to [0, 1] range.
    """

    action_type: float = 0.0       # 0=fold, 0.2=check, 0.4=call, 0.6=bet, 0.8=raise, 1.0=all_in
    decision_time_norm: float = 0.5  # normalized by player's mean
    bet_fraction: float = 0.0       # bet/pot ratio
    position_score: float = 0.5     # 0=OOP, 1=IP
    street: float = 0.0             # 0=PF, 0.33=F, 0.67=T, 1.0=R
    pot_committed: float = 0.0      # fraction of stack invested
    aggression_context: float = 0.5  # local aggression level
    hand_strength: float = 0.5      # estimated equity (if known)

    def to_vector(self) -> list[float]:
        return [
            self.action_type, self.decision_time_norm, self.bet_fraction,
            self.position_score, self.street, self.pot_committed,
            self.aggression_context, self.hand_strength,
        ]


_FEATURE_DIM = 8


@dataclass
class SkillEstimate:
    """Estimated opponent skill level."""

    skill_rating: float = 0.5      # 0-1 (0=beginner, 1=expert)
    confidence: float = 0.0         # 0-1 confidence in estimate
    skill_label: str = "unknown"    # categorical label

    @staticmethod
    def label_from_rating(rating: float) -> str:
        if rating < 0.2:
            return "fish"
        if rating < 0.4:
            return "recreational"
        if rating < 0.6:
            return "regular"
        if rating < 0.8:
            return "skilled"
        return "expert"


class Conv1DLayer:
    """Simple 1D convolution for local pattern extraction.

    Applies kernel_size filters across the time dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        seed: int = 42,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        rng = random.Random(seed)
        scale = math.sqrt(2.0 / (in_channels * kernel_size))
        self.weights = [
            [[rng.gauss(0, scale) for _ in range(in_channels)]
             for _ in range(kernel_size)]
            for _ in range(out_channels)
        ]
        self.bias = [0.0] * out_channels

    def forward(self, sequence: list[list[float]]) -> list[list[float]]:
        """Apply 1D convolution.

        Args:
            sequence: T × in_channels input.

        Returns:
            (T - kernel_size + 1) × out_channels output.
        """
        t = len(sequence)
        if t < self.kernel_size:
            return [[0.0] * self.out_channels]

        output = []
        for i in range(t - self.kernel_size + 1):
            frame = []
            for oc in range(self.out_channels):
                val = self.bias[oc]
                for k in range(self.kernel_size):
                    for ic in range(self.in_channels):
                        val += self.weights[oc][k][ic] * sequence[i + k][ic]
                frame.append(max(0.0, val))  # ReLU
            output.append(frame)

        return output


class LSTMCell:
    """Simple LSTM cell for sequential processing.

    Implements standard LSTM equations:
        f = sigmoid(Wf @ [h, x] + bf)   # forget gate
        i = sigmoid(Wi @ [h, x] + bi)   # input gate
        o = sigmoid(Wo @ [h, x] + bo)   # output gate
        c_tilde = tanh(Wc @ [h, x] + bc)
        c = f * c_prev + i * c_tilde
        h = o * tanh(c)
    """

    def __init__(self, input_dim: int, hidden_dim: int, seed: int = 42) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        concat_dim = input_dim + hidden_dim

        rng = random.Random(seed)
        scale = math.sqrt(2.0 / concat_dim)

        # Gate weights: f, i, o, c (4 gates)
        self.wf = [[rng.gauss(0, scale) for _ in range(concat_dim)] for _ in range(hidden_dim)]
        self.bf = [0.0] * hidden_dim
        self.wi = [[rng.gauss(0, scale) for _ in range(concat_dim)] for _ in range(hidden_dim)]
        self.bi = [0.0] * hidden_dim
        self.wo = [[rng.gauss(0, scale) for _ in range(concat_dim)] for _ in range(hidden_dim)]
        self.bo = [0.0] * hidden_dim
        self.wc = [[rng.gauss(0, scale) for _ in range(concat_dim)] for _ in range(hidden_dim)]
        self.bc = [0.0] * hidden_dim

    def _sigmoid(self, x: float) -> float:
        if x > 20:
            return 1.0
        if x < -20:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    def _matmul(self, matrix: list[list[float]], vec: list[float]) -> list[float]:
        return [sum(matrix[i][j] * vec[j] for j in range(len(vec))) for i in range(len(matrix))]

    def forward(
        self,
        x: list[float],
        h_prev: list[float],
        c_prev: list[float],
    ) -> tuple[list[float], list[float]]:
        """Single LSTM step. Returns (h_new, c_new)."""
        concat = h_prev + x

        f_gate = [self._sigmoid(v + self.bf[i]) for i, v in enumerate(self._matmul(self.wf, concat))]
        i_gate = [self._sigmoid(v + self.bi[i]) for i, v in enumerate(self._matmul(self.wi, concat))]
        o_gate = [self._sigmoid(v + self.bo[i]) for i, v in enumerate(self._matmul(self.wo, concat))]
        c_tilde = [math.tanh(v + self.bc[i]) for i, v in enumerate(self._matmul(self.wc, concat))]

        c_new = [f_gate[i] * c_prev[i] + i_gate[i] * c_tilde[i] for i in range(self.hidden_dim)]
        h_new = [o_gate[i] * math.tanh(c_new[i]) for i in range(self.hidden_dim)]

        return h_new, c_new

    def forward_sequence(self, sequence: list[list[float]]) -> list[float]:
        """Process a sequence, return final hidden state."""
        h = [0.0] * self.hidden_dim
        c = [0.0] * self.hidden_dim
        for x in sequence:
            h, c = self.forward(x, h, c)
        return h


class SkillEstimatorModel:
    """CNN-LSTM model for opponent skill estimation.

    Processes a window of recent decisions through:
    1. Conv1D for local pattern extraction
    2. LSTM for sequential modeling
    3. FC head for skill rating prediction

    Args:
        window_size: Number of recent decisions to analyze.
        conv_channels: Number of CNN output channels.
        lstm_hidden: LSTM hidden dimension.
        seed: Random seed.
    """

    def __init__(
        self,
        window_size: int = 20,
        conv_channels: int = 16,
        lstm_hidden: int = 16,
        seed: int = 42,
    ) -> None:
        self.window_size = window_size

        self.conv = Conv1DLayer(
            in_channels=_FEATURE_DIM,
            out_channels=conv_channels,
            kernel_size=3,
            seed=seed,
        )
        self.lstm = LSTMCell(
            input_dim=conv_channels,
            hidden_dim=lstm_hidden,
            seed=seed + 1,
        )

        # Output head: lstm_hidden → 1
        rng = random.Random(seed + 2)
        scale = math.sqrt(2.0 / lstm_hidden)
        self.output_w = [rng.gauss(0, scale) for _ in range(lstm_hidden)]
        self.output_b = 0.0

    def predict(self, decisions: list[DecisionFeature]) -> SkillEstimate:
        """Predict skill level from a sequence of decisions.

        Args:
            decisions: Recent decision features (most recent last).

        Returns:
            SkillEstimate with rating and confidence.
        """
        if len(decisions) < 3:
            return SkillEstimate(skill_rating=0.5, confidence=0.0, skill_label="unknown")

        # Truncate to window
        window = decisions[-self.window_size:]

        # Convert to vectors
        sequence = [d.to_vector() for d in window]

        # CNN: extract local patterns
        conv_out = self.conv.forward(sequence)

        # LSTM: sequential processing
        final_hidden = self.lstm.forward_sequence(conv_out)

        # Output head: sigmoid for 0-1 rating
        raw = self.output_b + sum(
            self.output_w[i] * final_hidden[i] for i in range(len(final_hidden))
        )
        rating = 1.0 / (1.0 + math.exp(-raw))

        # Confidence based on number of decisions
        confidence = min(1.0, len(window) / self.window_size)

        label = SkillEstimate.label_from_rating(rating)

        return SkillEstimate(
            skill_rating=rating,
            confidence=confidence,
            skill_label=label,
        )


class BidirectionalLSTM:
    """Bidirectional LSTM that processes sequence in both directions.

    Inspired by Chess Rating from Moves (2409.11506) which showed
    bidirectional processing captures both build-up and resolution
    patterns in decision sequences.

    Output is concatenation of forward and backward final hidden states.
    """

    def __init__(self, input_dim: int, hidden_dim: int, seed: int = 42) -> None:
        self.hidden_dim = hidden_dim
        self.forward_lstm = LSTMCell(input_dim, hidden_dim, seed=seed)
        self.backward_lstm = LSTMCell(input_dim, hidden_dim, seed=seed + 100)

    def forward_sequence(self, sequence: list[list[float]]) -> list[float]:
        """Process sequence bidirectionally, return concatenated hidden states."""
        if not sequence:
            return [0.0] * (self.hidden_dim * 2)

        fwd_hidden = self.forward_lstm.forward_sequence(sequence)
        bwd_hidden = self.backward_lstm.forward_sequence(list(reversed(sequence)))
        return fwd_hidden + bwd_hidden


class BayesianSkillTracker:
    """OpenSkill-inspired Bayesian skill rating using online updates.

    Implements a simplified Plackett-Luce model (OpenSkill, 2401.05451)
    for tracking opponent skill as a posterior distribution.

    The skill estimate is maintained as a Gaussian (mu, sigma) that
    updates with each observed decision quality score.

    Usage::

        tracker = BayesianSkillTracker()
        tracker.update(decision_quality=0.8)  # good decision
        tracker.update(decision_quality=0.2)  # bad decision
        mu, sigma = tracker.estimate()
    """

    def __init__(
        self,
        prior_mu: float = 0.5,
        prior_sigma: float = 0.25,
        dynamics_sigma: float = 0.01,
    ) -> None:
        self.mu = prior_mu
        self.sigma = prior_sigma
        self.dynamics_sigma = dynamics_sigma
        self.n_observations = 0

    def update(self, decision_quality: float) -> None:
        """Update skill estimate with a new decision quality observation.

        decision_quality: 0-1 score (0=terrible, 1=optimal).
        """
        # Add dynamics noise (skill can change over time)
        self.sigma = math.sqrt(self.sigma ** 2 + self.dynamics_sigma ** 2)

        # Bayesian update: treat observation as Gaussian with known variance
        obs_sigma = 0.3  # observation noise
        k = self.sigma ** 2 / (self.sigma ** 2 + obs_sigma ** 2)
        self.mu = self.mu + k * (decision_quality - self.mu)
        self.sigma = math.sqrt((1 - k) * self.sigma ** 2)
        self.n_observations += 1

    def estimate(self) -> tuple[float, float]:
        """Return (mu, sigma) of skill estimate."""
        return (max(0.0, min(1.0, self.mu)), self.sigma)

    @property
    def confidence(self) -> float:
        """Confidence based on posterior precision (1 - normalized sigma)."""
        return max(0.0, min(1.0, 1.0 - self.sigma / 0.25))

    def reset(self) -> None:
        self.mu = 0.5
        self.sigma = 0.25
        self.n_observations = 0


class EnhancedSkillEstimator:
    """Enhanced CNN-BiLSTM + Bayesian tracking skill estimator.

    Combines:
    1. Neural pattern recognition (Conv1D → BiLSTM → FC)
    2. Bayesian online tracking (OpenSkill-style Gaussian updates)

    The neural model captures complex decision patterns while
    the Bayesian tracker provides smooth, uncertainty-aware estimates.

    Args:
        window_size: Decisions for neural model.
        seed: Random seed.
    """

    def __init__(self, window_size: int = 20, seed: int = 42) -> None:
        self.window_size = window_size

        # CNN
        conv_channels = 16
        self.conv = Conv1DLayer(
            in_channels=_FEATURE_DIM,
            out_channels=conv_channels,
            kernel_size=3,
            seed=seed,
        )

        # Bidirectional LSTM
        lstm_hidden = 16
        self.bilstm = BidirectionalLSTM(
            input_dim=conv_channels,
            hidden_dim=lstm_hidden,
            seed=seed + 1,
        )

        # Output head: 2*lstm_hidden → 1
        bilstm_out_dim = lstm_hidden * 2
        rng = random.Random(seed + 2)
        scale = math.sqrt(2.0 / bilstm_out_dim)
        self.output_w = [rng.gauss(0, scale) for _ in range(bilstm_out_dim)]
        self.output_b = 0.0

        # Bayesian tracker
        self.bayesian = BayesianSkillTracker()

        # Decision buffer
        self._decisions: list[DecisionFeature] = []
        self._max_buffer = window_size * 2

    def record_decision(self, decision: DecisionFeature) -> None:
        """Record a decision and update Bayesian tracker."""
        self._decisions.append(decision)
        if len(self._decisions) > self._max_buffer:
            self._decisions = self._decisions[-self._max_buffer:]

        # Simple decision quality heuristic for Bayesian update
        quality = self._decision_quality(decision)
        self.bayesian.update(quality)

    def _decision_quality(self, d: DecisionFeature) -> float:
        """Heuristic quality score from decision features."""
        # Higher action diversity, appropriate timing, position awareness → higher quality
        quality = 0.5
        # Aggressive actions in position = higher quality
        quality += (d.action_type - 0.4) * 0.3 * d.position_score
        # Moderate timing = higher quality (not too fast, not too slow)
        quality -= abs(d.decision_time_norm - 0.5) * 0.2
        # Using pot odds = higher quality
        quality += d.bet_fraction * 0.1
        return max(0.0, min(1.0, quality))

    def estimate(self) -> SkillEstimate:
        """Combined neural + Bayesian skill estimate."""
        if len(self._decisions) < 3:
            return SkillEstimate(skill_rating=0.5, confidence=0.0, skill_label="unknown")

        # Neural estimate
        window = self._decisions[-self.window_size:]
        sequence = [d.to_vector() for d in window]
        conv_out = self.conv.forward(sequence)
        bilstm_hidden = self.bilstm.forward_sequence(conv_out)

        raw = self.output_b + sum(
            self.output_w[i] * bilstm_hidden[i]
            for i in range(len(self.output_w))
        )
        neural_rating = 1.0 / (1.0 + math.exp(-raw))

        # Bayesian estimate
        bayes_mu, bayes_sigma = self.bayesian.estimate()

        # Blend: weight by Bayesian confidence
        bayes_conf = self.bayesian.confidence
        window_conf = min(1.0, len(window) / self.window_size)

        # More Bayesian data → trust it more
        w_bayes = min(0.5, bayes_conf * 0.5)
        w_neural = 1.0 - w_bayes
        rating = w_neural * neural_rating + w_bayes * bayes_mu

        confidence = max(window_conf, bayes_conf)
        label = SkillEstimate.label_from_rating(rating)

        return SkillEstimate(
            skill_rating=rating,
            confidence=confidence,
            skill_label=label,
        )

    @property
    def decisions_recorded(self) -> int:
        return len(self._decisions)

    @property
    def bayesian_estimate(self) -> tuple[float, float]:
        """Access Bayesian (mu, sigma) directly."""
        return self.bayesian.estimate()

    def reset(self) -> None:
        self._decisions.clear()
        self.bayesian.reset()


def _extract_aggregate_features(decisions: list[DecisionFeature]) -> list[float]:
    """Extract 20 aggregate features from a decision sequence.

    These hand-crafted features capture the statistical signature of a
    player's skill level, enabling a simple MLP to achieve high accuracy
    without requiring CNN-BiLSTM backpropagation.

    Features (20-dim):
        [0]  mean action type (aggression level)
        [1]  std action type (action diversity)
        [2]  fraction of aggressive actions (bet/raise/allin)
        [3]  fraction of passive actions (fold/check/call)
        [4]  mean decision timing
        [5]  std decision timing (consistency)
        [6]  mean bet fraction
        [7]  std bet fraction (sizing precision)
        [8]  mean position score
        [9]  position-action correlation (positional awareness)
        [10] mean street reached
        [11] fraction of river decisions
        [12] mean pot committed
        [13] mean aggression context
        [14] mean hand strength
        [15] hand-strength-action correlation (hand reading ability)
        [16] num decisions (normalized)
        [17] timing-action correlation
        [18] bet sizing entropy (sizing diversity)
        [19] street progression (sees later streets)
    """
    if not decisions:
        return [0.0] * 20

    n = len(decisions)
    actions = [d.action_type for d in decisions]
    timings = [d.decision_time_norm for d in decisions]
    sizings = [d.bet_fraction for d in decisions]
    positions = [d.position_score for d in decisions]
    streets = [d.street for d in decisions]
    strengths = [d.hand_strength for d in decisions]

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def _std(xs: list[float]) -> float:
        if len(xs) < 2:
            return 0.0
        m = _mean(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

    def _corr(xs: list[float], ys: list[float]) -> float:
        if len(xs) < 2:
            return 0.0
        mx, my = _mean(xs), _mean(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        sy = math.sqrt(sum((y - my) ** 2 for y in ys))
        if sx == 0 or sy == 0:
            return 0.0
        return max(-1.0, min(1.0, cov / (sx * sy)))

    aggressive = sum(1 for a in actions if a >= 0.6) / n
    passive = sum(1 for a in actions if a < 0.6) / n
    river_frac = sum(1 for s in streets if s >= 0.9) / n

    # Bet sizing entropy (more diverse = less skilled)
    sizing_bins = [0] * 5
    for s in sizings:
        if s <= 0:
            sizing_bins[0] += 1
        elif s < 0.5:
            sizing_bins[1] += 1
        elif s < 1.0:
            sizing_bins[2] += 1
        elif s < 1.5:
            sizing_bins[3] += 1
        else:
            sizing_bins[4] += 1
    entropy = 0.0
    for c in sizing_bins:
        if c > 0:
            p = c / n
            entropy -= p * math.log(p + 1e-10)
    entropy /= math.log(5)  # normalize to [0, 1]

    return [
        _mean(actions),              # 0: aggression level
        _std(actions),               # 1: action diversity
        aggressive,                  # 2: aggressive fraction
        passive,                     # 3: passive fraction
        _mean(timings),              # 4: mean timing
        _std(timings),               # 5: timing consistency
        _mean(sizings),              # 6: mean bet sizing
        _std(sizings),               # 7: sizing precision
        _mean(positions),            # 8: mean position
        _corr(positions, actions),   # 9: positional awareness
        _mean(streets),              # 10: mean street
        river_frac,                  # 11: river frequency
        _mean([d.pot_committed for d in decisions]),  # 12: commitment
        _mean([d.aggression_context for d in decisions]),  # 13: aggression ctx
        _mean(strengths),            # 14: mean hand strength
        _corr(strengths, actions),   # 15: hand reading ability
        min(n / 20.0, 1.0),          # 16: sample size
        _corr(timings, actions),     # 17: timing-action correlation
        entropy,                     # 18: sizing entropy
        max(streets),                # 19: deepest street
    ]


_SKILL_FEATURE_DIM = 20


class SkillTrainer:
    """Supervised trainer using aggregate features + 2-layer MLP.

    Instead of relying on the CNN-BiLSTM (which has random weights),
    this trainer extracts 20 aggregate statistical features from each
    decision sequence and trains a SimpleNN (MLP with full backprop)
    to predict the skill rating.

    This achieves much higher accuracy because:
    1. Features are meaningful (not random CNN-BiLSTM outputs)
    2. Full backprop through both MLP layers (not just FC head)
    3. Proper mini-batch SGD with learning rate decay

    Usage::

        trainer = SkillTrainer(learning_rate=0.01)
        for epoch in range(500):
            loss, acc = trainer.train_epoch(training_data)
    """

    def __init__(
        self,
        estimator: EnhancedSkillEstimator | SkillEstimatorModel | None = None,
        learning_rate: float = 0.01,
        lr_decay: float = 0.999,
        min_lr: float = 0.0001,
        hidden_dim: int = 32,
        seed: int = 42,
    ) -> None:
        self.estimator = estimator
        self.lr = learning_rate
        self.lr_decay = lr_decay
        self.min_lr = min_lr

        # MLP: 20 features → hidden_dim → 1
        rng = random.Random(seed)
        scale_h = math.sqrt(2.0 / _SKILL_FEATURE_DIM)
        scale_o = math.sqrt(2.0 / hidden_dim)

        self.w1 = [[rng.gauss(0, scale_h) for _ in range(_SKILL_FEATURE_DIM)]
                    for _ in range(hidden_dim)]
        self.b1 = [0.0] * hidden_dim
        self.w2 = [rng.gauss(0, scale_o) for _ in range(hidden_dim)]
        self.b2 = 0.0
        self.hidden_dim = hidden_dim

    def _forward(self, features: list[float]) -> tuple[list[float], list[float], float]:
        """Forward pass through 2-layer MLP.

        Returns: (hidden_raw, hidden_activated, predicted_skill)
        """
        # Hidden layer
        hidden_raw = []
        hidden = []
        for h in range(self.hidden_dim):
            val = self.b1[h] + sum(self.w1[h][j] * features[j]
                                    for j in range(len(features)))
            hidden_raw.append(val)
            hidden.append(max(0.0, val))  # ReLU

        # Output: sigmoid
        raw = self.b2 + sum(self.w2[h] * hidden[h] for h in range(self.hidden_dim))
        raw = max(-20.0, min(20.0, raw))
        predicted = 1.0 / (1.0 + math.exp(-raw))

        return hidden_raw, hidden, predicted

    def _backward(self, features: list[float], hidden_raw: list[float],
                  hidden: list[float], predicted: float, target: float) -> float:
        """Full backprop through both layers. Returns MSE loss."""
        loss = (predicted - target) ** 2

        # Output gradient: d_loss/d_raw = 2*(pred-target) * sigmoid'
        d_raw = 2.0 * (predicted - target) * predicted * (1.0 - predicted)

        # Backward: output layer
        d_hidden = [0.0] * self.hidden_dim
        for h in range(self.hidden_dim):
            d_hidden[h] = d_raw * self.w2[h]
            self.w2[h] -= self.lr * d_raw * hidden[h]
        self.b2 -= self.lr * d_raw

        # Backward: hidden layer (ReLU)
        for h in range(self.hidden_dim):
            if hidden_raw[h] <= 0:
                continue  # ReLU killed this neuron
            grad = d_hidden[h]
            for j in range(len(features)):
                self.w1[h][j] -= self.lr * grad * features[j]
            self.b1[h] -= self.lr * grad

        return loss

    def predict(self, decisions: list[DecisionFeature]) -> float:
        """Predict skill rating from a decision sequence."""
        features = _extract_aggregate_features(decisions)
        _, _, predicted = self._forward(features)
        return predicted

    def train_epoch(
        self,
        training_data: list[tuple[list[DecisionFeature], float]],
        shuffle: bool = True,
    ) -> tuple[float, float]:
        """Train one epoch. Returns (avg_loss, classification_accuracy)."""
        if not training_data:
            return 0.0, 0.0

        indices = list(range(len(training_data)))
        if shuffle:
            rng = random.Random()
            rng.shuffle(indices)

        total_loss = 0.0
        correct = 0
        n = 0

        for idx in indices:
            decisions, true_skill = training_data[idx]
            if len(decisions) < 3:
                continue

            features = _extract_aggregate_features(decisions)
            hidden_raw, hidden, predicted = self._forward(features)
            loss = self._backward(features, hidden_raw, hidden, predicted, true_skill)
            total_loss += loss
            n += 1

            if SkillEstimate.label_from_rating(predicted) == SkillEstimate.label_from_rating(true_skill):
                correct += 1

        self.lr = max(self.min_lr, self.lr * self.lr_decay)
        return total_loss / max(n, 1), correct / max(n, 1)

    def evaluate(
        self,
        test_data: list[tuple[list[DecisionFeature], float]],
    ) -> tuple[float, float, float]:
        """Evaluate without updating. Returns (mse, accuracy, mae)."""
        total_loss = 0.0
        total_mae = 0.0
        correct = 0
        n = 0

        for decisions, true_skill in test_data:
            if len(decisions) < 3:
                continue
            features = _extract_aggregate_features(decisions)
            _, _, predicted = self._forward(features)
            total_loss += (predicted - true_skill) ** 2
            total_mae += abs(predicted - true_skill)
            n += 1

            if SkillEstimate.label_from_rating(predicted) == SkillEstimate.label_from_rating(true_skill):
                correct += 1

        if n == 0:
            return 0.0, 0.0, 0.0
        return total_loss / n, correct / n, total_mae / n


def generate_training_data(
    n_samples: int = 5000,
    window_size: int = 20,
    seed: int = 42,
) -> list[tuple[list[DecisionFeature], float]]:
    """Generate labeled (decision_sequence, true_skill) pairs for training.

    Creates diverse player archetypes and simulates realistic decision
    sequences. Higher skill players make more aggressive positional plays,
    use precise bet sizing, and have consistent timing.

    Args:
        n_samples: Number of training examples to generate.
        window_size: Number of decisions per sequence.
        seed: Random seed for reproducibility.

    Returns:
        List of (decisions, skill_label) tuples ready for SkillTrainer.
    """
    rng = random.Random(seed)
    data: list[tuple[list[DecisionFeature], float]] = []

    for _ in range(n_samples):
        # Random skill level (uniform)
        skill = rng.random()

        # Generate decisions characteristic of this skill level
        decisions: list[DecisionFeature] = []
        n_decisions = rng.randint(max(3, window_size // 2), window_size)

        for _ in range(n_decisions):
            # Action selection: skilled players bet/raise more in position
            if rng.random() < skill * 0.5 + 0.15:
                action = rng.choice([0.6, 0.8, 1.0])  # aggressive
            else:
                action = rng.choice([0.0, 0.2, 0.4])  # passive

            # Timing: skilled = consistent moderate timing
            timing_base = 0.4 + skill * 0.15
            timing_noise = rng.gauss(0, 0.2 * (1.0 - skill * 0.6))
            timing = max(0.0, min(1.0, timing_base + timing_noise))

            # Bet sizing: skilled = tighter around 0.5-0.75 pot
            if action >= 0.6:
                sizing_base = 0.5 + skill * 0.15
                sizing_noise = rng.gauss(0, 0.15 * (1.0 - skill * 0.5))
                sizing = max(0.1, min(2.0, sizing_base + sizing_noise))
            else:
                sizing = 0.0

            # Position awareness: skilled players act differently IP vs OOP
            position = rng.random()

            # Street distribution: skilled players see more streets
            street = rng.choice([0.0, 0.33, 0.67, 1.0])
            if skill < 0.3 and rng.random() < 0.4:
                street = 0.0  # weak players fold preflop more

            # Pot committed
            pot_committed = rng.random() * (0.3 + skill * 0.2)

            # Aggression context
            aggression = skill * 0.4 + rng.gauss(0, 0.1)
            aggression = max(0.0, min(1.0, aggression))

            # Hand strength awareness: skilled players correlate actions with strength
            hand_str = rng.random()
            if skill > 0.6 and hand_str > 0.7 and action < 0.6:
                # Skilled players with strong hands bet more
                action = rng.choice([0.6, 0.8]) if rng.random() < 0.7 else action

            decisions.append(DecisionFeature(
                action_type=action,
                decision_time_norm=timing,
                bet_fraction=sizing,
                position_score=position,
                street=street,
                pot_committed=pot_committed,
                aggression_context=aggression,
                hand_strength=hand_str,
            ))

        data.append((decisions, skill))

    return data


class OnlineSkillEstimator:
    """Wrapper that accumulates decisions and estimates skill online.

    Usage::

        estimator = OnlineSkillEstimator()
        estimator.record_decision(DecisionFeature(action_type=0.8, ...))
        estimator.record_decision(DecisionFeature(action_type=0.4, ...))
        estimate = estimator.estimate()
    """

    def __init__(self, window_size: int = 20, seed: int = 42) -> None:
        self.model = SkillEstimatorModel(window_size=window_size, seed=seed)
        self._decisions: list[DecisionFeature] = []
        self._max_buffer = window_size * 2

    def record_decision(self, decision: DecisionFeature) -> None:
        """Record a new decision."""
        self._decisions.append(decision)
        if len(self._decisions) > self._max_buffer:
            self._decisions = self._decisions[-self._max_buffer:]

    def estimate(self) -> SkillEstimate:
        """Get current skill estimate."""
        return self.model.predict(self._decisions)

    @property
    def decisions_recorded(self) -> int:
        return len(self._decisions)

    def reset(self) -> None:
        self._decisions.clear()
