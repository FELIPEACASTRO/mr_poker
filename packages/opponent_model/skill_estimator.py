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
