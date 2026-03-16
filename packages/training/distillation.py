"""Perfect Info Distillation — train a student to approximate a teacher that sees all cards.

The teacher network receives extended features including opponent hole cards,
allowing it to make "perfect information" decisions. The student network only
sees standard features (its own cards, board, action history) but is trained
to mimic the teacher's output distribution via KL-divergence minimisation.

This produces a student that has learned to approximate "what would I do if
I could see everything?" — capturing implicit tells and opponent-range
reasoning that the teacher encodes from perfect information.

Reference: Inspired by policy distillation techniques in Libratus/Pluribus.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from packages.cfr_agent.deep_cfr import (
    FEATURE_DIM,
    NUM_ACTIONS,
    ACTION_INDEX,
    SimpleNN,
    extract_features,
)
from packages.common.types import ActionType
from packages.strategy.mixed import ActionDistribution

# Teacher gets 5 extra features encoding opponent hole cards
TEACHER_EXTRA_DIM = 5
TEACHER_FEATURE_DIM = FEATURE_DIM + TEACHER_EXTRA_DIM


@dataclass
class DistillationResult:
    """Metrics from a distillation run."""
    teacher_accuracy: float = 0.0
    student_accuracy: float = 0.0
    kl_divergence: float = 0.0


def _softmax(logits: list[float]) -> list[float]:
    """Numerically stable softmax."""
    max_v = max(logits) if logits else 0.0
    exps = [math.exp(v - max_v) for v in logits]
    total = sum(exps)
    if total == 0:
        return [1.0 / len(logits)] * len(logits)
    return [e / total for e in exps]


def _kl_divergence(p: list[float], q: list[float]) -> float:
    """KL(p || q) — how much q diverges from p."""
    kl = 0.0
    for pi, qi in zip(p, q):
        if pi > 1e-10:
            qi_safe = max(qi, 1e-10)
            kl += pi * math.log(pi / qi_safe)
    return kl


def generate_perfect_info_features(
    standard_features: list[float],
    opponent_high_rank: float = 0.0,
    opponent_low_rank: float = 0.0,
    opponent_suited: float = 0.0,
    opponent_pair: float = 0.0,
    opponent_hand_strength: float = 0.0,
) -> list[float]:
    """Extend standard features with opponent hole card info (5 floats).

    Args:
        standard_features: The FEATURE_DIM-length vector from extract_features.
        opponent_high_rank: 0-1 normalized high card rank of opponent.
        opponent_low_rank: 0-1 normalized low card rank of opponent.
        opponent_suited: 1.0 if opponent's hole cards are suited, else 0.0.
        opponent_pair: 1.0 if opponent has a pair, else 0.0.
        opponent_hand_strength: 0-1 overall hand strength estimate.

    Returns:
        Extended feature vector of length TEACHER_FEATURE_DIM.
    """
    return standard_features + [
        opponent_high_rank,
        opponent_low_rank,
        opponent_suited,
        opponent_pair,
        opponent_hand_strength,
    ]


@dataclass
class TrajectoryPoint:
    """A single decision point from a game trajectory."""
    standard_features: list[float] = field(default_factory=list)
    extended_features: list[float] = field(default_factory=list)
    action_taken: int = 0  # index into ACTION_INDEX
    legal_mask: list[bool] = field(default_factory=lambda: [True] * NUM_ACTIONS)


class PerfectInfoDistiller:
    """Distill perfect-information teacher knowledge into an imperfect-info student.

    Pipeline:
        1. Train a teacher that sees all cards (perfect information).
        2. Distill the teacher's strategy into a student that only sees its own cards.
        3. The student learns implicit opponent-modelling from the teacher's behaviour.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        seed: int = 42,
    ) -> None:
        self.rng = random.Random(seed)
        self.hidden_dim = hidden_dim

        # Teacher: sees standard features + opponent hole card features
        self.teacher = SimpleNN(
            TEACHER_FEATURE_DIM, hidden_dim, NUM_ACTIONS, seed=seed
        )
        # Student: only sees standard features
        self.student = SimpleNN(
            FEATURE_DIM, hidden_dim, NUM_ACTIONS, seed=seed + 1
        )

    def train_teacher(
        self,
        trajectories: list[TrajectoryPoint],
        epochs: int = 10,
        lr: float = 0.001,
    ) -> float:
        """Train the teacher network on trajectories with full information.

        The teacher is trained with supervised learning on the extended features
        (which include opponent hole cards) to predict the correct action.

        Returns:
            Final average loss.
        """
        if not trajectories:
            return 0.0

        total_loss = 0.0
        count = 0
        for _ in range(epochs):
            self.rng.shuffle(trajectories)
            for tp in trajectories:
                # One-hot target from the action taken
                target = [0.0] * NUM_ACTIONS
                target[tp.action_taken] = 1.0
                loss = self.teacher.train_step(tp.extended_features, target, lr=lr)
                total_loss += loss
                count += 1

        return total_loss / max(count, 1)

    def distill(
        self,
        trajectories: list[TrajectoryPoint],
        epochs: int = 20,
        lr: float = 0.001,
    ) -> DistillationResult:
        """Distill teacher knowledge into the student via KL-divergence.

        For each trajectory point, get the teacher's softmax output on extended
        features, then train the student (on standard features) to match.

        Returns:
            DistillationResult with accuracy and KL divergence metrics.
        """
        if not trajectories:
            return DistillationResult()

        total_kl = 0.0
        teacher_correct = 0
        student_correct = 0
        n = 0

        for _ in range(epochs):
            self.rng.shuffle(trajectories)
            for tp in trajectories:
                # Teacher's soft targets
                teacher_logits = self.teacher.forward(tp.extended_features)
                teacher_probs = _softmax(teacher_logits)

                # Mask illegal actions
                for i in range(NUM_ACTIONS):
                    if not tp.legal_mask[i]:
                        teacher_probs[i] = 0.0
                t_sum = sum(teacher_probs)
                if t_sum > 0:
                    teacher_probs = [p / t_sum for p in teacher_probs]
                else:
                    teacher_probs = [1.0 / NUM_ACTIONS] * NUM_ACTIONS

                # Train student to match teacher's soft distribution
                loss = self.student.train_step(
                    tp.standard_features, teacher_probs, lr=lr
                )

                # Evaluate
                student_logits = self.student.forward(tp.standard_features)
                student_probs = _softmax(student_logits)

                kl = _kl_divergence(teacher_probs, student_probs)
                total_kl += kl

                # Accuracy: does the argmax match the true action?
                teacher_action = teacher_probs.index(max(teacher_probs))
                student_action = student_probs.index(max(student_probs))

                if teacher_action == tp.action_taken:
                    teacher_correct += 1
                if student_action == tp.action_taken:
                    student_correct += 1
                n += 1

        return DistillationResult(
            teacher_accuracy=teacher_correct / max(n, 1),
            student_accuracy=student_correct / max(n, 1),
            kl_divergence=total_kl / max(n, 1),
        )

    def get_student_strategy(
        self,
        features: list[float],
        legal_actions: set[ActionType],
    ) -> ActionDistribution:
        """Get the student's strategy for a given information set."""
        logits = self.student.forward(features)
        probs_raw = _softmax(logits)

        probs: dict[ActionType, float] = {}
        for action in legal_actions:
            idx = ACTION_INDEX.get(action)
            if idx is not None:
                probs[action] = probs_raw[idx]

        total = sum(probs.values())
        if total > 0:
            probs = {a: p / total for a, p in probs.items()}
        else:
            n = len(legal_actions) or 1
            probs = {a: 1.0 / n for a in legal_actions}

        return ActionDistribution(probabilities=probs)

    def get_teacher_strategy(
        self,
        extended_features: list[float],
        legal_actions: set[ActionType],
    ) -> ActionDistribution:
        """Get the teacher's strategy (requires extended features)."""
        logits = self.teacher.forward(extended_features)
        probs_raw = _softmax(logits)

        probs: dict[ActionType, float] = {}
        for action in legal_actions:
            idx = ACTION_INDEX.get(action)
            if idx is not None:
                probs[action] = probs_raw[idx]

        total = sum(probs.values())
        if total > 0:
            probs = {a: p / total for a, p in probs.items()}
        else:
            n = len(legal_actions) or 1
            probs = {a: 1.0 / n for a in legal_actions}

        return ActionDistribution(probabilities=probs)
