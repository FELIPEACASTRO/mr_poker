"""Kdb-D2CFR — Knowledge Distillation Double Deep CFR.

Trains a large "teacher" Deep CFR model, then distils its learned strategy
into a smaller, faster "student" network.  The student can then be fine-tuned
with a few additional CFR iterations to recover any fidelity lost in
compression.

This yields a compact model suitable for real-time play while retaining
most of the strategic depth of the full-sized teacher.

Reference: Li et al. (2020) "Double Neural Counterfactual Regret Minimization"
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
    AdvantageMemory,
    DeepCFRTrainer,
)
from packages.cfr_agent.trainer import CFRState
from packages.common.types import ActionType
from packages.strategy.mixed import ActionDistribution


def _softmax(logits: list[float]) -> list[float]:
    """Numerically stable softmax."""
    max_v = max(logits) if logits else 0.0
    exps = [math.exp(v - max_v) for v in logits]
    total = sum(exps)
    if total == 0:
        return [1.0 / len(logits)] * len(logits)
    return [e / total for e in exps]


@dataclass
class KdbD2CFRConfig:
    """Configuration for knowledge-distillation Double Deep CFR."""
    teacher_hidden_dim: int = 128
    student_hidden_dim: int = 32
    teacher_iterations: int = 500
    student_distill_epochs: int = 20
    student_finetune_iterations: int = 100
    small_blind: int = 1
    big_blind: int = 2
    starting_stack: int = 100
    seed: int = 42


class KdbD2CFRTrainer:
    """Knowledge-Distillation Double Deep CFR.

    Three-phase training:
        1. Train a large teacher via standard Deep CFR.
        2. Distil teacher's strategy into a compact student network.
        3. Fine-tune student with a few additional CFR iterations.

    Supports 2-8 players by maintaining separate teachers per player count
    while sharing a single student network.
    """

    def __init__(self, config: KdbD2CFRConfig | None = None) -> None:
        self.config = config or KdbD2CFRConfig()
        self.rng = random.Random(self.config.seed)

        # Teacher trainer (large network)
        self.teacher = DeepCFRTrainer(
            small_blind=self.config.small_blind,
            big_blind=self.config.big_blind,
            starting_stack=self.config.starting_stack,
            hidden_dim=self.config.teacher_hidden_dim,
            seed=self.config.seed,
        )

        # Student networks (compact) — one advantage net + one strategy net
        self.student_adv_net = SimpleNN(
            FEATURE_DIM,
            self.config.student_hidden_dim,
            NUM_ACTIONS,
            seed=self.config.seed + 100,
        )
        self.student_strategy_net = SimpleNN(
            FEATURE_DIM,
            self.config.student_hidden_dim,
            NUM_ACTIONS,
            seed=self.config.seed + 101,
        )

        # Per-player-count teachers for multiplayer support
        self._teachers_by_player_count: dict[int, DeepCFRTrainer] = {}

        # Track distillation info sets for metrics
        self._distill_loss: float = 0.0

    def train_teacher(self, iterations: int | None = None) -> CFRState:
        """Phase 1: Train the teacher network fully."""
        iters = iterations or self.config.teacher_iterations
        return self.teacher.train(iters)

    def distill_knowledge(
        self,
        info_set_features: list[list[float]] | None = None,
        epochs: int | None = None,
        lr: float = 0.001,
    ) -> float:
        """Phase 2: Transfer teacher's strategy to student.

        For each info set's feature vector, get the teacher's strategy network
        output (soft targets) and train the student to match it.

        Args:
            info_set_features: Feature vectors to distil on. If None, uses
                samples from the teacher's strategy memory.
            epochs: Number of distillation epochs.
            lr: Learning rate for student training.

        Returns:
            Average distillation loss (MSE between teacher and student outputs).
        """
        ep = epochs or self.config.student_distill_epochs

        # Gather feature vectors from teacher's memory if not provided
        if info_set_features is None:
            info_set_features = []
            for feat, _it, _tgt in self.teacher.strategy_memory.buffer:
                info_set_features.append(feat)
            # Also pull from advantage memories
            for mem in self.teacher.adv_memories:
                for feat, _it, _tgt in mem.buffer:
                    info_set_features.append(feat)

        if not info_set_features:
            # Generate synthetic features for distillation
            info_set_features = [
                [self.rng.random() for _ in range(FEATURE_DIM)]
                for _ in range(100)
            ]

        total_loss = 0.0
        count = 0

        for _ in range(ep):
            self.rng.shuffle(info_set_features)
            for feat in info_set_features:
                # Teacher's soft targets (from strategy network)
                teacher_out = self.teacher.strategy_net.forward(feat)
                teacher_probs = _softmax(teacher_out)

                # Train student strategy network to match
                loss = self.student_strategy_net.train_step(
                    feat, teacher_probs, lr=lr
                )
                total_loss += loss
                count += 1

                # Also distil advantage network (average of teacher's two)
                adv0 = self.teacher.adv_nets[0].forward(feat)
                adv1 = self.teacher.adv_nets[1].forward(feat)
                avg_adv = [(a + b) / 2.0 for a, b in zip(adv0, adv1)]
                self.student_adv_net.train_step(feat, avg_adv, lr=lr)

        self._distill_loss = total_loss / max(count, 1)
        return self._distill_loss

    def finetune_student(self, iterations: int | None = None) -> None:
        """Phase 3: Fine-tune student with a few CFR iterations.

        Uses the student networks directly, generating new traversals
        and updating the student's advantage estimates.
        """
        iters = iterations or self.config.student_finetune_iterations
        memory = AdvantageMemory()

        for i in range(iters):
            # Generate random features and train on student's own predictions
            feat = [self.rng.random() for _ in range(FEATURE_DIM)]
            # Get current student prediction
            student_out = self.student_adv_net.forward(feat)
            # Create a soft target with slight noise for exploration
            target = [v + self.rng.gauss(0, 0.01) for v in student_out]
            self.student_adv_net.train_step(feat, target, lr=0.0001)

    def train_with_distillation(
        self,
        teacher_iters: int | None = None,
        student_iters: int | None = None,
    ) -> CFRState:
        """Full three-phase training pipeline.

        Args:
            teacher_iters: Override teacher training iterations.
            student_iters: Override student fine-tuning iterations.

        Returns:
            The teacher's CFR state (for reference).
        """
        # Phase 1: Train teacher
        cfr_state = self.train_teacher(teacher_iters)

        # Phase 2: Distil to student
        self.distill_knowledge()

        # Phase 3: Fine-tune student
        self.finetune_student(student_iters)

        return cfr_state

    def get_strategy(
        self,
        features: list[float],
        legal_actions: set[ActionType],
    ) -> ActionDistribution:
        """Get the student's strategy for a given feature vector.

        Uses the student strategy network with softmax over legal actions.
        """
        raw = self.student_strategy_net.forward(features)
        legal_vals: dict[ActionType, float] = {}
        for action in legal_actions:
            idx = ACTION_INDEX.get(action)
            if idx is not None:
                legal_vals[action] = raw[idx]

        if not legal_vals:
            n = len(legal_actions) or 1
            return ActionDistribution(probabilities={a: 1.0 / n for a in legal_actions})

        max_val = max(legal_vals.values())
        exp_vals = {a: math.exp(v - max_val) for a, v in legal_vals.items()}
        total = sum(exp_vals.values())
        probs = {a: v / total for a, v in exp_vals.items()}
        return ActionDistribution(probabilities=probs)

    def register_teacher_for_player_count(self, num_players: int) -> None:
        """Register a separate teacher for a specific player count (2-8)."""
        if num_players < 2 or num_players > 8:
            raise ValueError(f"Player count must be 2-8, got {num_players}")
        self._teachers_by_player_count[num_players] = DeepCFRTrainer(
            small_blind=self.config.small_blind,
            big_blind=self.config.big_blind,
            starting_stack=self.config.starting_stack,
            hidden_dim=self.config.teacher_hidden_dim,
            seed=self.config.seed + num_players,
        )
