"""Tests for SkillTrainer — supervised training with aggregate features + MLP.

Validates the complete training pipeline:
1. Feature extraction from decision sequences
2. MLP forward/backward passes
3. Training convergence on synthetic data
4. Evaluation accuracy metrics
"""
from __future__ import annotations

import pytest

from packages.opponent_model.skill_estimator import (
    DecisionFeature,
    SkillEstimate,
    SkillTrainer,
    generate_training_data,
    _extract_aggregate_features,
)


class TestAggregateFeatures:
    """Test aggregate feature extraction from decision sequences."""

    def test_empty_sequence(self):
        features = _extract_aggregate_features([])
        assert len(features) == 20
        assert all(f == 0.0 for f in features)

    def test_feature_dimensions(self):
        decisions = [
            DecisionFeature(action_type=0.8, bet_fraction=0.5, position_score=0.7),
            DecisionFeature(action_type=0.2, bet_fraction=0.0, position_score=0.3),
            DecisionFeature(action_type=0.6, bet_fraction=0.7, position_score=0.9),
        ]
        features = _extract_aggregate_features(decisions)
        assert len(features) == 20

    def test_aggressive_player_features(self):
        """Aggressive players should have high mean action type."""
        decisions = [
            DecisionFeature(action_type=0.8) for _ in range(10)
        ]
        features = _extract_aggregate_features(decisions)
        assert features[0] > 0.7  # mean action type
        assert features[2] > 0.9  # aggressive fraction

    def test_passive_player_features(self):
        """Passive players should have low mean action type."""
        decisions = [
            DecisionFeature(action_type=0.2) for _ in range(10)
        ]
        features = _extract_aggregate_features(decisions)
        assert features[0] < 0.3  # mean action type
        assert features[3] > 0.9  # passive fraction

    def test_features_bounded(self):
        """All features should be in reasonable range."""
        decisions = [
            DecisionFeature(
                action_type=0.5, decision_time_norm=0.5,
                bet_fraction=0.5, position_score=0.5,
                street=0.5, pot_committed=0.3,
                aggression_context=0.4, hand_strength=0.6,
            )
            for _ in range(20)
        ]
        features = _extract_aggregate_features(decisions)
        for i, f in enumerate(features):
            assert -1.5 <= f <= 2.5, f"Feature {i} out of range: {f}"


class TestGenerateTrainingData:
    """Test synthetic training data generation."""

    def test_generates_correct_count(self):
        data = generate_training_data(n_samples=100, seed=42)
        assert len(data) == 100

    def test_data_format(self):
        data = generate_training_data(n_samples=10, seed=42)
        for decisions, skill in data:
            assert isinstance(decisions, list)
            assert isinstance(skill, float)
            assert 0.0 <= skill <= 1.0
            assert len(decisions) >= 3

    def test_skill_distribution(self):
        """Skills should be uniformly distributed."""
        data = generate_training_data(n_samples=1000, seed=42)
        skills = [s for _, s in data]
        mean_skill = sum(skills) / len(skills)
        assert 0.4 < mean_skill < 0.6  # roughly uniform

    def test_different_seeds_different_data(self):
        data1 = generate_training_data(n_samples=10, seed=1)
        data2 = generate_training_data(n_samples=10, seed=2)
        skills1 = [s for _, s in data1]
        skills2 = [s for _, s in data2]
        assert skills1 != skills2


class TestSkillTrainer:
    """Test the MLP-based skill trainer."""

    def test_initialization(self):
        trainer = SkillTrainer(seed=42)
        assert len(trainer.w1) == 32  # default hidden dim
        assert len(trainer.w2) == 32

    def test_custom_hidden_dim(self):
        trainer = SkillTrainer(hidden_dim=64, seed=42)
        assert len(trainer.w1) == 64
        assert len(trainer.w2) == 64

    def test_predict(self):
        trainer = SkillTrainer(seed=42)
        decisions = [
            DecisionFeature(action_type=0.8, bet_fraction=0.5)
            for _ in range(10)
        ]
        pred = trainer.predict(decisions)
        assert 0.0 <= pred <= 1.0

    def test_train_epoch_returns_loss_and_acc(self):
        trainer = SkillTrainer(seed=42)
        data = generate_training_data(n_samples=50, seed=42)
        loss, acc = trainer.train_epoch(data)
        assert loss >= 0.0
        assert 0.0 <= acc <= 1.0

    def test_evaluate_returns_metrics(self):
        trainer = SkillTrainer(seed=42)
        data = generate_training_data(n_samples=50, seed=42)
        mse, acc, mae = trainer.evaluate(data)
        assert mse >= 0.0
        assert 0.0 <= acc <= 1.0
        assert mae >= 0.0

    def test_training_reduces_loss(self):
        """Loss should decrease over multiple epochs."""
        trainer = SkillTrainer(learning_rate=0.01, seed=42)
        data = generate_training_data(n_samples=200, seed=42)

        initial_loss, _ = trainer.train_epoch(data)
        for _ in range(20):
            loss, _ = trainer.train_epoch(data)
        final_loss = loss

        assert final_loss < initial_loss

    def test_training_improves_accuracy(self):
        """Accuracy should improve with training."""
        trainer = SkillTrainer(learning_rate=0.01, seed=42)
        train_data = generate_training_data(n_samples=500, seed=42)
        test_data = generate_training_data(n_samples=100, seed=99)

        _, initial_acc, _ = trainer.evaluate(test_data)

        for _ in range(30):
            trainer.train_epoch(train_data)

        _, final_acc, _ = trainer.evaluate(test_data)
        assert final_acc > initial_acc

    def test_lr_decay(self):
        trainer = SkillTrainer(learning_rate=0.01, lr_decay=0.99, seed=42)
        data = generate_training_data(n_samples=50, seed=42)
        initial_lr = trainer.lr
        trainer.train_epoch(data)
        assert trainer.lr < initial_lr

    def test_convergence_with_enough_data(self):
        """With enough data and epochs, should achieve >50% accuracy."""
        trainer = SkillTrainer(learning_rate=0.01, hidden_dim=48, seed=42)
        train_data = generate_training_data(n_samples=1000, seed=42)
        test_data = generate_training_data(n_samples=200, seed=99)

        for _ in range(50):
            trainer.train_epoch(train_data)

        _, accuracy, mae = trainer.evaluate(test_data)
        assert accuracy > 0.50, f"Accuracy {accuracy:.2%} should be > 50%"
        assert mae < 0.20, f"MAE {mae:.4f} should be < 0.20"
