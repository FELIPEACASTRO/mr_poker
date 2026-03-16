"""Tests for PokerBench-based PolicyTableModel training pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from packages.dataset_builder.pokerbench_adapter import PokerBenchAdapter
from packages.policy_table.model import PolicyTableModel
from packages.policy_table.registry import PolicyRegistry
from packages.policy_table.trainer import PolicyTableTrainer
from services.model_service.service import ModelService

# Reuse sample instructions from the adapter tests
INSTRUCTION_CHECK = (
    "You are a specialist in playing 6-handed No Limit Texas Holdem. "
    "The following will be a game scenario and you need to make the optimal decision. "
    "Here is a game summary: The small blind is 0.5 chips and the big blind is 1 chips. "
    "Everyone started with 100 chips. The player positions involved in this game are "
    "UTG, HJ, CO, BTN, SB, BB. In this hand, your position is BTN, and your holding is "
    "[King of Heart and Three of Heart]. Before the flop, BTN raise 2.5 chips, and BB call. "
    "Assume that all other players that is not mentioned folded. The flop comes "
    "Ten Of Heart, Three Of Spade, and Two Of Diamond, then BB bet 4 chips, and BTN call. "
    "The turn comes Five Of Diamond, then BB check. Now it is your turn to make a move. "
    "To remind you, the current pot size is 13.0 chips, and your holding is "
    "[King of Heart and Three of Heart]. Decide on an action based on the strength of "
    "your hand on this board, your position, and actions before you. "
    "Do not explain your answer. Your optimal action is:"
)

INSTRUCTION_BET = (
    "You are a specialist in playing 6-handed No Limit Texas Holdem. "
    "The following will be a game scenario and you need to make the optimal decision. "
    "Here is a game summary: The small blind is 0.5 chips and the big blind is 1 chips. "
    "Everyone started with 100 chips. The player positions involved in this game are "
    "UTG, HJ, CO, BTN, SB, BB. In this hand, your position is BTN, and your holding is "
    "[Ace of Heart and Jack of Club]. Before the flop, BTN raise 2.5 chips, and BB call. "
    "Assume that all other players that is not mentioned folded. The flop comes "
    "Four Of Spade, Four Of Heart, and Three Of Heart, then BB check, and BTN check. "
    "The turn comes Jack Of Heart, then BB bet 4 chips, and BTN call. "
    "The river comes Eight Of Club, then BB check. Now it is your turn to make a move. "
    "To remind you, the current pot size is 13.0 chips, and your holding is "
    "[Ace of Heart and Jack of Club]. Decide on an action based on the strength of "
    "your hand on this board, your position, and actions before you. "
    "Do not explain your answer. Your optimal action is:"
)

INSTRUCTION_PREFLOP = (
    "You are a specialist in playing 6-handed No Limit Texas Holdem. "
    "The following will be a game scenario and you need to make the optimal decision. "
    "Here is a game summary: The small blind is 0.5 chips and the big blind is 1 chips. "
    "Everyone started with 100 chips. The player positions involved in this game are "
    "UTG, HJ, CO, BTN, SB, BB. In this hand, your position is CO, and your holding is "
    "[Ace of Spade and King of Diamond]. Before the flop, UTG raise 2.5 chips. "
    "Now it is your turn to make a move. "
    "To remind you, the current pot size is 4.0 chips, and your holding is "
    "[Ace of Spade and King of Diamond]. Decide on an action based on the strength of "
    "your hand on this board, your position, and actions before you. "
    "Do not explain your answer. Your optimal action is:"
)

SAMPLE_POKERBENCH_ROWS = [
    {"instruction": INSTRUCTION_CHECK, "output": "check"},
    {"instruction": INSTRUCTION_BET, "output": "bet 10"},
    {"instruction": INSTRUCTION_PREFLOP, "output": "raise 7.5"},
    {"instruction": INSTRUCTION_CHECK, "output": "bet 6"},
    {"instruction": INSTRUCTION_BET, "output": "bet 8"},
]


class TestPokerBenchTrainingPipeline:
    """Test the full pipeline: adapter -> trainer -> model."""

    def test_adapter_to_trainer(self) -> None:
        adapter = PokerBenchAdapter(equity_samples=0)
        converted = adapter.convert_batch(SAMPLE_POKERBENCH_ROWS)
        assert len(converted) == 5

        trainer = PolicyTableTrainer()
        model = trainer.train(converted, model_name="test_pokerbench")
        assert isinstance(model, PolicyTableModel)
        assert model.metadata["row_count"] == 5
        assert model.metadata["bucket_count"] > 0

    def test_model_predicts_after_training(self) -> None:
        adapter = PokerBenchAdapter(equity_samples=0)
        converted = adapter.convert_batch(SAMPLE_POKERBENCH_ROWS)
        trainer = PolicyTableTrainer()
        model = trainer.train(converted, model_name="test_predict")

        # Pick a bucket key from the training data
        bucket_key = converted[0]["bucket_info"]["bucket_key"]
        prediction = model.predict(bucket_key)
        assert prediction["action"] in ("check", "bet", "raise", "fold", "call", "all_in")
        assert prediction["confidence"] > 0.0
        assert prediction["support"] > 0

    def test_model_save_and_load(self, tmp_path) -> None:
        adapter = PokerBenchAdapter(equity_samples=0)
        converted = adapter.convert_batch(SAMPLE_POKERBENCH_ROWS)
        trainer = PolicyTableTrainer()
        model = trainer.train(converted, model_name="test_save")

        registry = PolicyRegistry(str(tmp_path))
        saved = registry.save(model)
        assert saved["model_id"] == model.model_id

        loaded = registry.load(model.model_id)
        assert loaded.model_id == model.model_id
        assert loaded.metadata["model_name"] == "test_save"
        assert len(loaded.table) == len(model.table)

    def test_bucket_coverage(self) -> None:
        adapter = PokerBenchAdapter(equity_samples=0)
        converted = adapter.convert_batch(SAMPLE_POKERBENCH_ROWS)
        trainer = PolicyTableTrainer()
        model = trainer.train(converted, model_name="test_buckets")

        # Should have at least 1 bucket
        assert model.metadata["bucket_count"] >= 1
        # All bucket entries should have action, confidence, support
        for entry in model.table.values():
            assert "action" in entry
            assert "confidence" in entry
            assert "support" in entry

    def test_action_distribution(self) -> None:
        adapter = PokerBenchAdapter(equity_samples=0)
        converted = adapter.convert_batch(SAMPLE_POKERBENCH_ROWS)
        actions = {row["label_action"] for row in converted}
        # Our sample has check, bet, raise
        assert "check" in actions
        assert "bet" in actions
        assert "raise" in actions


class TestModelServicePokerBench:
    """Test the ModelService.train_policy_table_from_pokerbench method."""

    def _make_service(self, tmp_path) -> ModelService:
        solver_labels = MagicMock()
        return ModelService(solver_labels=solver_labels, model_dir=str(tmp_path))

    def test_train_from_pokerbench_with_mock_dataset(self, tmp_path) -> None:
        service = self._make_service(tmp_path)

        # Create a fake datasets module whose load_dataset returns our sample
        import types

        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = lambda *a, **kw: SAMPLE_POKERBENCH_ROWS  # type: ignore[attr-defined]

        with patch.dict("sys.modules", {"datasets": fake_datasets}):
            result = service.train_policy_table_from_pokerbench(
                max_rows=5,
                equity_samples=0,
                model_name="test_service",
                config="easy",
            )

        assert "model_id" in result
        assert result["model_name"] == "test_service"
        assert result["source"] == "pokerbench"
        assert result["training_rows"] == 5
        assert result["bucket_count"] > 0

    def test_train_from_pokerbench_no_datasets_lib(self, tmp_path) -> None:
        service = self._make_service(tmp_path)
        with patch.dict("sys.modules", {"datasets": None}):
            with pytest.raises((RuntimeError, ImportError)):
                service.train_policy_table_from_pokerbench(
                    max_rows=5, equity_samples=0
                )
