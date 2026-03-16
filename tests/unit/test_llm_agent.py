"""Tests for the LLM-based poker agent with mocked inference."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from packages.common.types import ActionType
from packages.dataset_builder.pokerbench_adapter import _parse_action_output
from packages.llm_agent.prompt import (
    _card_to_text,
    build_pokerbench_prompt,
    build_prompt_from_spot,
)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


class TestCardToText:
    def test_king_of_hearts(self) -> None:
        assert _card_to_text("Kh") == "King of Heart"

    def test_ace_of_spades(self) -> None:
        assert _card_to_text("As") == "Ace of Spade"

    def test_ten_of_diamonds(self) -> None:
        assert _card_to_text("Td") == "Ten of Diamond"

    def test_two_of_clubs(self) -> None:
        assert _card_to_text("2c") == "Two of Club"


class TestBuildPokerbenchPrompt:
    def test_preflop_prompt(self) -> None:
        prompt = build_pokerbench_prompt(
            position="BTN",
            hole_cards=["Ah", "Kd"],
            board=[],
            pot=4.0,
            street="pre_flop",
        )
        assert "your position is BTN" in prompt
        assert "Ace of Heart" in prompt
        assert "King of Diamond" in prompt
        assert "pot size is 4.0 chips" in prompt
        assert "Your optimal action is:" in prompt
        # No board cards on preflop
        assert "flop comes" not in prompt

    def test_flop_prompt(self) -> None:
        prompt = build_pokerbench_prompt(
            position="BB",
            hole_cards=["Qh", "Jc"],
            board=["Th", "3s", "2d"],
            pot=13.0,
            street="flop",
        )
        assert "your position is BB" in prompt
        assert "Queen of Heart" in prompt
        assert "flop comes" in prompt.lower() or "Flop comes" in prompt
        assert "Ten of Heart" in prompt

    def test_turn_prompt_has_board(self) -> None:
        prompt = build_pokerbench_prompt(
            position="BTN",
            hole_cards=["Kh", "3h"],
            board=["Th", "3s", "2d", "5d"],
            pot=13.0,
            street="turn",
        )
        assert "turn comes" in prompt.lower() or "Turn comes" in prompt
        assert "Five of Diamond" in prompt

    def test_river_prompt_has_full_board(self) -> None:
        prompt = build_pokerbench_prompt(
            position="BB",
            hole_cards=["Ah", "Jc"],
            board=["4s", "4h", "3h", "Jh", "8c"],
            pot=13.0,
            street="river",
        )
        assert "river comes" in prompt.lower() or "River comes" in prompt
        assert "Eight of Club" in prompt


class TestBuildPromptFromSpot:
    def test_from_spot_dict(self) -> None:
        spot = {
            "actor_seat": 0,
            "button_seat": 0,
            "hole_cards": ["Ah", "Kd"],
            "board": [],
            "pot": 4.0,
            "street": "pre_flop",
        }
        prompt = build_prompt_from_spot(spot)
        assert "your position is BTN" in prompt
        assert "Ace of Heart" in prompt

    def test_from_spot_bb(self) -> None:
        spot = {
            "actor_seat": 1,
            "button_seat": 0,
            "hole_cards": ["Qh", "Jc"],
            "board": ["Th", "3s", "2d"],
            "pot": 13.0,
            "street": "flop",
        }
        prompt = build_prompt_from_spot(spot)
        assert "your position is BB" in prompt


# ---------------------------------------------------------------------------
# Action parsing (reused from adapter)
# ---------------------------------------------------------------------------


class TestActionParsing:
    def test_check(self) -> None:
        assert _parse_action_output("check") == ("check", 0)

    def test_fold(self) -> None:
        assert _parse_action_output("fold") == ("fold", 0)

    def test_bet_with_amount(self) -> None:
        assert _parse_action_output("bet 10") == ("bet", 10)

    def test_raise_with_amount(self) -> None:
        assert _parse_action_output("raise 64") == ("raise", 64)

    def test_whitespace_handling(self) -> None:
        assert _parse_action_output("  check  ") == ("check", 0)

    def test_caps_handling(self) -> None:
        assert _parse_action_output("CHECK") == ("check", 0)

    def test_all_in(self) -> None:
        action, _ = _parse_action_output("all in")
        assert action == "all_in"

    def test_call_with_amount(self) -> None:
        assert _parse_action_output("call 5") == ("call", 5)


# ---------------------------------------------------------------------------
# LLMPokerAgent with mocked inference
# ---------------------------------------------------------------------------


class TestLLMPokerAgent:
    def _make_agent_with_mock(self, llm_output: str):
        """Create an LLMPokerAgent with mocked inference."""
        from packages.llm_agent.agent import LLMPokerAgent

        agent = LLMPokerAgent.__new__(LLMPokerAgent)
        # Initialize BaselineAgent parts
        from packages.baseline_agent.factory import BaselineAgentFactory
        agent._equity_estimator = BaselineAgentFactory.create_equity_estimator()
        agent._decision_strategy = BaselineAgentFactory.create_decision_strategy()
        # Mock inference
        agent.inference = MagicMock()
        agent.inference.model_id = "test-model"
        agent.inference.generate = MagicMock(return_value=llm_output)
        return agent

    def test_action_map_coverage(self) -> None:
        from packages.llm_agent.agent import _ACTION_MAP

        assert _ACTION_MAP["fold"] == ActionType.FOLD
        assert _ACTION_MAP["check"] == ActionType.CHECK
        assert _ACTION_MAP["call"] == ActionType.CALL
        assert _ACTION_MAP["bet"] == ActionType.BET
        assert _ACTION_MAP["raise"] == ActionType.RAISE
        assert _ACTION_MAP["all_in"] == ActionType.ALL_IN
        assert _ACTION_MAP["all-in"] == ActionType.ALL_IN

    def test_llm_inference_class(self) -> None:
        from packages.llm_agent.inference import LLMInference

        inference = LLMInference("test-model", quantize=False)
        assert inference.model_id == "test-model"
        assert not inference.is_loaded
        assert inference.max_new_tokens == 20

    def test_llm_inference_requires_transformers(self) -> None:
        from packages.llm_agent.inference import LLMInference

        inference = LLMInference("test-model")
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises((RuntimeError, ImportError)):
                inference._load()


# ---------------------------------------------------------------------------
# TournamentService integration
# ---------------------------------------------------------------------------


class TestTournamentServiceLLM:
    def test_agent_factory_llm_prefix(self) -> None:
        from services.tournament_service.service import TournamentService

        ts = TournamentService(engine=MagicMock(), model_registry=MagicMock())
        with patch("packages.llm_agent.LLMPokerAgent") as mock_cls:
            mock_cls.return_value = MagicMock()
            agent = ts._agent_factory("llm:felipesp1983/poker-solver-qwen-1.5b-sft")
            mock_cls.assert_called_once_with(
                model_id="felipesp1983/poker-solver-qwen-1.5b-sft"
            )

    def test_agent_factory_unknown_raises(self) -> None:
        from services.tournament_service.service import TournamentService

        ts = TournamentService(engine=MagicMock(), model_registry=MagicMock())
        with pytest.raises(KeyError):
            ts._agent_factory("unknown_agent")
