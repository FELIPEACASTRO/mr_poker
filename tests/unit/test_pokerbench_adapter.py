"""Tests for the PokerBenchAdapter that converts HuggingFace PokerBench rows."""

from __future__ import annotations

import pytest

from packages.dataset_builder.pokerbench_adapter import (
    PokerBenchAdapter,
    _board_texture,
    _canonical_hole,
    _parse_action_output,
    _parse_card_text,
    parse_pokerbench_instruction,
)


# ---------------------------------------------------------------------------
# Sample PokerBench instructions (from the real dataset)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Card parsing
# ---------------------------------------------------------------------------


class TestParseCardText:
    def test_king_of_heart(self) -> None:
        assert _parse_card_text("King of Heart") == "Kh"

    def test_ace_of_spade(self) -> None:
        assert _parse_card_text("Ace of Spade") == "As"

    def test_ten_of_diamond(self) -> None:
        assert _parse_card_text("Ten Of Diamond") == "Td"

    def test_two_of_club(self) -> None:
        assert _parse_card_text("Two Of Club") == "2c"

    def test_invalid_returns_none(self) -> None:
        assert _parse_card_text("Joker of Wild") is None

    def test_empty_returns_none(self) -> None:
        assert _parse_card_text("") is None


# ---------------------------------------------------------------------------
# Action output parsing
# ---------------------------------------------------------------------------


class TestParseActionOutput:
    def test_check(self) -> None:
        assert _parse_action_output("check") == ("check", 0)

    def test_fold(self) -> None:
        assert _parse_action_output("fold") == ("fold", 0)

    def test_call(self) -> None:
        assert _parse_action_output("call") == ("call", 0)

    def test_bet_with_amount(self) -> None:
        assert _parse_action_output("bet 10") == ("bet", 10)

    def test_raise_with_amount(self) -> None:
        assert _parse_action_output("raise 64") == ("raise", 64)

    def test_all_in(self) -> None:
        action, _ = _parse_action_output("all in")
        assert action == "all_in"


# ---------------------------------------------------------------------------
# Canonical hole classification
# ---------------------------------------------------------------------------


class TestCanonicalHole:
    def test_suited(self) -> None:
        assert _canonical_hole(["Kh", "3h"]) == "K3s"

    def test_offsuit(self) -> None:
        assert _canonical_hole(["Ah", "Jc"]) == "AJo"

    def test_pair(self) -> None:
        assert _canonical_hole(["Qs", "Qh"]) == "QQ"

    def test_reversed_order(self) -> None:
        assert _canonical_hole(["3h", "Kh"]) == "K3s"

    def test_invalid(self) -> None:
        assert _canonical_hole(["Ah"]) == "??"


# ---------------------------------------------------------------------------
# Board texture
# ---------------------------------------------------------------------------


class TestBoardTexture:
    def test_preflop(self) -> None:
        assert _board_texture([]) == "preflop"

    def test_rainbow_unpaired(self) -> None:
        texture = _board_texture(["Th", "3s", "2d"])
        assert "rainbow" in texture
        assert "unpaired" in texture

    def test_monotone(self) -> None:
        texture = _board_texture(["Ah", "Kh", "Qh"])
        assert "monotone" in texture

    def test_paired(self) -> None:
        texture = _board_texture(["4s", "4h", "3h"])
        assert "paired" in texture


# ---------------------------------------------------------------------------
# Full instruction parsing
# ---------------------------------------------------------------------------


class TestParseInstruction:
    def test_turn_check_scenario(self) -> None:
        result = parse_pokerbench_instruction(INSTRUCTION_CHECK)
        assert result is not None
        assert result["position"] == "BTN"
        assert result["hole_cards"] == ["Kh", "3h"]
        assert result["hole_class"] == "K3s"
        assert result["street"] == "turn"
        assert len(result["board"]) == 4
        assert result["pot"] == 13.0
        assert result["is_ip"] is True

    def test_river_bet_scenario(self) -> None:
        result = parse_pokerbench_instruction(INSTRUCTION_BET)
        assert result is not None
        assert result["position"] == "BTN"
        assert result["hole_cards"] == ["Ah", "Jc"]
        assert result["street"] == "river"
        assert len(result["board"]) == 5
        assert result["pot"] == 13.0

    def test_preflop_scenario(self) -> None:
        result = parse_pokerbench_instruction(INSTRUCTION_PREFLOP)
        assert result is not None
        assert result["position"] == "CO"
        assert result["hole_class"] == "AKo"
        assert result["street"] == "pre_flop"
        assert len(result["board"]) == 0
        assert result["board_texture"] == "preflop"

    def test_invalid_instruction(self) -> None:
        assert parse_pokerbench_instruction("This is not a poker hand") is None

    def test_flop_cards_parsed(self) -> None:
        result = parse_pokerbench_instruction(INSTRUCTION_CHECK)
        assert result is not None
        assert "Th" in result["board"]
        assert "3s" in result["board"]
        assert "2d" in result["board"]

    def test_to_call_when_facing_bet(self) -> None:
        """When BB checks on turn, hero has to_call=0."""
        result = parse_pokerbench_instruction(INSTRUCTION_CHECK)
        assert result is not None
        assert result["to_call"] == 0.0

    def test_to_call_preflop_facing_raise(self) -> None:
        result = parse_pokerbench_instruction(INSTRUCTION_PREFLOP)
        assert result is not None
        assert result["to_call"] == 2.5


# ---------------------------------------------------------------------------
# Adapter convert_row
# ---------------------------------------------------------------------------


class TestPokerBenchAdapter:
    def setup_method(self) -> None:
        self.adapter = PokerBenchAdapter(equity_samples=50, equity_seed=42)

    def test_convert_check_row(self) -> None:
        row = self.adapter.convert_row(INSTRUCTION_CHECK, "check", row_index=0)
        assert row is not None
        assert row["source_type"] == "pokerbench"
        assert row["label_action"] == "check"
        assert row["label_confidence"] == 0.95
        assert row["label_source"] == "pokerbench_solver"
        assert row["street"] == "turn"
        assert row["hole_class"] == "K3s"
        assert "bucket_key" in row["bucket_info"]
        assert isinstance(row["taxonomy_tags"], list)
        assert row["split"] in ("train", "validation", "test")

    def test_convert_bet_row(self) -> None:
        row = self.adapter.convert_row(INSTRUCTION_BET, "bet 10", row_index=1)
        assert row is not None
        assert row["label_action"] == "bet"
        assert row["label_amount"] == 10
        assert row["street"] == "river"

    def test_convert_preflop_row(self) -> None:
        row = self.adapter.convert_row(INSTRUCTION_PREFLOP, "raise 7.5", row_index=2)
        assert row is not None
        assert row["label_action"] == "raise"
        assert row["label_amount"] == 7
        assert row["street"] == "pre_flop"
        assert row["hole_class"] == "AKo"

    def test_convert_invalid_returns_none(self) -> None:
        row = self.adapter.convert_row("invalid text", "check", row_index=99)
        assert row is None

    def test_convert_row_has_required_fields(self) -> None:
        row = self.adapter.convert_row(INSTRUCTION_CHECK, "check", row_index=0)
        assert row is not None
        required_fields = {
            "row_id", "source_type", "source_key", "session_id", "hand_id",
            "actor_seat", "street", "hole_class", "board_texture", "pot",
            "to_call", "pot_odds", "estimated_equity", "legal_actions",
            "bucket_info", "label_action", "label_confidence", "label_source",
            "taxonomy_tags", "split",
        }
        assert required_fields.issubset(row.keys())

    def test_bucket_info_has_bucket_key(self) -> None:
        row = self.adapter.convert_row(INSTRUCTION_CHECK, "check", row_index=0)
        assert row is not None
        bucket_info = row["bucket_info"]
        assert "bucket_key" in bucket_info
        assert "|" in bucket_info["bucket_key"]

    def test_estimated_equity_is_float(self) -> None:
        row = self.adapter.convert_row(INSTRUCTION_CHECK, "check", row_index=0)
        assert row is not None
        assert isinstance(row["estimated_equity"], float)
        assert 0.0 <= row["estimated_equity"] <= 1.0


# ---------------------------------------------------------------------------
# Batch conversion
# ---------------------------------------------------------------------------


class TestPokerBenchAdapterBatch:
    def setup_method(self) -> None:
        self.adapter = PokerBenchAdapter(equity_samples=50, equity_seed=42)

    def test_convert_batch(self) -> None:
        batch = [
            {"instruction": INSTRUCTION_CHECK, "output": "check"},
            {"instruction": INSTRUCTION_BET, "output": "bet 10"},
            {"instruction": "invalid", "output": "fold"},
        ]
        results = self.adapter.convert_batch(batch)
        assert len(results) == 2  # 3rd row invalid, skipped

    def test_convert_batch_empty(self) -> None:
        results = self.adapter.convert_batch([])
        assert results == []

    def test_batch_source_keys_unique(self) -> None:
        batch = [
            {"instruction": INSTRUCTION_CHECK, "output": "check"},
            {"instruction": INSTRUCTION_BET, "output": "bet 10"},
        ]
        results = self.adapter.convert_batch(batch)
        source_keys = [r["source_key"] for r in results]
        assert len(source_keys) == len(set(source_keys))


# ---------------------------------------------------------------------------
# Integration with DatasetBuilder format
# ---------------------------------------------------------------------------


class TestDatasetBuilderIntegration:
    def test_pokerbench_rows_in_builder(self) -> None:
        """PokerBench rows can be passed to DatasetBuilder via include_external_rows."""
        from packages.dataset_builder.builder import DatasetBuilder

        class FakeStore:
            def get_decision_traces(self):
                return []

        adapter = PokerBenchAdapter(equity_samples=50, equity_seed=42)
        row = adapter.convert_row(INSTRUCTION_CHECK, "check", row_index=0)
        assert row is not None

        builder = DatasetBuilder(FakeStore(), report_dir="/tmp/test_pokerbench_ds")
        rows = builder.build_rows(include_external_rows=[row])
        assert len(rows) == 1
        assert rows[0]["source_type"] == "pokerbench"

    def test_manifest_counts_pokerbench(self) -> None:
        from packages.dataset_builder.builder import DatasetBuilder

        class FakeStore:
            def get_decision_traces(self):
                return []

        adapter = PokerBenchAdapter(equity_samples=50, equity_seed=42)
        pokerbench_rows = adapter.convert_batch([
            {"instruction": INSTRUCTION_CHECK, "output": "check"},
            {"instruction": INSTRUCTION_BET, "output": "bet 10"},
        ])

        builder = DatasetBuilder(FakeStore(), report_dir="/tmp/test_pokerbench_ds2")
        rows = builder.build_rows(include_external_rows=pokerbench_rows)
        manifest = builder.build_manifest(rows, dataset_name="pokerbench_test")

        assert manifest["row_count"] == 2
        assert manifest["source_counts"]["pokerbench"] == 2
