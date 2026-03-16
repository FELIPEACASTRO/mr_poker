"""Adapter to convert RZ412/PokerBench dataset rows into mr_poker JSONL format.

PokerBench contains 574K+ solver-computed optimal decisions for 6-handed NLHE.
Each row has an ``instruction`` (natural-language game state) and an ``output``
(optimal action).  This adapter parses the text, extracts structured features,
and produces rows compatible with :class:`DatasetBuilder`.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from uuid import uuid4

from packages.dataset_builder.builder import stable_split
from packages.solver_like import bucketize_spot
from packages.taxonomy.spot_taxonomy import classify_trace

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Card / rank helpers
# ---------------------------------------------------------------------------

_RANK_WORDS: dict[str, str] = {
    "ace": "A", "king": "K", "queen": "Q", "jack": "J", "ten": "T",
    "nine": "9", "eight": "8", "seven": "7", "six": "6", "five": "5",
    "four": "4", "three": "3", "two": "2", "deuce": "2",
}

_SUIT_WORDS: dict[str, str] = {
    "heart": "h", "hearts": "h",
    "diamond": "d", "diamonds": "d",
    "club": "c", "clubs": "c",
    "spade": "s", "spades": "s",
}

_POSITION_IP: frozenset[str] = frozenset({"BTN", "CO", "HJ"})

_STREET_FROM_BOARD_SIZE: dict[int, str] = {
    0: "pre_flop",
    3: "flop",
    4: "turn",
    5: "river",
}

# Pre-compiled patterns
_HOLDING_RE = re.compile(
    r"\[([A-Za-z]+ of [A-Za-z]+) and ([A-Za-z]+ of [A-Za-z]+)\]"
)
_POSITION_RE = re.compile(r"your position is (\w+)")
_POT_RE = re.compile(r"current pot size is ([\d.]+) chips")
_STARTING_STACK_RE = re.compile(r"Everyone started with (\d+) chips")

_FLOP_RE = re.compile(
    r"[Tt]he flop comes ([A-Za-z]+ [Oo]f [A-Za-z]+),\s*"
    r"([A-Za-z]+ [Oo]f [A-Za-z]+),\s*and ([A-Za-z]+ [Oo]f [A-Za-z]+)"
)
_TURN_RE = re.compile(r"[Tt]he turn comes ([A-Za-z]+ [Oo]f [A-Za-z]+)")
_RIVER_RE = re.compile(r"[Tt]he river comes ([A-Za-z]+ [Oo]f [A-Za-z]+)")

_ACTION_RE = re.compile(
    r"(\w+) (fold|check|call|bet|raise|all-in|all in)(?: ([\d.]+) chips)?"
)

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_card_text(text: str) -> str | None:
    """Convert 'King of Heart' → 'Kh', 'Ten Of Spade' → 'Ts', etc."""
    parts = text.strip().lower().replace(" of ", " ").split()
    if len(parts) != 2:
        return None
    rank = _RANK_WORDS.get(parts[0])
    suit = _SUIT_WORDS.get(parts[1])
    if rank is None or suit is None:
        return None
    return f"{rank}{suit}"


def _parse_action_output(output: str) -> tuple[str, int]:
    """Parse PokerBench output like 'check', 'fold', 'bet 10', 'raise 64'.

    Returns (action_type, amount).
    """
    output = output.strip().lower()
    if output in ("check", "fold", "call"):
        return output, 0
    parts = output.split()
    if len(parts) == 2 and parts[0] in ("bet", "raise", "call"):
        try:
            amount = int(float(parts[1]))
        except ValueError:
            amount = 0
        return parts[0], amount
    if "all" in output:
        return "all_in", 0
    return output, 0


def _canonical_hole(cards: list[str]) -> str:
    """Derive canonical hole class from two card strings like ['Kh', '3h']."""
    if len(cards) != 2:
        return "??"
    rank_order = {r: i for i, r in enumerate("23456789TJQKA", start=2)}
    r1, r2 = rank_order.get(cards[0][0], 0), rank_order.get(cards[1][0], 0)
    if r1 < r2:
        r1, r2 = r2, r1
        cards = [cards[1], cards[0]]
    high_rank_char = cards[0][0]
    low_rank_char = cards[1][0]
    if high_rank_char == low_rank_char:
        return f"{high_rank_char}{low_rank_char}"
    suited = "s" if cards[0][1] == cards[1][1] else "o"
    return f"{high_rank_char}{low_rank_char}{suited}"


def _board_texture(board: list[str]) -> str:
    """Compute board texture string from card strings like ['Th', '3s', '2d']."""
    if len(board) < 3:
        return "preflop"
    rank_order = {r: i for i, r in enumerate("23456789TJQKA", start=2)}
    suits = [c[1] for c in board]
    ranks = sorted(rank_order.get(c[0], 0) for c in board)
    unique_suits = len(set(suits))
    paired = len(set(ranks)) < len(ranks)
    span = max(ranks) - min(ranks)
    connectivity = "connected" if span <= 4 else "disconnected"
    suit_label = "monotone" if unique_suits == 1 else "two-tone" if unique_suits == 2 else "rainbow"
    paired_label = "paired" if paired else "unpaired"
    return f"{suit_label}-{paired_label}-{connectivity}"


def _compute_to_call(actions_on_street: list[tuple[str, str, float]], hero_pos: str) -> float:
    """Estimate to_call from the last action on the current street before hero's turn."""
    last_bet = 0.0
    hero_invested = 0.0
    for pos, action, amount in actions_on_street:
        if pos == hero_pos:
            if action in ("call", "bet", "raise"):
                hero_invested += amount
        else:
            if action in ("bet", "raise"):
                last_bet = amount
            elif action == "call":
                pass  # doesn't change to_call for hero
    return max(0.0, last_bet - hero_invested)


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------


def parse_pokerbench_instruction(instruction: str) -> dict[str, Any] | None:
    """Parse a PokerBench instruction string into structured game state.

    Returns ``None`` if the instruction cannot be parsed.
    """
    # Position
    pos_match = _POSITION_RE.search(instruction)
    if not pos_match:
        return None
    position = pos_match.group(1).upper()

    # Hole cards
    holding_match = _HOLDING_RE.search(instruction)
    if not holding_match:
        return None
    card1 = _parse_card_text(holding_match.group(1))
    card2 = _parse_card_text(holding_match.group(2))
    if card1 is None or card2 is None:
        return None
    hole_cards = [card1, card2]

    # Starting stack
    stack_match = _STARTING_STACK_RE.search(instruction)
    starting_stack = int(stack_match.group(1)) if stack_match else 100

    # Board cards
    board: list[str] = []
    flop_match = _FLOP_RE.search(instruction)
    if flop_match:
        for g in (1, 2, 3):
            card = _parse_card_text(flop_match.group(g))
            if card:
                board.append(card)
    turn_match = _TURN_RE.search(instruction)
    if turn_match:
        card = _parse_card_text(turn_match.group(1))
        if card:
            board.append(card)
    river_match = _RIVER_RE.search(instruction)
    if river_match:
        card = _parse_card_text(river_match.group(1))
        if card:
            board.append(card)

    # Pot
    pot_match = _POT_RE.search(instruction)
    pot = float(pot_match.group(1)) if pot_match else 0.0

    # Street
    street = _STREET_FROM_BOARD_SIZE.get(len(board), "pre_flop")

    # Position → IP/OOP (simplified: BTN/CO/HJ are IP vs BB/SB)
    is_ip = position in _POSITION_IP

    # Estimate to_call from action text (simplified heuristic)
    # Look at the last action before "Now it is your turn"
    to_call = 0.0
    turn_marker = instruction.find("Now it is your turn")
    if turn_marker > 0:
        before_turn = instruction[:turn_marker]
        # Find the current street section
        street_sections = re.split(r"(?:The flop comes|The turn comes|The river comes)", before_turn)
        current_section = street_sections[-1] if street_sections else before_turn
        # Parse actions in this street section
        actions_in_section = _ACTION_RE.findall(current_section)
        last_bet_amount = 0.0
        hero_invested_this_street = 0.0
        for actor, action, amount_str in actions_in_section:
            amount = float(amount_str) if amount_str else 0.0
            if actor.upper() == position:
                if action in ("bet", "raise", "call"):
                    hero_invested_this_street += amount
            else:
                if action in ("bet", "raise"):
                    last_bet_amount = amount
        to_call = max(0.0, last_bet_amount - hero_invested_this_street)

    # Derived features
    pot_odds = (to_call / (pot + to_call)) if to_call > 0 else 0.0
    effective_stack = starting_stack  # approximate
    spr = (effective_stack / pot) if pot > 0 else 999.0

    hole_class = _canonical_hole(hole_cards)
    texture = _board_texture(board)

    # Legal actions (PokerBench always has these available)
    legal_actions = ["fold", "check", "call", "bet", "raise", "all_in"]
    if to_call > 0:
        legal_actions = ["fold", "call", "raise", "all_in"]
    else:
        legal_actions = ["check", "bet", "raise", "all_in"]

    return {
        "position": position,
        "hole_cards": hole_cards,
        "hole_class": hole_class,
        "board": board,
        "board_texture": texture,
        "street": street,
        "pot": pot,
        "to_call": to_call,
        "pot_odds": round(pot_odds, 4),
        "spr": round(spr, 4),
        "effective_stack": effective_stack,
        "is_ip": is_ip,
        "legal_actions": sorted(legal_actions),
        "starting_stack": starting_stack,
    }


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------


class PokerBenchAdapter:
    """Converts PokerBench rows into mr_poker DatasetBuilder-compatible rows.

    Each PokerBench row has ``instruction`` (game state text) and ``output``
    (solver-optimal action).  The adapter parses these, extracts features,
    computes bucketing, and produces rows in the same JSONL schema used by
    :class:`~packages.dataset_builder.builder.DatasetBuilder`.
    """

    def __init__(self, *, equity_samples: int = 200, equity_seed: int | None = 42) -> None:
        self._equity_samples = equity_samples
        self._equity_seed = equity_seed

    def convert_row(
        self,
        instruction: str,
        output: str,
        *,
        row_index: int = 0,
        config: str = "default",
    ) -> dict[str, Any] | None:
        """Convert a single PokerBench row to a DatasetBuilder-compatible dict.

        Returns ``None`` if the instruction cannot be parsed.
        """
        parsed = parse_pokerbench_instruction(instruction)
        if parsed is None:
            return None

        action_type, action_amount = _parse_action_output(output)

        # Estimate equity if possible
        estimated_equity = self._estimate_equity(
            parsed["hole_cards"], parsed["board"]
        )

        # Build spot dict for bucketing
        spot = {
            "street": parsed["street"],
            "hole_class": parsed["hole_class"],
            "board_texture": parsed["board_texture"],
            "pot": parsed["pot"],
            "to_call": parsed["to_call"],
            "pot_odds": parsed["pot_odds"],
            "spr": parsed["spr"],
            "estimated_equity": estimated_equity,
            "actor_seat": 0 if parsed["is_ip"] else 1,
            "button_seat": 0,
        }

        bucket_info = bucketize_spot(spot)

        # Build pseudo-trace for taxonomy classification
        pseudo_trace = {
            "street": parsed["street"],
            "action_type": action_type,
            "hole_class": parsed["hole_class"],
            "board_texture": parsed["board_texture"],
            "to_call": parsed["to_call"],
            "estimated_equity": estimated_equity,
            "pot_odds": parsed["pot_odds"],
        }
        taxonomy_tags = classify_trace(pseudo_trace)

        source_key = f"pokerbench:{config}:{row_index}"

        return {
            "row_id": str(uuid4()),
            "source_type": "pokerbench",
            "source_key": source_key,
            "session_id": None,
            "hand_id": None,
            "actor_seat": 0 if parsed["is_ip"] else 1,
            "street": parsed["street"],
            "hole_class": parsed["hole_class"],
            "board_texture": parsed["board_texture"],
            "pot": parsed["pot"],
            "to_call": parsed["to_call"],
            "pot_odds": parsed["pot_odds"],
            "estimated_equity": estimated_equity,
            "legal_actions": parsed["legal_actions"],
            "bucket_info": bucket_info,
            "label_action": action_type,
            "label_amount": action_amount,
            "label_confidence": 0.95,
            "label_source": "pokerbench_solver",
            "taxonomy_tags": taxonomy_tags,
            "split": stable_split(source_key),
            "pokerbench_position": parsed["position"],
            "pokerbench_config": config,
        }

    def convert_batch(
        self,
        rows: list[dict[str, str]],
        *,
        config: str = "default",
    ) -> list[dict[str, Any]]:
        """Convert a batch of PokerBench rows.

        Each item in *rows* must have ``instruction`` and ``output`` keys.
        Rows that cannot be parsed are silently skipped (logged as warnings).
        """
        results: list[dict[str, Any]] = []
        skipped = 0
        for i, row in enumerate(rows):
            converted = self.convert_row(
                row["instruction"],
                row["output"],
                row_index=i,
                config=config,
            )
            if converted is not None:
                results.append(converted)
            else:
                skipped += 1

        if skipped > 0:
            logger.warning(
                "PokerBench conversion: %d/%d rows skipped (unparseable)",
                skipped,
                len(rows),
            )
        logger.info(
            "PokerBench conversion: %d rows converted from config=%s",
            len(results),
            config,
        )
        return results

    def _estimate_equity(self, hole_cards: list[str], board: list[str]) -> float:
        """Estimate equity using the project's Monte Carlo estimator."""
        try:
            from packages.engine.models import Card
            from packages.equity import estimate_equity

            hero = [Card.from_str(c) for c in hole_cards]
            board_cards = [Card.from_str(c) for c in board]
            return round(
                estimate_equity(
                    hero,
                    board_cards,
                    samples=self._equity_samples,
                    seed=self._equity_seed,
                ),
                4,
            )
        except Exception:
            logger.debug("Equity estimation failed, using heuristic fallback")
            return self._heuristic_equity(hole_cards, board)

    @staticmethod
    def _heuristic_equity(hole_cards: list[str], board: list[str]) -> float:
        """Simple heuristic equity when Monte Carlo is unavailable."""
        rank_values = {r: i for i, r in enumerate("23456789TJQKA", start=2)}
        r1 = rank_values.get(hole_cards[0][0], 5)
        r2 = rank_values.get(hole_cards[1][0], 5)
        high = max(r1, r2)
        suited = hole_cards[0][1] == hole_cards[1][1]
        paired = r1 == r2

        base = 0.35
        if paired:
            base = 0.55 + (high - 8) * 0.02
        elif high >= 12:
            base = 0.45 + (high - 12) * 0.03
        if suited:
            base += 0.03
        if len(board) > 0:
            base *= 0.9  # discount for more info available
        return round(min(0.95, max(0.15, base)), 4)
