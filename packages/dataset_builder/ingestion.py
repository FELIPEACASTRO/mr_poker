"""Dataset ingestion pipeline for external poker datasets.

Supports:
- PHH format (Poker Hand History - uoftcprg/phh-dataset)
- PokerBench (RZ412/PokerBench from HuggingFace)
- Nemotron-Personas (nvidia/Nemotron-Personas-Singapore)
- PokerStars hand histories (text format)
- Generic CSV/JSONL with poker decision features

Reference: docs/106_Analise_Estrategica_Avancada.md Section 1.1
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from packages.common.types import ActionType, Street

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants and mappings
# ---------------------------------------------------------------------------

_ACTION_TYPE_FLOAT: dict[str, float] = {
    "fold": 0.0,
    "check": 0.2,
    "call": 0.4,
    "bet": 0.6,
    "raise": 0.8,
    "all_in": 1.0,
}

_STREET_FLOAT: dict[str, float] = {
    "pre_flop": 0.0,
    "preflop": 0.0,
    "flop": 0.33,
    "turn": 0.67,
    "river": 1.0,
}

_STREET_FROM_BOARD_LEN: dict[int, str] = {
    0: "pre_flop",
    3: "flop",
    4: "turn",
    5: "river",
}

_POSITION_NAMES: list[str] = ["SB", "BB", "UTG", "MP", "CO", "BTN"]

_RANK_CHARS = "23456789TJQKA"
_RANK_VALUES: dict[str, int] = {r: i for i, r in enumerate(_RANK_CHARS, start=2)}


def _stable_split(key: str) -> str:
    """Deterministic train/val/test split (70/15/15) from a string key."""
    value = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    if value < 70:
        return "train"
    if value < 85:
        return "validation"
    return "test"


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _safe_float(v: Any, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _canonical_hole(cards: list[str]) -> str:
    """Derive canonical hole class from two card strings like ['Kh', '3h']."""
    if len(cards) != 2:
        return "??"
    r1 = _RANK_VALUES.get(cards[0][0], 0)
    r2 = _RANK_VALUES.get(cards[1][0], 0)
    if r1 < r2:
        r1, r2 = r2, r1
        cards = [cards[1], cards[0]]
    h, l = cards[0][0], cards[1][0]
    if h == l:
        return f"{h}{l}"
    suited = "s" if cards[0][1] == cards[1][1] else "o"
    return f"{h}{l}{suited}"


def _board_texture(board: list[str]) -> str:
    """Classify board texture from card strings."""
    if len(board) < 3:
        return "preflop"
    suits = [c[1] for c in board if len(c) >= 2]
    ranks = sorted(_RANK_VALUES.get(c[0], 0) for c in board if len(c) >= 2)
    unique_suits = len(set(suits))
    paired = len(set(ranks)) < len(ranks)
    span = max(ranks) - min(ranks) if ranks else 0
    connectivity = "connected" if span <= 4 else "disconnected"
    suit_label = (
        "monotone" if unique_suits == 1
        else "two-tone" if unique_suits == 2
        else "rainbow"
    )
    paired_label = "paired" if paired else "unpaired"
    return f"{suit_label}-{paired_label}-{connectivity}"


# ---------------------------------------------------------------------------
# IngestionResult
# ---------------------------------------------------------------------------


@dataclass
class IngestionResult:
    """Result of a dataset ingestion operation."""

    source_path: str = ""
    source_type: str = ""
    total_records: int = 0
    valid_records: int = 0
    skipped_records: int = 0
    output_path: str = ""
    manifest_path: str = ""
    feature_stats: dict[str, dict[str, float]] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary of the ingestion."""
        skip_pct = (
            f" ({self.skipped_records / self.total_records * 100:.1f}% skipped)"
            if self.total_records > 0
            else ""
        )
        return (
            f"Ingested {self.valid_records}/{self.total_records} records "
            f"from {self.source_type}{skip_pct} -> {self.output_path}"
        )


# ---------------------------------------------------------------------------
# PHHIngester
# ---------------------------------------------------------------------------


_PHH_SUIT_MAP: dict[str, str] = {"c": "c", "d": "d", "h": "h", "s": "s"}

# PHH actions: f=fold, cc=check/call, cbr=bet/raise with amount
_PHH_ACTION_RE = re.compile(r"^(f|cc|cb|cr|p[bs]?)(\d*)$", re.IGNORECASE)


class PHHIngester:
    """Ingester for Poker Hand History (PHH) format files.

    PHH is a structured plain-text format used by uoftcprg/phh-dataset.
    Each file describes a single hand with variant, antes, blinds, stacks,
    and per-street action sequences.

    Supports both ``.phh`` (single hand) and ``.jsonl`` (one JSON-encoded
    PHH object per line) variants.
    """

    def ingest_phh_file(self, path: str) -> list[dict[str, Any]]:
        """Parse a PHH file and return internal-format records.

        Args:
            path: Path to a ``.phh`` or ``.jsonl`` file.

        Returns:
            List of decision-record dicts ready for the pipeline.
        """
        p = Path(path)
        if not p.exists():
            logger.warning("PHH file not found: %s", path)
            return []

        if p.suffix == ".jsonl":
            return self._ingest_jsonl(p)
        return self._ingest_single(p)

    # -- internal ----------------------------------------------------------

    def _ingest_jsonl(self, path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("PHH JSONL: skip malformed line %d", line_no)
                continue
            records.extend(self._parse_phh_object(obj, f"{path.name}:{line_no}"))
        return records

    def _ingest_single(self, path: Path) -> list[dict[str, Any]]:
        """Parse a single .phh plain-text file."""
        text = path.read_text(encoding="utf-8")
        obj = self._parse_phh_text(text)
        return self._parse_phh_object(obj, path.name)

    @staticmethod
    def _parse_phh_text(text: str) -> dict[str, Any]:
        """Parse PHH plain-text key:value format into a dict."""
        result: dict[str, Any] = {}
        current_key = ""
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line and not line.startswith(" "):
                key, _, value = line.partition(":")
                key = key.strip().lower()
                value = value.strip()
                current_key = key
                # Try to parse lists/numbers
                if value.startswith("["):
                    try:
                        result[key] = json.loads(value)
                    except json.JSONDecodeError:
                        result[key] = value
                else:
                    result[key] = value
            elif current_key:
                # Continuation line
                existing = result.get(current_key, "")
                result[current_key] = f"{existing} {line}".strip()
        return result

    def _parse_phh_object(
        self, obj: dict[str, Any], source_label: str
    ) -> list[dict[str, Any]]:
        """Convert a parsed PHH object into decision records."""
        records: list[dict[str, Any]] = []

        hand_id = obj.get("hand") or obj.get("id") or source_label
        num_players = int(obj.get("players", obj.get("num_players", 2)))
        stacks = obj.get("stacks") or obj.get("starting_stacks", [])
        if isinstance(stacks, str):
            try:
                stacks = json.loads(stacks)
            except json.JSONDecodeError:
                stacks = []
        stacks = [_safe_float(s) for s in stacks]

        blinds = obj.get("blinds") or obj.get("blind_amounts", [])
        if isinstance(blinds, str):
            try:
                blinds = json.loads(blinds)
            except json.JSONDecodeError:
                blinds = []
        blinds = [_safe_float(b) for b in blinds]
        big_blind = max(blinds) if blinds else 1.0

        # Board cards
        board_str = obj.get("board", "")
        board = self._parse_card_list(board_str) if board_str else []

        # Parse actions per street
        streets_data = self._extract_streets(obj)
        pot = sum(blinds) if blinds else 0.0

        for street_name, actions_raw in streets_data:
            street_board = self._board_for_street(board, street_name)
            actions = self._parse_action_list(actions_raw, num_players)

            for action_idx, (seat, action_type, amount) in enumerate(actions):
                to_call = self._compute_to_call(actions[:action_idx], seat)
                pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.0
                stack = stacks[seat] if seat < len(stacks) else 100.0
                pot_committed = (stack - max(0, stack - amount)) / stack if stack > 0 else 0.0

                # Position score: later seats are more IP
                position_score = seat / max(1, num_players - 1)

                source_key = f"phh:{hand_id}:{street_name}:{action_idx}"
                record = {
                    "row_id": str(uuid4()),
                    "source_type": "phh",
                    "source_key": source_key,
                    "hand_id": str(hand_id),
                    "actor_seat": seat,
                    "street": street_name,
                    "hole_class": None,
                    "board_texture": _board_texture(street_board),
                    "board": street_board,
                    "pot": round(pot, 2),
                    "to_call": round(to_call, 2),
                    "pot_odds": round(pot_odds, 4),
                    "action_type": action_type,
                    "amount": round(amount, 2),
                    "bet_fraction": round(amount / pot, 4) if pot > 0 else 0.0,
                    "position_score": round(position_score, 4),
                    "pot_committed": round(pot_committed, 4),
                    "big_blind": big_blind,
                    "num_players": num_players,
                    "split": _stable_split(source_key),
                }
                records.append(record)

                # Update pot
                if action_type in ("call", "bet", "raise", "all_in"):
                    pot += amount

        return records

    def _extract_streets(
        self, obj: dict[str, Any]
    ) -> list[tuple[str, Any]]:
        """Extract per-street action data from PHH object."""
        streets: list[tuple[str, Any]] = []

        # Try structured keys first
        for key, street_name in [
            ("preflop_actions", "pre_flop"),
            ("flop_actions", "flop"),
            ("turn_actions", "turn"),
            ("river_actions", "river"),
        ]:
            if key in obj:
                streets.append((street_name, obj[key]))

        # Try combined 'actions' key (list of lists per street)
        if not streets and "actions" in obj:
            actions_val = obj["actions"]
            if isinstance(actions_val, str):
                try:
                    actions_val = json.loads(actions_val)
                except json.JSONDecodeError:
                    actions_val = [actions_val]
            if isinstance(actions_val, list):
                street_names = ["pre_flop", "flop", "turn", "river"]
                if actions_val and isinstance(actions_val[0], list):
                    for i, street_actions in enumerate(actions_val):
                        if i < len(street_names):
                            streets.append((street_names[i], street_actions))
                else:
                    # Single flat list → all pre_flop
                    streets.append(("pre_flop", actions_val))

        return streets

    @staticmethod
    def _parse_action_list(
        actions_raw: Any, num_players: int
    ) -> list[tuple[int, str, float]]:
        """Parse PHH action tokens into (seat, action_type, amount) tuples."""
        results: list[tuple[int, str, float]] = []
        if isinstance(actions_raw, str):
            tokens = actions_raw.replace(",", " ").split()
        elif isinstance(actions_raw, list):
            tokens = [str(t) for t in actions_raw]
        else:
            return results

        seat_idx = 0
        for token in tokens:
            token = token.strip().lower()
            if not token:
                continue
            m = _PHH_ACTION_RE.match(token)
            if m:
                code = m.group(1).lower()
                amt_str = m.group(2)
                amount = _safe_float(amt_str, 0.0)
                seat = seat_idx % num_players

                if code == "f":
                    results.append((seat, "fold", 0.0))
                elif code == "cc":
                    results.append((seat, "call" if amount > 0 else "check", amount))
                elif code in ("cb", "cr"):
                    results.append((seat, "raise" if code == "cr" else "bet", amount))
                elif code.startswith("p"):
                    # Posting blinds — skip for decision records
                    pass

                seat_idx += 1
            elif token in ("fold", "check", "call", "bet", "raise", "all_in", "all-in"):
                seat = seat_idx % num_players
                action = token.replace("-", "_")
                results.append((seat, action, 0.0))
                seat_idx += 1

        return results

    @staticmethod
    def _compute_to_call(
        prior_actions: list[tuple[int, str, float]], hero_seat: int
    ) -> float:
        """Estimate the amount the hero must call based on prior actions."""
        last_bet = 0.0
        hero_invested = 0.0
        for seat, action, amount in prior_actions:
            if seat == hero_seat:
                if action in ("call", "bet", "raise"):
                    hero_invested += amount
            else:
                if action in ("bet", "raise"):
                    last_bet = amount
        return max(0.0, last_bet - hero_invested)

    @staticmethod
    def _parse_card_list(val: Any) -> list[str]:
        """Parse various card-list representations into ['Ah', 'Kd', ...] form."""
        if isinstance(val, list):
            return [str(c) for c in val if c]
        if isinstance(val, str):
            val = val.strip().strip("[]")
            if not val:
                return []
            # Try JSON
            try:
                parsed = json.loads(f"[{val}]")
                return [str(c) for c in parsed]
            except json.JSONDecodeError:
                pass
            # Space/comma separated
            return [c.strip().strip("'\"") for c in re.split(r"[,\s]+", val) if c.strip()]
        return []

    @staticmethod
    def _board_for_street(board: list[str], street: str) -> list[str]:
        """Return the board cards visible on a given street."""
        if street == "pre_flop":
            return []
        if street == "flop":
            return board[:3]
        if street == "turn":
            return board[:4]
        return board[:5]


# ---------------------------------------------------------------------------
# PokerBenchIngester
# ---------------------------------------------------------------------------


class PokerBenchIngester:
    """Ingester for PokerBench (RZ412/PokerBench) dataset files.

    PokerBench contains solver-computed optimal decisions for 6-handed NLHE.
    Each row has an ``instruction`` (natural-language game state) and an
    ``output`` (optimal action).

    Supports both SFT (supervised fine-tuning) and GRPO (RL) output formats.
    """

    # Re-use card/position parsing from pokerbench_adapter
    _RANK_WORDS: dict[str, str] = {
        "ace": "A", "king": "K", "queen": "Q", "jack": "J", "ten": "T",
        "nine": "9", "eight": "8", "seven": "7", "six": "6", "five": "5",
        "four": "4", "three": "3", "two": "2", "deuce": "2",
    }
    _SUIT_WORDS: dict[str, str] = {
        "heart": "h", "hearts": "h", "diamond": "d", "diamonds": "d",
        "club": "c", "clubs": "c", "spade": "s", "spades": "s",
    }

    _HOLDING_RE = re.compile(
        r"\[([A-Za-z]+ of [A-Za-z]+) and ([A-Za-z]+ of [A-Za-z]+)\]"
    )
    _POSITION_RE = re.compile(r"your position is (\w+)")
    _POT_RE = re.compile(r"current pot size is ([\d.]+) chips")
    _STACK_RE = re.compile(r"Everyone started with (\d+) chips")
    _FLOP_RE = re.compile(
        r"[Tt]he flop comes ([A-Za-z]+ [Oo]f [A-Za-z]+),\s*"
        r"([A-Za-z]+ [Oo]f [A-Za-z]+),\s*and ([A-Za-z]+ [Oo]f [A-Za-z]+)"
    )
    _TURN_RE = re.compile(r"[Tt]he turn comes ([A-Za-z]+ [Oo]f [A-Za-z]+)")
    _RIVER_RE = re.compile(r"[Tt]he river comes ([A-Za-z]+ [Oo]f [A-Za-z]+)")

    _IP_POSITIONS: frozenset[str] = frozenset({"BTN", "CO", "HJ"})

    def ingest_pokerbench(self, path: str) -> list[dict[str, Any]]:
        """Parse PokerBench JSONL or CSV file into internal records.

        Args:
            path: Path to a JSONL or CSV file with ``instruction`` and
                ``output`` columns.

        Returns:
            List of decision-record dicts.
        """
        p = Path(path)
        if not p.exists():
            logger.warning("PokerBench file not found: %s", path)
            return []

        rows = self._load_rows(p)
        records: list[dict[str, Any]] = []

        for idx, row in enumerate(rows):
            instruction = row.get("instruction", "")
            output = row.get("output", "")
            if not instruction or not output:
                continue

            parsed = self._parse_instruction(instruction)
            if parsed is None:
                continue

            action_type, action_amount = self._parse_output(output)
            source_key = f"pokerbench:{p.stem}:{idx}"

            pot = parsed["pot"]
            to_call = parsed["to_call"]
            pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.0
            is_ip = parsed["position"] in self._IP_POSITIONS

            record = {
                "row_id": str(uuid4()),
                "source_type": "pokerbench",
                "source_key": source_key,
                "hand_id": None,
                "actor_seat": 0 if is_ip else 1,
                "street": parsed["street"],
                "hole_cards": parsed["hole_cards"],
                "hole_class": _canonical_hole(parsed["hole_cards"]),
                "board": parsed["board"],
                "board_texture": _board_texture(parsed["board"]),
                "pot": round(pot, 2),
                "to_call": round(to_call, 2),
                "pot_odds": round(pot_odds, 4),
                "action_type": action_type,
                "amount": action_amount,
                "bet_fraction": (
                    round(action_amount / pot, 4) if pot > 0 and action_amount > 0
                    else 0.0
                ),
                "position_score": 1.0 if is_ip else 0.0,
                "position": parsed["position"],
                "starting_stack": parsed["starting_stack"],
                "label_action": action_type,
                "label_amount": action_amount,
                "label_confidence": 0.95,
                "label_source": "pokerbench_solver",
                "format": row.get("format", "sft"),
                "split": _stable_split(source_key),
            }

            # GRPO-specific fields
            if "reward" in row:
                record["reward"] = _safe_float(row["reward"])
            if "chosen" in row:
                record["chosen"] = row["chosen"]
            if "rejected" in row:
                record["rejected"] = row["rejected"]

            records.append(record)

        return records

    # -- internal ----------------------------------------------------------

    @staticmethod
    def _load_rows(path: Path) -> list[dict[str, str]]:
        """Load rows from JSONL or CSV."""
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            rows: list[dict[str, str]] = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return rows
        elif path.suffix == ".csv":
            reader = csv.DictReader(text.splitlines())
            return list(reader)
        else:
            # Try JSONL first, then CSV
            try:
                rows = []
                for line in text.splitlines():
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
                return rows
            except json.JSONDecodeError:
                reader = csv.DictReader(text.splitlines())
                return list(reader)

    def _parse_card_text(self, text: str) -> str | None:
        """Convert 'King of Hearts' -> 'Kh'."""
        parts = text.strip().lower().replace(" of ", " ").split()
        if len(parts) != 2:
            return None
        rank = self._RANK_WORDS.get(parts[0])
        suit = self._SUIT_WORDS.get(parts[1])
        if rank is None or suit is None:
            return None
        return f"{rank}{suit}"

    def _parse_instruction(self, instruction: str) -> dict[str, Any] | None:
        """Extract structured game state from PokerBench instruction text."""
        pos_match = self._POSITION_RE.search(instruction)
        if not pos_match:
            return None
        position = pos_match.group(1).upper()

        holding_match = self._HOLDING_RE.search(instruction)
        if not holding_match:
            return None
        card1 = self._parse_card_text(holding_match.group(1))
        card2 = self._parse_card_text(holding_match.group(2))
        if card1 is None or card2 is None:
            return None
        hole_cards = [card1, card2]

        stack_match = self._STACK_RE.search(instruction)
        starting_stack = int(stack_match.group(1)) if stack_match else 100

        board: list[str] = []
        flop_match = self._FLOP_RE.search(instruction)
        if flop_match:
            for g in (1, 2, 3):
                c = self._parse_card_text(flop_match.group(g))
                if c:
                    board.append(c)
        turn_match = self._TURN_RE.search(instruction)
        if turn_match:
            c = self._parse_card_text(turn_match.group(1))
            if c:
                board.append(c)
        river_match = self._RIVER_RE.search(instruction)
        if river_match:
            c = self._parse_card_text(river_match.group(1))
            if c:
                board.append(c)

        pot_match = self._POT_RE.search(instruction)
        pot = float(pot_match.group(1)) if pot_match else 0.0

        street = _STREET_FROM_BOARD_LEN.get(len(board), "pre_flop")

        # Estimate to_call from actions before hero's turn
        to_call = 0.0
        turn_marker = instruction.find("Now it is your turn")
        if turn_marker > 0:
            before_turn = instruction[:turn_marker]
            street_sections = re.split(
                r"(?:The flop comes|The turn comes|The river comes)", before_turn
            )
            current_section = street_sections[-1] if street_sections else before_turn
            action_re = re.compile(
                r"(\w+) (fold|check|call|bet|raise|all-in|all in)"
                r"(?: ([\d.]+) chips)?"
            )
            last_bet = 0.0
            hero_invested = 0.0
            for actor, action, amt_str in action_re.findall(current_section):
                amount = float(amt_str) if amt_str else 0.0
                if actor.upper() == position:
                    if action in ("bet", "raise", "call"):
                        hero_invested += amount
                else:
                    if action in ("bet", "raise"):
                        last_bet = amount
            to_call = max(0.0, last_bet - hero_invested)

        return {
            "position": position,
            "hole_cards": hole_cards,
            "board": board,
            "street": street,
            "pot": pot,
            "to_call": to_call,
            "starting_stack": starting_stack,
        }

    @staticmethod
    def _parse_output(output: str) -> tuple[str, float]:
        """Parse PokerBench output like 'check', 'fold', 'bet 10', 'raise 64'."""
        output = output.strip().lower()
        if output in ("check", "fold", "call"):
            return output, 0.0
        parts = output.split()
        if len(parts) == 2 and parts[0] in ("bet", "raise", "call"):
            return parts[0], _safe_float(parts[1])
        if "all" in output:
            return "all_in", 0.0
        return output, 0.0


# ---------------------------------------------------------------------------
# NemotronPersonasIngester
# ---------------------------------------------------------------------------


class NemotronPersonasIngester:
    """Ingester for Nemotron-Personas (nvidia/Nemotron-Personas-Singapore).

    Extracts demographic and personality fields from conversation-style
    persona records and maps them to MR_POKER's PersonalityProfile
    (Big Five traits) and SyntheticPlayerStats archetype templates.
    """

    # Keyword-to-trait mappings for inferring Big Five from persona text
    _OPENNESS_KEYWORDS: list[str] = [
        "creative", "curious", "adventurous", "imaginative", "open-minded",
        "unconventional", "artistic", "experimental", "innovative",
    ]
    _CONSCIENTIOUSNESS_KEYWORDS: list[str] = [
        "disciplined", "organized", "meticulous", "careful", "thorough",
        "responsible", "reliable", "diligent", "methodical", "systematic",
    ]
    _EXTRAVERSION_KEYWORDS: list[str] = [
        "outgoing", "energetic", "talkative", "assertive", "sociable",
        "enthusiastic", "bold", "confident", "dominant", "aggressive",
    ]
    _AGREEABLENESS_KEYWORDS: list[str] = [
        "kind", "cooperative", "trusting", "helpful", "empathetic",
        "compassionate", "friendly", "accommodating", "patient", "gentle",
    ]
    _NEUROTICISM_KEYWORDS: list[str] = [
        "anxious", "nervous", "emotional", "impulsive", "moody",
        "sensitive", "stressed", "volatile", "reactive", "temperamental",
    ]

    # Archetype detection keywords
    _ARCHETYPE_SIGNALS: dict[str, list[str]] = {
        "nit": ["cautious", "conservative", "risk-averse", "careful", "patient"],
        "tag": ["selective", "aggressive", "strategic", "disciplined", "focused"],
        "lag": ["aggressive", "creative", "unpredictable", "bold", "loose"],
        "calling_station": [
            "curious", "passive", "patient", "trusting", "accommodating",
        ],
        "maniac": [
            "impulsive", "reckless", "aggressive", "wild", "unpredictable",
        ],
    }

    def ingest_personas(self, path: str) -> list[dict[str, Any]]:
        """Parse Nemotron-Personas JSONL file into personality records.

        Args:
            path: Path to a JSONL file with persona conversation records.

        Returns:
            List of personality-profile dicts with Big Five traits and
            archetype mappings.
        """
        p = Path(path)
        if not p.exists():
            logger.warning("Nemotron-Personas file not found: %s", path)
            return []

        records: list[dict[str, Any]] = []
        for line_no, line in enumerate(
            p.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Nemotron: skip malformed line %d", line_no)
                continue

            record = self._convert_persona(obj, line_no)
            if record is not None:
                records.append(record)

        return records

    def _convert_persona(
        self, obj: dict[str, Any], line_no: int
    ) -> dict[str, Any] | None:
        """Convert a single persona object to a personality/archetype record."""
        # Extract text content for trait inference
        persona_text = self._extract_text(obj)
        if not persona_text:
            return None

        text_lower = persona_text.lower()

        # Infer Big Five traits
        openness = self._score_trait(text_lower, self._OPENNESS_KEYWORDS)
        conscientiousness = self._score_trait(
            text_lower, self._CONSCIENTIOUSNESS_KEYWORDS
        )
        extraversion = self._score_trait(text_lower, self._EXTRAVERSION_KEYWORDS)
        agreeableness = self._score_trait(text_lower, self._AGREEABLENESS_KEYWORDS)
        neuroticism = self._score_trait(text_lower, self._NEUROTICISM_KEYWORDS)

        # Detect archetype
        archetype = self._detect_archetype(text_lower)

        # Map personality to poker stats
        stats = self._personality_to_stats(
            openness, conscientiousness, extraversion, agreeableness, neuroticism,
            archetype,
        )

        # Extract demographics if available
        demographics = self._extract_demographics(obj)

        source_key = f"nemotron:{line_no}"
        return {
            "row_id": str(uuid4()),
            "source_type": "nemotron_personas",
            "source_key": source_key,
            "personality": {
                "openness": round(openness, 4),
                "conscientiousness": round(conscientiousness, 4),
                "extraversion": round(extraversion, 4),
                "agreeableness": round(agreeableness, 4),
                "neuroticism": round(neuroticism, 4),
            },
            "archetype": archetype,
            "poker_stats": stats,
            "demographics": demographics,
            "persona_text_preview": persona_text[:200],
            "split": _stable_split(source_key),
        }

    @staticmethod
    def _extract_text(obj: dict[str, Any]) -> str:
        """Extract the main text content from a persona record."""
        # Nemotron-Personas may have conversations or direct persona descriptions
        if "conversations" in obj:
            parts: list[str] = []
            for turn in obj["conversations"]:
                val = turn.get("value", turn.get("content", ""))
                if val:
                    parts.append(str(val))
            return " ".join(parts)
        if "persona" in obj:
            return str(obj["persona"])
        if "description" in obj:
            return str(obj["description"])
        if "text" in obj:
            return str(obj["text"])
        # Fallback: concatenate all string values
        parts = [str(v) for v in obj.values() if isinstance(v, str)]
        return " ".join(parts)

    @staticmethod
    def _score_trait(text: str, keywords: list[str]) -> float:
        """Score a personality trait based on keyword presence (0.0-1.0)."""
        if not text:
            return 0.5
        hits = sum(1 for kw in keywords if kw in text)
        # Sigmoid-like mapping: 0 hits -> 0.3, 3+ hits -> 0.9
        raw = 0.3 + 0.6 * (1.0 - math.exp(-hits * 0.7))
        return _clamp(raw)

    def _detect_archetype(self, text: str) -> str:
        """Detect the best-matching poker archetype from persona text."""
        best_archetype = "tag"  # default
        best_score = 0
        for archetype, keywords in self._ARCHETYPE_SIGNALS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > best_score:
                best_score = score
                best_archetype = archetype
        return best_archetype

    @staticmethod
    def _personality_to_stats(
        openness: float,
        conscientiousness: float,
        extraversion: float,
        agreeableness: float,
        neuroticism: float,
        archetype: str,
    ) -> dict[str, float]:
        """Map Big Five personality traits to poker statistics.

        Uses trait-to-behavior mappings inspired by psychological research:
        - Openness -> creative/unconventional plays, bluff frequency
        - Conscientiousness -> disciplined play, GTO adherence
        - Extraversion -> aggression, action-seeking
        - Agreeableness -> passive play, calling frequency
        - Neuroticism -> tilt propensity, timing variance
        """
        # Base VPIP: extraversion increases, conscientiousness decreases
        vpip = _clamp(0.20 + extraversion * 0.15 - conscientiousness * 0.08
                       + agreeableness * 0.05)
        # PFR: extraversion and openness increase, agreeableness decreases
        pfr = _clamp(0.15 + extraversion * 0.12 + openness * 0.05
                      - agreeableness * 0.08)
        # 3-bet: openness + extraversion drive aggression
        three_bet = _clamp(0.04 + extraversion * 0.06 + openness * 0.04
                           - agreeableness * 0.03)
        # Fold to c-bet: agreeableness increases, conscientiousness decreases
        fold_to_cbet = _clamp(0.45 + agreeableness * 0.10
                              - conscientiousness * 0.08 - openness * 0.05)
        # Aggression factor
        aggression_factor = _clamp(
            1.5 + extraversion * 2.0 + openness * 0.5 - agreeableness * 1.0,
            lo=0.5, hi=5.0,
        )
        # WTSD: agreeableness (calling) increases, conscientiousness decreases
        wtsd = _clamp(0.25 + agreeableness * 0.10 - conscientiousness * 0.05
                       + neuroticism * 0.03)
        # C-bet frequency: conscientiousness + extraversion
        cbet_freq = _clamp(0.55 + conscientiousness * 0.10 + extraversion * 0.08
                           - agreeableness * 0.05)
        # Tilt propensity from neuroticism
        tilt_propensity = _clamp(neuroticism * 0.8 + (1.0 - conscientiousness) * 0.2)

        return {
            "vpip": round(vpip, 4),
            "pfr": round(pfr, 4),
            "three_bet": round(three_bet, 4),
            "fold_to_cbet": round(fold_to_cbet, 4),
            "aggression_factor": round(aggression_factor, 2),
            "wtsd": round(wtsd, 4),
            "cbet_freq": round(cbet_freq, 4),
            "archetype": archetype,
            "tilt_propensity": round(tilt_propensity, 4),
        }

    @staticmethod
    def _extract_demographics(obj: dict[str, Any]) -> dict[str, Any]:
        """Extract demographic fields if present in the persona record."""
        demo: dict[str, Any] = {}
        for key in ("age", "gender", "location", "occupation", "education",
                     "nationality", "language", "region"):
            if key in obj:
                demo[key] = obj[key]
        return demo


# ---------------------------------------------------------------------------
# PokerStarsIngester
# ---------------------------------------------------------------------------


class PokerStarsIngester:
    """Ingester for PokerStars hand history text files.

    Parses the standard PokerStars hand history format including:
    - Hand headers (``PokerStars Hand #NNNNN``)
    - Table and seat assignments
    - Blind postings
    - Per-street actions (folds, calls, raises, bets, checks, all-in)
    - Board cards and showdown info
    """

    _HAND_RE = re.compile(
        r"PokerStars (?:Zoom )?Hand #(\d+).*"
    )
    _TABLE_RE = re.compile(r"Table '([^']+)'")
    _SEAT_RE = re.compile(
        r"Seat (\d+): (\S+) \((\d+(?:\.\d+)?) in chips\)"
    )
    _BLIND_RE = re.compile(
        r"(\S+): posts (?:small|big) blind (\d+(?:\.\d+)?)"
    )
    _ACTION_RE = re.compile(
        r"(\S+): (folds|checks|calls|bets|raises)(?: (\d+(?:\.\d+)?))?"
        r"(?: to (\d+(?:\.\d+)?))?"
        r"( and is all-in)?"
    )
    _BOARD_RE = re.compile(r"\[([2-9TJQKAcdhs ]+)\]")
    _HOLE_RE = re.compile(
        r"Dealt to (\S+) \[([2-9TJQKAcdhs ]+)\]"
    )
    _SHOWDOWN_RE = re.compile(
        r"(\S+): shows \[([2-9TJQKAcdhs ]+)\]"
    )

    _STREET_MARKERS: list[tuple[str, str]] = [
        ("*** HOLE CARDS ***", "pre_flop"),
        ("*** FLOP ***", "flop"),
        ("*** TURN ***", "turn"),
        ("*** RIVER ***", "river"),
        ("*** SHOW DOWN ***", "showdown"),
    ]

    def ingest_pokerstars(self, path: str) -> list[dict[str, Any]]:
        """Parse a PokerStars hand history text file.

        Args:
            path: Path to a PokerStars hand history text file. May
                contain multiple hands separated by blank lines.

        Returns:
            List of decision-record dicts.
        """
        p = Path(path)
        if not p.exists():
            logger.warning("PokerStars file not found: %s", path)
            return []

        text = p.read_text(encoding="utf-8", errors="replace")
        hands = self._split_hands(text)

        records: list[dict[str, Any]] = []
        for hand_text in hands:
            records.extend(self._parse_hand(hand_text))

        return records

    @staticmethod
    def _split_hands(text: str) -> list[str]:
        """Split a multi-hand file into individual hand blocks."""
        blocks: list[str] = []
        current: list[str] = []
        for line in text.splitlines():
            if line.startswith("PokerStars") and current:
                blocks.append("\n".join(current))
                current = []
            current.append(line)
        if current:
            blocks.append("\n".join(current))
        return blocks

    def _parse_hand(self, text: str) -> list[dict[str, Any]]:
        """Parse a single PokerStars hand into decision records."""
        records: list[dict[str, Any]] = []

        # Hand ID
        hand_match = self._HAND_RE.search(text)
        hand_id = hand_match.group(1) if hand_match else str(uuid4())[:8]

        # Seats
        players: dict[str, dict[str, Any]] = {}
        for m in self._SEAT_RE.finditer(text):
            seat_num = int(m.group(1))
            name = m.group(2)
            stack = float(m.group(3))
            players[name] = {"seat": seat_num, "stack": stack, "hole_cards": []}

        if not players:
            return records

        num_players = len(players)

        # Hole cards
        for m in self._HOLE_RE.finditer(text):
            name = m.group(1)
            cards_str = m.group(2)
            cards = cards_str.strip().split()
            if name in players:
                players[name]["hole_cards"] = cards

        # Showdown cards
        for m in self._SHOWDOWN_RE.finditer(text):
            name = m.group(1)
            cards_str = m.group(2)
            cards = cards_str.strip().split()
            if name in players and not players[name]["hole_cards"]:
                players[name]["hole_cards"] = cards

        # Blinds → initial pot
        pot = 0.0
        for m in self._BLIND_RE.finditer(text):
            pot += float(m.group(2))

        big_blind = 0.0
        for m in self._BLIND_RE.finditer(text):
            bb = float(m.group(2))
            if bb > big_blind:
                big_blind = bb
        if big_blind == 0.0:
            big_blind = 1.0

        # Split into streets
        street_sections = self._split_streets(text)

        # Board tracking
        board: list[str] = []
        action_idx = 0

        for street_name, section_text in street_sections:
            if street_name in ("showdown",):
                continue

            # Extract board for this street
            board_match = self._BOARD_RE.search(section_text)
            if board_match:
                board = board_match.group(1).strip().split()

            # Parse actions
            for m in self._ACTION_RE.finditer(section_text):
                name = m.group(1)
                raw_action = m.group(2)
                amount_str = m.group(3) or m.group(4) or ""
                is_all_in = bool(m.group(5))

                if name not in players:
                    continue

                player = players[name]
                amount = _safe_float(amount_str)

                # Map raw action to canonical type
                if is_all_in:
                    action_type = "all_in"
                elif raw_action == "folds":
                    action_type = "fold"
                elif raw_action == "checks":
                    action_type = "check"
                elif raw_action == "calls":
                    action_type = "call"
                elif raw_action == "bets":
                    action_type = "bet"
                elif raw_action == "raises":
                    action_type = "raise"
                else:
                    action_type = raw_action.rstrip("s")

                seat = player["seat"]
                position_score = seat / max(1, num_players - 1)
                stack = player["stack"]
                pot_committed = amount / stack if stack > 0 else 0.0

                street_board = PHHIngester._board_for_street(board, street_name)
                hole = player["hole_cards"]

                source_key = f"pokerstars:{hand_id}:{action_idx}"
                record = {
                    "row_id": str(uuid4()),
                    "source_type": "pokerstars",
                    "source_key": source_key,
                    "hand_id": hand_id,
                    "player_name": name,
                    "actor_seat": seat,
                    "street": street_name,
                    "hole_cards": hole if hole else None,
                    "hole_class": _canonical_hole(hole) if len(hole) == 2 else None,
                    "board": street_board,
                    "board_texture": _board_texture(street_board),
                    "pot": round(pot, 2),
                    "to_call": round(amount, 2) if action_type == "call" else 0.0,
                    "pot_odds": (
                        round(amount / (pot + amount), 4)
                        if action_type == "call" and (pot + amount) > 0
                        else 0.0
                    ),
                    "action_type": action_type,
                    "amount": round(amount, 2),
                    "bet_fraction": round(amount / pot, 4) if pot > 0 else 0.0,
                    "position_score": round(position_score, 4),
                    "pot_committed": round(_clamp(pot_committed), 4),
                    "big_blind": big_blind,
                    "num_players": num_players,
                    "is_all_in": is_all_in,
                    "split": _stable_split(source_key),
                }
                records.append(record)
                action_idx += 1

                # Update pot
                if action_type in ("call", "bet", "raise", "all_in"):
                    pot += amount

        return records

    def _split_streets(self, text: str) -> list[tuple[str, str]]:
        """Split hand text into per-street sections."""
        markers: list[tuple[int, str]] = []
        for marker_text, street_name in self._STREET_MARKERS:
            idx = text.find(marker_text)
            if idx >= 0:
                markers.append((idx, street_name))
        markers.sort()

        sections: list[tuple[str, str]] = []
        for i, (start_idx, street_name) in enumerate(markers):
            end_idx = markers[i + 1][0] if i + 1 < len(markers) else len(text)
            sections.append((street_name, text[start_idx:end_idx]))

        return sections


# ---------------------------------------------------------------------------
# GenericCSVIngester
# ---------------------------------------------------------------------------


class GenericCSVIngester:
    """Ingester for generic CSV or JSONL files with poker decision features.

    Uses a configurable column mapping to translate arbitrary column names
    into the internal record schema.
    """

    #: Default column mapping; keys are internal field names, values are
    #: the expected column names in the source file.
    DEFAULT_MAPPING: dict[str, str] = {
        "action_type": "action",
        "street": "street",
        "pot": "pot",
        "to_call": "to_call",
        "amount": "amount",
        "hole_cards": "hole_cards",
        "board": "board",
        "position": "position",
        "player_name": "player",
        "hand_id": "hand_id",
    }

    def ingest_csv(
        self, path: str, column_mapping: dict[str, str] | None = None
    ) -> list[dict[str, Any]]:
        """Ingest a generic CSV or JSONL poker dataset.

        Args:
            path: Path to the data file (CSV or JSONL).
            column_mapping: Dict mapping internal field names to column names
                in the source file. Defaults to :attr:`DEFAULT_MAPPING`.

        Returns:
            List of decision-record dicts.
        """
        p = Path(path)
        if not p.exists():
            logger.warning("Generic CSV file not found: %s", path)
            return []

        mapping = {**self.DEFAULT_MAPPING, **(column_mapping or {})}
        rows = self._load_rows(p)
        records: list[dict[str, Any]] = []

        for idx, row in enumerate(rows):
            record = self._map_row(row, mapping, idx, p.stem)
            if record is not None:
                records.append(record)

        return records

    @staticmethod
    def _load_rows(path: Path) -> list[dict[str, str]]:
        """Load rows from CSV or JSONL."""
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            rows: list[dict[str, str]] = []
            for line in text.splitlines():
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return rows
        # CSV
        reader = csv.DictReader(text.splitlines())
        return list(reader)

    @staticmethod
    def _map_row(
        row: dict[str, Any],
        mapping: dict[str, str],
        idx: int,
        file_stem: str,
    ) -> dict[str, Any] | None:
        """Map a source row to an internal record using the column mapping."""

        def _get(internal_name: str, default: Any = None) -> Any:
            col = mapping.get(internal_name, internal_name)
            return row.get(col, row.get(internal_name, default))

        action_raw = _get("action_type", "")
        if not action_raw:
            return None
        action_type = str(action_raw).strip().lower().replace("-", "_")

        pot = _safe_float(_get("pot", 0))
        to_call = _safe_float(_get("to_call", 0))
        amount = _safe_float(_get("amount", 0))
        street = str(_get("street", "pre_flop")).strip().lower().replace("-", "_")

        # Parse hole cards
        hole_raw = _get("hole_cards")
        hole_cards: list[str] = []
        if hole_raw:
            if isinstance(hole_raw, list):
                hole_cards = hole_raw
            elif isinstance(hole_raw, str):
                hole_raw = hole_raw.strip().strip("[]")
                hole_cards = [c.strip().strip("'\"") for c in re.split(r"[,\s]+", hole_raw) if c.strip()]

        # Parse board
        board_raw = _get("board")
        board: list[str] = []
        if board_raw:
            if isinstance(board_raw, list):
                board = board_raw
            elif isinstance(board_raw, str):
                board_raw = board_raw.strip().strip("[]")
                board = [c.strip().strip("'\"") for c in re.split(r"[,\s]+", board_raw) if c.strip()]

        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.0

        source_key = f"generic:{file_stem}:{idx}"
        return {
            "row_id": str(uuid4()),
            "source_type": "generic_csv",
            "source_key": source_key,
            "hand_id": str(_get("hand_id", "")),
            "player_name": str(_get("player_name", "")),
            "actor_seat": int(_safe_float(_get("actor_seat", 0))),
            "street": street,
            "hole_cards": hole_cards if hole_cards else None,
            "hole_class": _canonical_hole(hole_cards) if len(hole_cards) == 2 else None,
            "board": board if board else None,
            "board_texture": _board_texture(board) if board else None,
            "pot": round(pot, 2),
            "to_call": round(to_call, 2),
            "pot_odds": round(pot_odds, 4),
            "action_type": action_type,
            "amount": round(amount, 2),
            "bet_fraction": round(amount / pot, 4) if pot > 0 else 0.0,
            "split": _stable_split(source_key),
        }


# ---------------------------------------------------------------------------
# TrainingDataFormatter
# ---------------------------------------------------------------------------


class TrainingDataFormatter:
    """Formats ingested records for various training pipelines.

    Converts generic decision records into the specific data structures
    required by different model training loops:
    - CFR (Counterfactual Regret Minimization)
    - CNN-LSTM skill estimator
    - GRU behavior predictor
    - Synthetic player calibration
    """

    @staticmethod
    def format_for_cfr(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format records for CFR training.

        CFR needs information sets: (street, hole_class, board_texture,
        pot_odds, action_history) mapped to action frequencies.

        Args:
            records: Ingested decision records.

        Returns:
            List of CFR-compatible info-set dicts with strategy data.
        """
        cfr_rows: list[dict[str, Any]] = []
        for rec in records:
            street = rec.get("street", "pre_flop")
            hole_class = rec.get("hole_class", "??")
            board_texture = rec.get("board_texture", "preflop")
            pot = _safe_float(rec.get("pot", 0))
            to_call = _safe_float(rec.get("to_call", 0))
            pot_odds = _safe_float(rec.get("pot_odds", 0))
            action = rec.get("action_type") or rec.get("label_action", "fold")
            amount = _safe_float(rec.get("amount", 0))

            # SPR bucket
            spr = pot / to_call if to_call > 0 else 99.0
            spr_bucket = (
                "micro" if spr < 2 else
                "low" if spr < 5 else
                "medium" if spr < 13 else
                "high"
            )

            # Pot odds bucket
            po_bucket = (
                "free" if pot_odds == 0 else
                "cheap" if pot_odds < 0.2 else
                "moderate" if pot_odds < 0.35 else
                "expensive"
            )

            info_set_key = f"{street}|{hole_class}|{board_texture}|{po_bucket}|{spr_bucket}"

            cfr_rows.append({
                "info_set_key": info_set_key,
                "street": street,
                "hole_class": hole_class,
                "board_texture": board_texture,
                "pot_odds": round(pot_odds, 4),
                "spr_bucket": spr_bucket,
                "po_bucket": po_bucket,
                "action": action,
                "amount": round(amount, 2),
                "pot": round(pot, 2),
                "source_key": rec.get("source_key", ""),
            })

        return cfr_rows

    @staticmethod
    def format_for_skill_estimator(
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Format records as DecisionFeature vectors for the CNN-LSTM skill estimator.

        Each record is mapped to the 8-float DecisionFeature format:
        (action_type, decision_time_norm, bet_fraction, position_score,
         street, pot_committed, aggression_context, hand_strength).

        Args:
            records: Ingested decision records.

        Returns:
            List of dicts each containing a ``features`` key with an 8-float
            vector and metadata.
        """
        result: list[dict[str, Any]] = []
        for rec in records:
            action = str(rec.get("action_type") or rec.get("label_action", "fold"))
            action_float = _ACTION_TYPE_FLOAT.get(action, 0.0)

            street = str(rec.get("street", "pre_flop"))
            street_float = _STREET_FLOAT.get(street, 0.0)

            bet_fraction = _clamp(_safe_float(rec.get("bet_fraction", 0)))
            position_score = _clamp(_safe_float(rec.get("position_score", 0.5)))
            pot_committed = _clamp(_safe_float(rec.get("pot_committed", 0)))

            # Decision time — not always available
            decision_time = _clamp(_safe_float(rec.get("decision_time_norm", 0.5)))

            # Aggression context — derive from bet fraction and action
            aggression = 0.5
            if action in ("bet", "raise", "all_in"):
                aggression = _clamp(0.6 + bet_fraction * 0.3)
            elif action in ("check", "call"):
                aggression = _clamp(0.3 + bet_fraction * 0.1)
            elif action == "fold":
                aggression = 0.2

            # Hand strength — use equity if available
            hand_strength = _clamp(
                _safe_float(rec.get("hand_strength",
                            rec.get("estimated_equity", 0.5)))
            )

            features = [
                action_float,
                decision_time,
                bet_fraction,
                position_score,
                street_float,
                pot_committed,
                aggression,
                hand_strength,
            ]

            result.append({
                "features": features,
                "source_key": rec.get("source_key", ""),
                "hand_id": rec.get("hand_id"),
                "split": rec.get("split", "train"),
            })

        return result

    @staticmethod
    def format_for_behavior_predictor(
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Format records as ActionEvent-compatible dicts for the GRU predictor.

        Maps each record to the behavior_prediction.ActionEvent schema:
        (action, street, bet_fraction, position, facing_bet).

        Args:
            records: Ingested decision records.

        Returns:
            List of ActionEvent-compatible dicts.
        """
        result: list[dict[str, Any]] = []
        for rec in records:
            action = str(rec.get("action_type") or rec.get("label_action", "fold"))
            street = str(rec.get("street", "pre_flop"))
            # Normalize street name for behavior_prediction module
            street_norm = street.replace("pre_flop", "preflop")
            bet_fraction = _clamp(_safe_float(rec.get("bet_fraction", 0)))
            position_score = _safe_float(rec.get("position_score", 0.5))
            position = 1 if position_score >= 0.5 else 0
            to_call = _safe_float(rec.get("to_call", 0))
            facing_bet = to_call > 0

            result.append({
                "action": action,
                "street": street_norm,
                "bet_fraction": round(bet_fraction, 4),
                "position": position,
                "facing_bet": facing_bet,
                "source_key": rec.get("source_key", ""),
                "hand_id": rec.get("hand_id"),
                "split": rec.get("split", "train"),
            })

        return result

    @staticmethod
    def format_for_synthetic_player_calibration(
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Format records for synthetic player calibration.

        Aggregates per-player statistics (VPIP, PFR, aggression, etc.)
        from a sequence of decision records, producing calibration data
        that can tune SyntheticPlayerStats parameters.

        Args:
            records: Ingested decision records (should be from the same
                player or session for meaningful aggregation).

        Returns:
            List of player-stat summary dicts.
        """
        # Group records by player
        players: dict[str, list[dict[str, Any]]] = {}
        for rec in records:
            player_key = (
                rec.get("player_name")
                or rec.get("source_key", "").split(":")[0]
                or "unknown"
            )
            players.setdefault(player_key, []).append(rec)

        result: list[dict[str, Any]] = []
        for player_key, player_records in players.items():
            total = len(player_records)
            if total == 0:
                continue

            # Preflop decisions
            preflop = [
                r for r in player_records
                if r.get("street", "") in ("pre_flop", "preflop")
            ]
            preflop_total = len(preflop) or 1

            # VPIP: voluntarily put money in pot preflop (call/raise/bet, not blinds)
            vpip_actions = sum(
                1 for r in preflop
                if r.get("action_type") in ("call", "bet", "raise", "all_in")
            )
            vpip = vpip_actions / preflop_total

            # PFR: preflop raise
            pfr_actions = sum(
                1 for r in preflop
                if r.get("action_type") in ("raise", "all_in")
            )
            pfr = pfr_actions / preflop_total

            # Aggression: (bets + raises) / (calls + checks)
            bets_raises = sum(
                1 for r in player_records
                if r.get("action_type") in ("bet", "raise", "all_in")
            )
            calls_checks = sum(
                1 for r in player_records
                if r.get("action_type") in ("call", "check")
            )
            aggression_factor = (
                bets_raises / calls_checks if calls_checks > 0 else bets_raises
            )

            # WTSD proxy: fraction of hands reaching river
            river_decisions = sum(
                1 for r in player_records if r.get("street") == "river"
            )
            wtsd = river_decisions / max(1, len(set(
                r.get("hand_id") for r in player_records if r.get("hand_id")
            )))

            # Average bet sizing
            bet_fractions = [
                _safe_float(r.get("bet_fraction", 0))
                for r in player_records
                if r.get("action_type") in ("bet", "raise") and r.get("bet_fraction")
            ]
            avg_bet_fraction = (
                statistics.mean(bet_fractions) if bet_fractions else 0.5
            )

            result.append({
                "player_key": player_key,
                "total_decisions": total,
                "vpip": round(_clamp(vpip), 4),
                "pfr": round(_clamp(pfr), 4),
                "aggression_factor": round(min(10.0, aggression_factor), 2),
                "wtsd": round(_clamp(wtsd), 4),
                "avg_bet_fraction": round(_clamp(avg_bet_fraction), 4),
                "preflop_decisions": len(preflop),
                "river_decisions": river_decisions,
                "bets_raises": bets_raises,
                "calls_checks": calls_checks,
            })

        return result


# ---------------------------------------------------------------------------
# DatasetIngestionPipeline (orchestrator)
# ---------------------------------------------------------------------------


class DatasetIngestionPipeline:
    """Orchestrator for ingesting external poker datasets into MR_POKER format.

    Auto-detects file format and delegates to the appropriate ingester.
    Writes output as JSONL with train/val/test splits and generates
    a manifest with metadata and feature statistics.

    Usage::

        pipeline = DatasetIngestionPipeline(output_dir="var/datasets")
        result = pipeline.ingest("data/hands.phh", source_type="phh")
        print(result.summary())

        # Batch ingest
        results = pipeline.ingest_batch([
            {"source_path": "data/pokerbench.jsonl", "source_type": "pokerbench"},
            {"source_path": "data/stars_history.txt", "source_type": "pokerstars"},
        ])
    """

    _INGESTERS: dict[str, str] = {
        "phh": "PHHIngester",
        "pokerbench": "PokerBenchIngester",
        "nemotron": "NemotronPersonasIngester",
        "nemotron_personas": "NemotronPersonasIngester",
        "pokerstars": "PokerStarsIngester",
        "csv": "GenericCSVIngester",
        "generic": "GenericCSVIngester",
    }

    _EXT_TO_TYPE: dict[str, str] = {
        ".phh": "phh",
        ".stars": "pokerstars",
        ".txt": "pokerstars",
    }

    def __init__(self, output_dir: str = "var/datasets") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._phh = PHHIngester()
        self._pokerbench = PokerBenchIngester()
        self._nemotron = NemotronPersonasIngester()
        self._pokerstars = PokerStarsIngester()
        self._generic = GenericCSVIngester()
        self._formatter = TrainingDataFormatter()

    def ingest(
        self,
        source_path: str,
        source_type: str = "auto",
        *,
        column_mapping: dict[str, str] | None = None,
    ) -> IngestionResult:
        """Ingest a single dataset file.

        Args:
            source_path: Path to the source data file.
            source_type: One of ``"phh"``, ``"pokerbench"``, ``"nemotron"``,
                ``"pokerstars"``, ``"csv"``, ``"generic"``, or ``"auto"``
                (detect from extension/content).
            column_mapping: Optional column mapping for generic CSV ingestion.

        Returns:
            An :class:`IngestionResult` with paths and statistics.
        """
        if source_type == "auto":
            source_type = self._detect_type(source_path)

        logger.info("Ingesting %s as %s", source_path, source_type)

        # Dispatch to appropriate ingester
        records = self._run_ingester(source_path, source_type, column_mapping)

        total = len(records)
        if total == 0:
            return IngestionResult(
                source_path=source_path,
                source_type=source_type,
                total_records=0,
                valid_records=0,
                skipped_records=0,
            )

        # Validate and count
        valid_records = [r for r in records if r.get("row_id")]
        skipped = total - len(valid_records)

        # Compute feature statistics
        feature_stats = self._compute_feature_stats(valid_records)

        # Write output
        dataset_id = str(uuid4())
        target = self.output_dir / dataset_id
        target.mkdir(parents=True, exist_ok=True)

        # Write full JSONL
        rows_payload = "\n".join(json.dumps(r, default=str) for r in valid_records)
        if rows_payload:
            rows_payload += "\n"
        (target / "rows.jsonl").write_text(rows_payload, encoding="utf-8")

        # Write per-split files
        for split in ("train", "validation", "test"):
            split_rows = [r for r in valid_records if r.get("split") == split]
            split_payload = "\n".join(json.dumps(r, default=str) for r in split_rows)
            if split_payload:
                split_payload += "\n"
            (target / f"{split}.jsonl").write_text(split_payload, encoding="utf-8")

        # Write manifest
        split_counts = {}
        for r in valid_records:
            s = r.get("split", "train")
            split_counts[s] = split_counts.get(s, 0) + 1

        manifest = {
            "dataset_id": dataset_id,
            "source_path": source_path,
            "source_type": source_type,
            "total_records": total,
            "valid_records": len(valid_records),
            "skipped_records": skipped,
            "split_counts": split_counts,
            "feature_stats": feature_stats,
        }
        manifest_path = target / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )

        result = IngestionResult(
            source_path=source_path,
            source_type=source_type,
            total_records=total,
            valid_records=len(valid_records),
            skipped_records=skipped,
            output_path=str(target),
            manifest_path=str(manifest_path),
            feature_stats=feature_stats,
        )

        logger.info(result.summary())
        return result

    def ingest_batch(
        self, sources: list[dict[str, Any]]
    ) -> list[IngestionResult]:
        """Ingest multiple dataset files.

        Args:
            sources: List of dicts with at least ``source_path`` and
                optionally ``source_type`` and ``column_mapping``.

        Returns:
            List of :class:`IngestionResult` objects.
        """
        results: list[IngestionResult] = []
        for src in sources:
            source_path = src["source_path"]
            source_type = src.get("source_type", "auto")
            column_mapping = src.get("column_mapping")
            result = self.ingest(
                source_path, source_type, column_mapping=column_mapping
            )
            results.append(result)
        return results

    def _run_ingester(
        self,
        source_path: str,
        source_type: str,
        column_mapping: dict[str, str] | None,
    ) -> list[dict[str, Any]]:
        """Dispatch to the correct ingester."""
        if source_type == "phh":
            return self._phh.ingest_phh_file(source_path)
        elif source_type == "pokerbench":
            return self._pokerbench.ingest_pokerbench(source_path)
        elif source_type in ("nemotron", "nemotron_personas"):
            return self._nemotron.ingest_personas(source_path)
        elif source_type == "pokerstars":
            return self._pokerstars.ingest_pokerstars(source_path)
        elif source_type in ("csv", "generic"):
            return self._generic.ingest_csv(source_path, column_mapping)
        else:
            logger.warning("Unknown source type: %s, trying generic CSV", source_type)
            return self._generic.ingest_csv(source_path, column_mapping)

    def _detect_type(self, source_path: str) -> str:
        """Auto-detect dataset type from file extension and content."""
        p = Path(source_path)
        ext = p.suffix.lower()

        # Check extension
        if ext in self._EXT_TO_TYPE:
            return self._EXT_TO_TYPE[ext]

        # Peek at content
        if p.exists():
            try:
                head = p.read_text(encoding="utf-8", errors="replace")[:2000]
            except OSError:
                head = ""

            if "PokerStars" in head:
                return "pokerstars"
            if "instruction" in head and "output" in head:
                return "pokerbench"
            if "conversations" in head or "persona" in head:
                return "nemotron"
            if "variant" in head and ("actions" in head or "stacks" in head):
                return "phh"

        # Default fallback
        if ext in (".csv", ".jsonl", ".json"):
            return "generic"

        return "generic"

    @staticmethod
    def _compute_feature_stats(
        records: list[dict[str, Any]],
    ) -> dict[str, dict[str, float]]:
        """Compute mean/std/min/max for numeric features across records."""
        numeric_keys = ["pot", "to_call", "pot_odds", "amount", "bet_fraction",
                        "position_score", "pot_committed"]
        stats: dict[str, dict[str, float]] = {}

        for key in numeric_keys:
            values = []
            for r in records:
                v = r.get(key)
                if v is not None:
                    try:
                        values.append(float(v))
                    except (ValueError, TypeError):
                        pass

            if not values:
                continue

            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            stats[key] = {
                "mean": round(mean, 4),
                "std": round(std, 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "count": len(values),
            }

        return stats
