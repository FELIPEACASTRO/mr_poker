from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from packages.engine.models import Card
from packages.features.poker import RANK_ORDER


@dataclass(frozen=True)
class BoardTexture:
    """Detailed board texture analysis."""

    wetness: float
    is_monotone: bool
    is_two_tone: bool
    is_rainbow: bool
    is_paired: bool
    pair_count: int
    flush_possible: bool
    flush_draw_possible: bool
    straight_possible: bool
    straight_draw_possible: bool
    high_card_rank: int
    connectivity: float


def analyze_board(board: list[Card]) -> BoardTexture:
    """Analyze board texture for strategic decision making."""
    if len(board) < 3:
        return BoardTexture(
            wetness=0.0,
            is_monotone=False,
            is_two_tone=False,
            is_rainbow=False,
            is_paired=False,
            pair_count=0,
            flush_possible=False,
            flush_draw_possible=False,
            straight_possible=False,
            straight_draw_possible=False,
            high_card_rank=0,
            connectivity=0.0,
        )

    suits = [c.suit for c in board]
    ranks = sorted([RANK_ORDER[c.rank] for c in board])
    unique_suits = len(set(suits))
    suit_counts = {s: suits.count(s) for s in set(suits)}
    max_suit_count = max(suit_counts.values())

    # Pairing
    unique_ranks = len(set(ranks))
    pair_count = len(ranks) - unique_ranks
    is_paired = pair_count > 0

    # Suit texture
    is_monotone = unique_suits == 1
    is_two_tone = unique_suits == 2
    is_rainbow = unique_suits >= 3

    # Flush analysis
    flush_possible = max_suit_count >= 3 and len(board) >= 3
    flush_draw_possible = max_suit_count >= 2

    # Straight analysis
    unique_sorted = sorted(set(ranks))
    straight_possible = False
    straight_draw_possible = False
    if len(unique_sorted) >= 3:
        for i in range(len(unique_sorted) - 2):
            window = unique_sorted[i : i + min(5, len(unique_sorted) - i)]
            span = window[-1] - window[0]
            if len(window) >= 5 and span == 4:
                straight_possible = True
            if span <= 4:
                straight_draw_possible = True

    # Connectivity: average gap between consecutive sorted ranks
    gaps = [unique_sorted[i + 1] - unique_sorted[i] for i in range(len(unique_sorted) - 1)]
    connectivity = 1.0 / (1.0 + sum(gaps) / len(gaps)) if gaps else 0.0

    # Wetness: composite score (0-1) indicating how draw-heavy the board is
    wetness = 0.0
    if flush_draw_possible:
        wetness += 0.3
    if flush_possible:
        wetness += 0.2
    if straight_draw_possible:
        wetness += 0.2
    if straight_possible:
        wetness += 0.1
    if connectivity > 0.5:
        wetness += 0.1
    if not is_paired:
        wetness += 0.1

    return BoardTexture(
        wetness=min(1.0, wetness),
        is_monotone=is_monotone,
        is_two_tone=is_two_tone,
        is_rainbow=is_rainbow,
        is_paired=is_paired,
        pair_count=pair_count,
        flush_possible=flush_possible,
        flush_draw_possible=flush_draw_possible,
        straight_possible=straight_possible,
        straight_draw_possible=straight_draw_possible,
        high_card_rank=max(ranks) if ranks else 0,
        connectivity=round(connectivity, 4),
    )


class BoardTextureClassifier:
    """Classifies community cards into texture categories.

    Provides categorical classification (11 texture types), numeric
    features, and a wetness score for board analysis.

    Inspired by PokerBench texture classification.
    """

    TEXTURES = [
        "dry_rainbow",       # No draws, rainbow
        "dry_monotone",      # No draws, monotone
        "wet_flush_draw",    # Flush draw present
        "wet_straight_draw", # Straight draw present
        "wet_both_draws",    # Both flush and straight draws
        "paired",            # Board has a pair
        "double_paired",     # Board has two pairs
        "trips_on_board",    # Three of a kind on board
        "connected_low",     # Connected low cards (2-8)
        "connected_high",    # Connected high cards (9-A)
        "disconnected",      # No coordination
    ]

    def classify(self, board_cards: list[Card]) -> str:
        """Return the texture category for the given board."""
        if len(board_cards) < 3:
            return "disconnected"

        ranks = [RANK_ORDER[c.rank] for c in board_cards]
        suits = [c.suit for c in board_cards]
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)

        # Check for trips on board
        if any(v >= 3 for v in rank_counts.values()):
            return "trips_on_board"

        # Check for double paired (two+ pairs)
        num_pairs = sum(1 for v in rank_counts.values() if v >= 2)
        if num_pairs >= 2:
            return "double_paired"

        # Check for single pair
        if num_pairs == 1:
            return "paired"

        # Compute draw potentials
        has_flush_draw = self._has_flush_draw(suit_counts)
        has_straight_draw = self._has_straight_draw(ranks)

        if has_flush_draw and has_straight_draw:
            return "wet_both_draws"
        if has_flush_draw:
            return "wet_flush_draw"
        if has_straight_draw:
            return "wet_straight_draw"

        # Check connectivity
        sorted_ranks = sorted(set(ranks))
        is_connected = self._is_connected(sorted_ranks)

        if is_connected:
            avg_rank = sum(sorted_ranks) / len(sorted_ranks)
            if avg_rank < 8:  # 8 is roughly the midpoint (2-14 scale)
                return "connected_low"
            return "connected_high"

        # Check monotone (no draws but all same suit)
        if len(suit_counts) == 1:
            return "dry_monotone"

        # Check rainbow dry
        if len(suit_counts) >= 3 and not has_straight_draw:
            return "dry_rainbow"

        return "disconnected"

    def texture_features(self, board_cards: list[Card]) -> dict[str, float]:
        """Return numeric features for the board texture.

        Features:
            flush_potential: 0-1, how likely a flush is possible
            straight_potential: 0-1, how likely a straight is possible
            coordination: 0-1, overall board coordination
            high_card_pct: 0-1, fraction of high cards (T+)
            paired_level: 0=none, 0.33=pair, 0.67=two pair, 1.0=trips
        """
        if len(board_cards) < 3:
            return {
                "flush_potential": 0.0,
                "straight_potential": 0.0,
                "coordination": 0.0,
                "high_card_pct": 0.0,
                "paired_level": 0.0,
            }

        ranks = [RANK_ORDER[c.rank] for c in board_cards]
        suits = [c.suit for c in board_cards]
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)

        # Flush potential
        max_suit = max(suit_counts.values())
        flush_potential = 0.0
        if max_suit >= 4:
            flush_potential = 1.0
        elif max_suit >= 3:
            flush_potential = 0.7
        elif max_suit >= 2:
            flush_potential = 0.3

        # Straight potential
        sorted_unique = sorted(set(ranks))
        straight_potential = self._straight_potential(sorted_unique)

        # Coordination (combined flush + straight potential)
        coordination = min(1.0, (flush_potential + straight_potential) / 2.0)

        # High card percentage (T=10 and above)
        high_cards = sum(1 for r in ranks if r >= 10)
        high_card_pct = high_cards / len(ranks)

        # Paired level
        max_count = max(rank_counts.values())
        num_pairs = sum(1 for v in rank_counts.values() if v >= 2)
        if max_count >= 3:
            paired_level = 1.0
        elif num_pairs >= 2:
            paired_level = 0.67
        elif num_pairs == 1:
            paired_level = 0.33
        else:
            paired_level = 0.0

        return {
            "flush_potential": flush_potential,
            "straight_potential": straight_potential,
            "coordination": coordination,
            "high_card_pct": high_card_pct,
            "paired_level": paired_level,
        }

    def wetness_score(self, board_cards: list[Card]) -> float:
        """0.0 = very dry, 1.0 = very wet.

        Wetness is determined by draw potential (flush + straight),
        connectivity, and coordination of the board.
        """
        if len(board_cards) < 3:
            return 0.0

        features = self.texture_features(board_cards)
        # Weighted combination: flush draws are wetter than straight draws
        wetness = (
            features["flush_potential"] * 0.4
            + features["straight_potential"] * 0.35
            + features["coordination"] * 0.15
            + (1.0 - features["paired_level"]) * 0.1  # paired boards are drier
        )
        return min(1.0, max(0.0, wetness))

    @staticmethod
    def _has_flush_draw(suit_counts: Counter) -> bool:
        """True if any suit appears 3+ times (flush draw possible)."""
        return any(v >= 3 for v in suit_counts.values())

    @staticmethod
    def _has_straight_draw(ranks: list[int]) -> bool:
        """True if there are 3+ cards within a 5-card window."""
        sorted_unique = sorted(set(ranks))
        # Add low ace if ace present (for wheel draws)
        if 14 in sorted_unique:
            sorted_unique = [1] + sorted_unique

        for i in range(len(sorted_unique)):
            window_count = sum(
                1 for r in sorted_unique
                if sorted_unique[i] <= r <= sorted_unique[i] + 4
            )
            if window_count >= 3:
                return True
        return False

    @staticmethod
    def _is_connected(sorted_unique_ranks: list[int]) -> bool:
        """True if ranks form a connected sequence (gaps <= 2)."""
        if len(sorted_unique_ranks) < 2:
            return False
        for i in range(len(sorted_unique_ranks) - 1):
            if sorted_unique_ranks[i + 1] - sorted_unique_ranks[i] > 2:
                return False
        return True

    @staticmethod
    def _straight_potential(sorted_unique: list[int]) -> float:
        """Compute straight draw potential as a 0-1 score."""
        if len(sorted_unique) < 2:
            return 0.0

        # Add low ace for wheel
        ranks = list(sorted_unique)
        if 14 in ranks:
            ranks = [1] + ranks

        best = 0
        for i in range(len(ranks)):
            count = sum(
                1 for r in ranks
                if ranks[i] <= r <= ranks[i] + 4
            )
            best = max(best, count)

        if best >= 5:
            return 1.0
        if best >= 4:
            return 0.9
        if best >= 3:
            return 0.6
        if best >= 2:
            return 0.3
        return 0.0
