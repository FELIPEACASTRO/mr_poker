"""Theory of Mind (ToM) for poker -- models opponent's beliefs about us.

Inspired by Suspicion-Agent (arXiv:2309.17277, UTokyo):
- Level 0: Opponent plays fixed strategy (no adaptation)
- Level 1: Opponent models our strategy (what do they think we'll do?)
- Level 2: Opponent models our model of them (what do they think we think they'll do?)
- Level k: Recursive reasoning up to depth k

Key insight: If we know what the opponent THINKS our strategy is,
we can deviate from their expectations profitably.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum


class ToMLevel(IntEnum):
    """Theory of Mind reasoning depth."""

    LEVEL_0 = 0  # Fixed strategy, no modeling
    LEVEL_1 = 1  # Models our strategy
    LEVEL_2 = 2  # Models our model of them


# Action labels used throughout this module. These are plain strings
# rather than ActionType enums so the ToM engine stays decoupled from
# the game engine's enum -- callers can pass ActionType.value or raw
# strings interchangeably.
_ACTIONS = ("fold", "check", "call", "bet", "raise", "all_in")

# Streets
_STREETS = ("pre_flop", "flop", "turn", "river")

# Style labels
_STYLE_LABELS = (
    "tight_passive",
    "tight_aggressive",
    "loose_passive",
    "loose_aggressive",
)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


@dataclass
class BeliefAboutUs:
    """What the opponent believes our strategy is.

    All frequency values are in [0, 1].
    """

    perceived_vpip: float = 0.30
    perceived_pfr: float = 0.18
    perceived_aggression: float = 0.45
    perceived_bluff_frequency: float = 0.25
    perceived_fold_to_raise: float = 0.40
    perceived_cbet: float = 0.55
    perceived_style: str = "tight_aggressive"
    confidence: float = 0.0  # 0-1, how confident they are in their model

    def to_vector(self) -> list[float]:
        """Flatten numeric fields to a feature vector."""
        return [
            self.perceived_vpip,
            self.perceived_pfr,
            self.perceived_aggression,
            self.perceived_bluff_frequency,
            self.perceived_fold_to_raise,
            self.perceived_cbet,
            self.confidence,
        ]


@dataclass
class ToMPrediction:
    """Prediction about opponent's action considering their belief model."""

    action_probs: dict  # action_label -> float
    their_belief_about_us: BeliefAboutUs
    exploitation_opportunity: float  # 0-1, how exploitable their model is
    recommended_deviation: dict  # action_label -> float adjustment
    reasoning_level: ToMLevel


class _ActionTracker:
    """Tracks action frequencies per street for a single player.

    Used internally to build up perceived stats from visible actions.
    """

    def __init__(self) -> None:
        # street -> action -> count
        self.counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.total_per_street: dict[str, int] = defaultdict(int)
        self.total_actions: int = 0

        # Specific stat counters
        self.preflop_voluntary: int = 0
        self.preflop_raise: int = 0
        self.preflop_opportunities: int = 0
        self.aggressive_actions: int = 0
        self.passive_actions: int = 0
        self.folds_to_raise: int = 0
        self.raise_faced: int = 0
        self.cbets: int = 0
        self.cbet_opportunities: int = 0
        self.bluffs_shown: int = 0
        self.showdowns: int = 0

    def record(self, action: str, street: str, bet_fraction: float = 0.0) -> None:
        """Record a single action."""
        self.counts[street][action] += 1
        self.total_per_street[street] += 1
        self.total_actions += 1

        if street == "pre_flop":
            self.preflop_opportunities += 1
            if action in ("call", "bet", "raise", "all_in"):
                self.preflop_voluntary += 1
            if action in ("raise", "all_in"):
                self.preflop_raise += 1

        if action in ("bet", "raise", "all_in"):
            self.aggressive_actions += 1
        elif action in ("call", "check"):
            self.passive_actions += 1

    def record_fold_to_raise(self, folded: bool) -> None:
        self.raise_faced += 1
        if folded:
            self.folds_to_raise += 1

    def record_cbet_opportunity(self, did_cbet: bool) -> None:
        self.cbet_opportunities += 1
        if did_cbet:
            self.cbets += 1

    def record_showdown(self, was_bluff: bool) -> None:
        self.showdowns += 1
        if was_bluff:
            self.bluffs_shown += 1

    @property
    def vpip(self) -> float:
        if self.preflop_opportunities == 0:
            return 0.30  # prior
        return self.preflop_voluntary / self.preflop_opportunities

    @property
    def pfr(self) -> float:
        if self.preflop_opportunities == 0:
            return 0.18
        return self.preflop_raise / self.preflop_opportunities

    @property
    def aggression(self) -> float:
        total = self.aggressive_actions + self.passive_actions
        if total == 0:
            return 0.45
        return self.aggressive_actions / total

    @property
    def fold_to_raise(self) -> float:
        if self.raise_faced == 0:
            return 0.40
        return self.folds_to_raise / self.raise_faced

    @property
    def cbet(self) -> float:
        if self.cbet_opportunities == 0:
            return 0.55
        return self.cbets / self.cbet_opportunities

    @property
    def bluff_freq(self) -> float:
        if self.showdowns == 0:
            return 0.25
        return self.bluffs_shown / self.showdowns

    def frequency(self, action: str, street: str) -> float:
        """Raw frequency of action on street."""
        total = self.total_per_street.get(street, 0)
        if total == 0:
            return 1.0 / len(_ACTIONS)
        return self.counts.get(street, {}).get(action, 0) / total


def _classify_style(vpip: float, aggression: float) -> str:
    """Classify playing style from VPIP and aggression."""
    tight = vpip < 0.30
    aggressive = aggression > 0.50
    if tight and aggressive:
        return "tight_aggressive"
    if tight and not aggressive:
        return "tight_passive"
    if not tight and aggressive:
        return "loose_aggressive"
    return "loose_passive"


class TheoryOfMind:
    """Theory of Mind engine for poker.

    Maintains a model of what the opponent believes about our strategy.
    Uses this to find profitable deviations.

    Example:
    - If opponent thinks we're tight (perceived_vpip=0.22),
      we can profitably bluff more (they'll fold more to our bets)
    - If opponent thinks we're aggressive (perceived_aggression=0.65),
      we can trap more (they'll expect bets, check-raise works)
    """

    def __init__(
        self,
        our_actual_stats: dict | None = None,
        max_level: ToMLevel = ToMLevel.LEVEL_2,
        update_rate: float = 0.05,
    ) -> None:
        """
        Args:
            our_actual_stats: Our real stats dict with keys like 'vpip',
                'pfr', 'aggression', 'bluff_frequency', 'fold_to_raise', 'cbet'.
                If None, defaults are used.
            max_level: Maximum ToM reasoning depth.
            update_rate: How fast beliefs update (EMA alpha).
        """
        self.max_level = max_level
        self.update_rate = update_rate

        # Our actual strategy stats (ground truth)
        self._our_actual: dict[str, float] = our_actual_stats or {
            "vpip": 0.28,
            "pfr": 0.20,
            "aggression": 0.50,
            "bluff_frequency": 0.30,
            "fold_to_raise": 0.35,
            "cbet": 0.60,
        }

        # Track our visible actions (what opponent can observe)
        self._our_visible_tracker = _ActionTracker()

        # Track opponent's reactions to our actions (reveals their beliefs)
        # Maps (our_action, street) -> opponent_action -> count
        self._reaction_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._reaction_totals: dict[tuple[str, str], int] = defaultdict(int)

        # Level-2: track how opponent's reactions change over time
        # (reveals whether they are updating their model of us)
        self._reaction_history: list[dict] = []

        # Running belief estimate (EMA-smoothed)
        self._current_belief = BeliefAboutUs()

        # Number of observations
        self._n_observations: int = 0

    def record_our_action(
        self,
        action_type: str,
        street: str,
        bet_fraction: float = 0.0,
        was_visible: bool = True,
    ) -> None:
        """Record our action (visible to opponent at showdown or if called).

        Only visible actions update the opponent's perceived model of us.

        Args:
            action_type: Action string (fold, check, call, bet, raise, all_in).
            street: Street string (pre_flop, flop, turn, river).
            bet_fraction: Bet size as fraction of pot (0 if no bet).
            was_visible: Whether opponent could observe this action.
        """
        if was_visible:
            self._our_visible_tracker.record(action_type, street, bet_fraction)
            self._n_observations += 1
            # Refresh belief estimate periodically
            if self._n_observations % 5 == 0:
                self._refresh_belief()

    def record_opponent_reaction(
        self, their_action: str, to_our_action: str, street: str
    ) -> None:
        """Record how opponent reacted to our action.

        Their reaction pattern reveals what they believe about our strategy.
        For example, if they fold a lot to our bets, they perceive us as
        tight/strong. If they call a lot, they perceive us as bluffy.

        Args:
            their_action: What the opponent did (fold, call, raise, etc.).
            to_our_action: What our action was that they reacted to.
            street: Street where this occurred.
        """
        key = (to_our_action, street)
        self._reaction_counts[key][their_action] += 1
        self._reaction_totals[key] += 1

        # Store timestamped snapshot for Level-2 trend analysis
        self._reaction_history.append(
            {
                "our_action": to_our_action,
                "their_reaction": their_action,
                "street": street,
                "observation_num": self._n_observations,
            }
        )

    def estimate_their_belief(self) -> BeliefAboutUs:
        """Estimate what opponent believes our strategy is.

        Based on our visible actions (showdown hands, called bluffs).
        The opponent can only form beliefs from what they have seen.
        """
        self._refresh_belief()
        return self._current_belief

    def _refresh_belief(self) -> None:
        """Recompute belief estimate from visible action tracker."""
        tracker = self._our_visible_tracker
        n = tracker.total_actions

        if n == 0:
            self._current_belief = BeliefAboutUs()
            return

        # Confidence grows with observations (saturates around 100 actions)
        confidence = 1.0 - math.exp(-n / 40.0)

        # Blend prior with observed stats using EMA-like weighting
        alpha = confidence  # More data -> trust observations more

        vpip = (1 - alpha) * 0.30 + alpha * tracker.vpip
        pfr = (1 - alpha) * 0.18 + alpha * tracker.pfr
        aggression = (1 - alpha) * 0.45 + alpha * tracker.aggression
        bluff_freq = (1 - alpha) * 0.25 + alpha * tracker.bluff_freq
        fold_to_raise = (1 - alpha) * 0.40 + alpha * tracker.fold_to_raise
        cbet = (1 - alpha) * 0.55 + alpha * tracker.cbet

        # Adjust belief based on reaction patterns (Level-1+)
        # If opponent folds a lot to our bets, they think we're strong
        # If opponent calls a lot, they think we're bluffy
        reaction_adj = self._reaction_based_adjustment()
        vpip += reaction_adj.get("vpip_adj", 0.0)
        aggression += reaction_adj.get("aggression_adj", 0.0)
        bluff_freq += reaction_adj.get("bluff_adj", 0.0)

        style = _classify_style(vpip, aggression)

        self._current_belief = BeliefAboutUs(
            perceived_vpip=_clamp(vpip),
            perceived_pfr=_clamp(pfr),
            perceived_aggression=_clamp(aggression),
            perceived_bluff_frequency=_clamp(bluff_freq),
            perceived_fold_to_raise=_clamp(fold_to_raise),
            perceived_cbet=_clamp(cbet),
            perceived_style=style,
            confidence=_clamp(confidence),
        )

    def _reaction_based_adjustment(self) -> dict[str, float]:
        """Infer belief adjustments from opponent's reaction patterns.

        If opponent folds a lot to our bets -> they think we're tight/strong
        If opponent calls a lot -> they think we bluff too much
        If opponent raises a lot -> they think we're weak
        """
        adj: dict[str, float] = {}
        total_reactions = sum(self._reaction_totals.values())
        if total_reactions < 5:
            return adj

        # Aggregate fold/call/raise rates to our bets
        folds_to_bet = 0
        calls_to_bet = 0
        raises_to_bet = 0
        bet_reactions = 0

        for (our_action, street), reactions in self._reaction_counts.items():
            if our_action in ("bet", "raise", "all_in"):
                folds_to_bet += reactions.get("fold", 0)
                calls_to_bet += reactions.get("call", 0)
                raises_to_bet += reactions.get("raise", 0) + reactions.get("all_in", 0)
                bet_reactions += self._reaction_totals[(our_action, street)]

        if bet_reactions < 3:
            return adj

        fold_rate = folds_to_bet / bet_reactions
        call_rate = calls_to_bet / bet_reactions

        # High fold rate -> they think we're strong/tight
        # We can use this to bluff more
        if fold_rate > 0.50:
            adj["vpip_adj"] = -0.05  # They think we're tighter
            adj["aggression_adj"] = 0.05  # They think we're more aggressive
            adj["bluff_adj"] = -0.05  # They think we bluff less
        elif call_rate > 0.60:
            adj["vpip_adj"] = 0.05  # They think we're looser
            adj["bluff_adj"] = 0.10  # They think we bluff more
            adj["aggression_adj"] = -0.03

        return adj

    def predict_with_tom(
        self,
        info_set: str,
        legal_actions: set,
        level: ToMLevel | None = None,
    ) -> ToMPrediction:
        """Predict opponent's action accounting for their model of us.

        Level 0: Use raw frequency-based prediction (no ToM)
        Level 1: Adjust for their belief about our strategy
        Level 2: Adjust for their belief about our belief about them

        Args:
            info_set: Situation key (e.g., "flop_bet_IP").
            legal_actions: Set of legal action strings.
            level: ToM level to use. None = auto-select.

        Returns:
            ToMPrediction with action probabilities and exploitation info.
        """
        if level is None:
            level = self._auto_select_level()

        level = min(level, self.max_level)
        belief = self.estimate_their_belief()

        # Level 0: uniform or frequency-based (no belief modeling)
        if level == ToMLevel.LEVEL_0 or belief.confidence < 0.1:
            base_probs = self._level0_prediction(legal_actions)
            return ToMPrediction(
                action_probs=base_probs,
                their_belief_about_us=belief,
                exploitation_opportunity=0.0,
                recommended_deviation={},
                reasoning_level=ToMLevel.LEVEL_0,
            )

        # Level 1: adjust predictions based on their belief about us
        base_probs = self._level0_prediction(legal_actions)
        l1_probs = self._level1_adjustment(base_probs, belief, legal_actions)

        if level == ToMLevel.LEVEL_1:
            deviation = self.recommend_deviation(belief, self._our_actual, legal_actions)
            exploitation = self._compute_exploitation_opportunity(belief)
            return ToMPrediction(
                action_probs=l1_probs,
                their_belief_about_us=belief,
                exploitation_opportunity=exploitation,
                recommended_deviation=deviation,
                reasoning_level=ToMLevel.LEVEL_1,
            )

        # Level 2: adjust for their belief about our belief about them
        l2_probs = self._level2_adjustment(l1_probs, belief, legal_actions)
        deviation = self.recommend_deviation(belief, self._our_actual, legal_actions)
        exploitation = self._compute_exploitation_opportunity(belief)

        return ToMPrediction(
            action_probs=l2_probs,
            their_belief_about_us=belief,
            exploitation_opportunity=exploitation,
            recommended_deviation=deviation,
            reasoning_level=ToMLevel.LEVEL_2,
        )

    def _level0_prediction(self, legal_actions: set) -> dict:
        """Baseline prediction: aggregate reaction frequencies or uniform."""
        probs: dict[str, float] = {}
        total_reactions = sum(self._reaction_totals.values())

        if total_reactions < 5:
            # Uniform
            n = len(legal_actions)
            for a in legal_actions:
                action_key = a.value if hasattr(a, "value") else str(a)
                probs[action_key] = 1.0 / max(1, n)
            return probs

        # Aggregate action frequencies from reaction data
        action_counts: dict[str, int] = defaultdict(int)
        for reactions in self._reaction_counts.values():
            for action, count in reactions.items():
                action_counts[action] += count

        for a in legal_actions:
            action_key = a.value if hasattr(a, "value") else str(a)
            probs[action_key] = action_counts.get(action_key, 1)

        total = sum(probs.values())
        if total > 0:
            probs = {a: p / total for a, p in probs.items()}

        return probs

    def _level1_adjustment(
        self, base_probs: dict, belief: BeliefAboutUs, legal_actions: set
    ) -> dict:
        """Adjust predictions for Level-1 reasoning.

        If opponent believes we are tight, they will:
        - Fold more to our bets (they respect our bets)
        - Bluff us more (they think we fold too much)
        - Call less (they think we have it when we bet)
        """
        adjusted = dict(base_probs)
        n = len(legal_actions)
        if n == 0:
            return adjusted

        # Tightness perception -> they fold more to our bets
        tightness = 1.0 - belief.perceived_vpip  # higher = they think we're tighter
        aggression_perception = belief.perceived_aggression

        fold_boost = 0.0
        call_reduction = 0.0
        raise_boost = 0.0

        # If they think we're tight: fold more, call less to our aggression
        if tightness > 0.6:
            fold_boost = 0.10 * (tightness - 0.5) * belief.confidence
            call_reduction = 0.08 * (tightness - 0.5) * belief.confidence

        # If they think we're passive: raise us more (attack weakness)
        if aggression_perception < 0.40:
            raise_boost = 0.10 * (0.5 - aggression_perception) * belief.confidence

        # If they think we bluff a lot: call more, fold less
        if belief.perceived_bluff_frequency > 0.35:
            bluff_factor = belief.perceived_bluff_frequency - 0.25
            fold_boost -= 0.10 * bluff_factor * belief.confidence
            call_reduction -= 0.08 * bluff_factor * belief.confidence

        # Apply adjustments
        for action_key in adjusted:
            if action_key == "fold":
                adjusted[action_key] = max(0.01, adjusted[action_key] + fold_boost)
            elif action_key == "call":
                adjusted[action_key] = max(0.01, adjusted[action_key] - call_reduction)
            elif action_key in ("raise", "all_in"):
                adjusted[action_key] = max(0.01, adjusted[action_key] + raise_boost)

        # Re-normalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {a: p / total for a, p in adjusted.items()}

        return adjusted

    def _level2_adjustment(
        self, l1_probs: dict, belief: BeliefAboutUs, legal_actions: set
    ) -> dict:
        """Level-2: adjust for opponent modeling our model of them.

        If a sophisticated opponent knows we're modeling their beliefs,
        they may counter-adjust. We detect this from reaction trend changes.
        """
        adjusted = dict(l1_probs)

        # Check if opponent is adapting (reaction pattern change)
        adaptation_signal = self._detect_opponent_adaptation()

        if adaptation_signal < 0.2:
            # Opponent is not adapting, Level-1 is sufficient
            return adjusted

        # Opponent IS adapting: partially revert L1 adjustments
        # (they are counter-adjusting, so our L1 prediction becomes less reliable)
        revert_factor = adaptation_signal * 0.5  # don't fully revert

        # Move toward uniform as a hedge
        n = len(adjusted)
        if n > 0:
            uniform = 1.0 / n
            for action_key in adjusted:
                adjusted[action_key] = (
                    (1 - revert_factor) * adjusted[action_key]
                    + revert_factor * uniform
                )

        # Re-normalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {a: p / total for a, p in adjusted.items()}

        return adjusted

    def _detect_opponent_adaptation(self) -> float:
        """Detect if opponent is changing their strategy over time.

        Compares reaction patterns from first half vs second half of history.
        Returns 0-1 signal (1 = significant adaptation detected).
        """
        history = self._reaction_history
        n = len(history)
        if n < 20:
            return 0.0

        mid = n // 2
        first_half = history[:mid]
        second_half = history[mid:]

        # Count fold/call/raise rates in each half
        def _rates(records: list[dict]) -> tuple[float, float, float]:
            folds = sum(1 for r in records if r["their_reaction"] == "fold")
            calls = sum(1 for r in records if r["their_reaction"] == "call")
            raises = sum(
                1
                for r in records
                if r["their_reaction"] in ("raise", "all_in")
            )
            total = max(1, len(records))
            return folds / total, calls / total, raises / total

        f1, c1, r1 = _rates(first_half)
        f2, c2, r2 = _rates(second_half)

        # L1 distance between distributions
        shift = abs(f1 - f2) + abs(c1 - c2) + abs(r1 - r2)

        # Normalize: max shift is 2.0 (complete reversal)
        return _clamp(shift / 0.6)  # 0.6 shift -> signal = 1.0

    def recommend_deviation(
        self, their_belief: BeliefAboutUs, our_actual: dict, legal_actions: set
    ) -> dict:
        """Recommend profitable deviations from our perceived strategy.

        If they think we're X, and we're actually Y, the profitable
        deviation is to move toward Z where Z exploits the gap between
        their belief and reality.

        Args:
            their_belief: What opponent believes about us.
            our_actual: Our actual strategy stats.
            legal_actions: Available actions.

        Returns:
            Dict mapping action labels to recommended probability adjustments
            (positive = do more, negative = do less).
        """
        deviations: dict[str, float] = {}

        # Gap analysis: where their model is most wrong
        vpip_gap = their_belief.perceived_vpip - our_actual.get("vpip", 0.28)
        agg_gap = their_belief.perceived_aggression - our_actual.get("aggression", 0.50)
        bluff_gap = their_belief.perceived_bluff_frequency - our_actual.get(
            "bluff_frequency", 0.30
        )
        fold_gap = their_belief.perceived_fold_to_raise - our_actual.get(
            "fold_to_raise", 0.35
        )

        # Scale adjustments by their confidence (more confident = more exploitable)
        conf = their_belief.confidence

        for a in legal_actions:
            action_key = a.value if hasattr(a, "value") else str(a)
            adj = 0.0

            if action_key == "fold":
                # If they think we fold a lot (positive gap), fold LESS
                # If they think we fold little, fold slightly more (trap)
                adj = -fold_gap * 0.15 * conf

            elif action_key == "call":
                # If they think we're tight (negative vpip_gap), call more to widen
                adj = -vpip_gap * 0.10 * conf

            elif action_key in ("bet", "raise"):
                # If they think we're passive (negative agg_gap), be MORE aggressive
                adj = -agg_gap * 0.12 * conf
                # If they think we don't bluff (negative bluff_gap), bluff MORE
                adj += -bluff_gap * 0.10 * conf

            elif action_key == "check":
                # If they think we're aggressive, trap more with checks
                adj = agg_gap * 0.08 * conf

            elif action_key == "all_in":
                # If they think we're tight and never bluff, occasional all-in bluff
                if vpip_gap < -0.05 and bluff_gap < -0.05:
                    adj = 0.03 * conf

            deviations[action_key] = _clamp(adj, -0.20, 0.20)

        return deviations

    def _compute_exploitation_opportunity(self, belief: BeliefAboutUs) -> float:
        """Compute how exploitable the opponent's model of us is.

        Larger gap between their perception and our reality = more opportunity.
        """
        actual = self._our_actual

        gaps = [
            abs(belief.perceived_vpip - actual.get("vpip", 0.28)),
            abs(belief.perceived_pfr - actual.get("pfr", 0.20)),
            abs(belief.perceived_aggression - actual.get("aggression", 0.50)),
            abs(belief.perceived_bluff_frequency - actual.get("bluff_frequency", 0.30)),
            abs(belief.perceived_fold_to_raise - actual.get("fold_to_raise", 0.35)),
            abs(belief.perceived_cbet - actual.get("cbet", 0.60)),
        ]

        # Average gap, scaled by their confidence
        avg_gap = sum(gaps) / len(gaps)
        # Max gap ~ 0.5 -> opportunity = 1.0
        opportunity = _clamp(avg_gap / 0.3) * belief.confidence
        return _clamp(opportunity)

    def deception_score(self) -> float:
        """How different our perceived strategy is from our actual strategy.

        Higher = opponent's model of us is more wrong = more exploitation
        opportunity.

        Returns:
            Float in [0, 1]. 0 = they know us perfectly. 1 = maximum deception.
        """
        belief = self.estimate_their_belief()
        return self._compute_exploitation_opportunity(belief)

    def update_after_showdown(
        self, we_showed_cards: bool, was_bluff: bool = False
    ) -> None:
        """Update beliefs after showdown.

        When we show cards (or get called), the opponent gets information
        about our actual strategy. This narrows the gap between their
        perception and reality.

        Args:
            we_showed_cards: Whether our cards were revealed.
            was_bluff: Whether we were bluffing (relevant if cards shown).
        """
        if we_showed_cards:
            self._our_visible_tracker.record_showdown(was_bluff)

            # Showing cards gives opponent a data point, increasing their
            # model confidence. If it was a bluff, they learn we bluff.
            # This is captured automatically by the tracker stats, but
            # we give an extra confidence bump.
            self._n_observations += 3  # showdown is worth multiple actions

            self._refresh_belief()

    def _auto_select_level(self) -> ToMLevel:
        """Automatically select the best ToM level based on data availability."""
        if self._n_observations < 10:
            return ToMLevel.LEVEL_0
        if self._n_observations < 30:
            return ToMLevel.LEVEL_1

        # Check if Level-2 is warranted (opponent is adapting)
        adaptation = self._detect_opponent_adaptation()
        if adaptation > 0.3 and self.max_level >= ToMLevel.LEVEL_2:
            return ToMLevel.LEVEL_2
        return ToMLevel.LEVEL_1

    def reset(self) -> None:
        """Reset ToM state for new opponent."""
        self._our_visible_tracker = _ActionTracker()
        self._reaction_counts = defaultdict(lambda: defaultdict(int))
        self._reaction_totals = defaultdict(int)
        self._reaction_history = []
        self._current_belief = BeliefAboutUs()
        self._n_observations = 0
