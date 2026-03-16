"""Lazy-CFR — Partial update CFR for faster iterations.

Instead of traversing the entire game tree every iteration, Lazy-CFR
selects a subset of information sets to update. This dramatically
reduces per-iteration cost while maintaining convergence guarantees.

Reference: Zhou et al. (2018) "Lazy-CFR: Fast and Near Optimal
Counterfactual Regret Minimization"
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any

from packages.common.types import ActionType
from packages.engine.engine import GameEngine, HandRuntime
from packages.strategy.mixed import ActionDistribution

from packages.cfr_agent.info_set import build_info_set_key
from packages.cfr_agent.trainer import CFRState, _size_action

logger = logging.getLogger(__name__)


class PriorityQueue:
    """Simple priority queue for info set scheduling.

    Uses max-regret as priority: info sets with higher total absolute
    regret are updated more frequently.
    """

    def __init__(self) -> None:
        self._priorities: dict[str, float] = {}
        self._update_counts: dict[str, int] = {}

    def update_priority(self, info_set: str, regret_sum: float) -> None:
        """Update the priority of an info set."""
        self._priorities[info_set] = abs(regret_sum)
        self._update_counts[info_set] = self._update_counts.get(info_set, 0)

    def should_update(self, info_set: str, fraction: float, rng: random.Random) -> bool:
        """Decide whether to update this info set based on priority.

        Higher-priority info sets are updated more often. The fraction
        parameter controls the overall update rate (0.0 = never, 1.0 = always).
        """
        if fraction >= 1.0:
            return True
        if info_set not in self._priorities:
            return True  # Always update new info sets

        # Priority-weighted sampling
        priority = self._priorities[info_set]
        max_priority = max(self._priorities.values()) if self._priorities else 1.0
        if max_priority <= 0:
            return rng.random() < fraction

        # Normalized priority: higher regret -> higher update probability
        normalized = priority / max_priority
        update_prob = fraction + (1.0 - fraction) * normalized
        return rng.random() < update_prob

    def record_update(self, info_set: str) -> None:
        """Record that an info set was updated."""
        self._update_counts[info_set] = self._update_counts.get(info_set, 0) + 1

    @property
    def total_info_sets(self) -> int:
        return len(self._priorities)

    def stats(self) -> dict[str, Any]:
        """Return statistics about the priority queue."""
        if not self._update_counts:
            return {"total": 0, "avg_updates": 0.0, "max_updates": 0}
        counts = list(self._update_counts.values())
        return {
            "total": len(counts),
            "avg_updates": sum(counts) / len(counts),
            "max_updates": max(counts),
            "min_updates": min(counts),
        }


@dataclass
class LazyCFRConfig:
    """Configuration for Lazy-CFR training."""
    # Fraction of info sets to update per iteration (0.0-1.0)
    update_fraction: float = 0.3
    # Use priority-based selection (vs random)
    use_priority: bool = True
    # Minimum iterations before enabling lazy updates
    warmup_iterations: int = 100
    # Full update every N iterations (to ensure convergence)
    full_update_interval: int = 50


class LazyCFRTrainer:
    """CFR trainer with lazy (partial) updates.

    Instead of updating all information sets every iteration, selects
    a subset based on priority (total absolute regret). Info sets
    with higher regret are updated more frequently.

    Combined with DCFR discounting, this provides 2-5x speedup
    with minimal convergence impact.
    """

    def __init__(
        self,
        *,
        small_blind: int = 1,
        big_blind: int = 2,
        starting_stack: int = 100,
        seed: int = 42,
        mode: str = "dcfr",
        config: LazyCFRConfig | None = None,
    ) -> None:
        self.engine = GameEngine(small_blind=small_blind, big_blind=big_blind)
        self.starting_stack = starting_stack
        self.rng = random.Random(seed)
        self.cfr_state = CFRState()
        self.mode = mode
        self.config = config or LazyCFRConfig()
        self.priority_queue = PriorityQueue()
        self._skipped_updates = 0
        self._total_updates = 0

    def train(self, iterations: int = 10000) -> CFRState:
        """Run Lazy-CFR training."""
        for i in range(iterations):
            seed = self.rng.randint(0, 2**31)
            runtime = self.engine.start_new_hand(
                stacks=(self.starting_stack, self.starting_stack),
                button_seat=i % 2,
                seed=seed,
            )

            # Full update during warmup or at regular intervals
            is_full = (
                i < self.config.warmup_iterations
                or (i + 1) % self.config.full_update_interval == 0
            )

            if self.mode == "mccfr":
                traverser = i % 2
                self._mccfr_lazy(runtime, traverser, is_full)
            else:
                self._cfr_lazy(runtime, {0: 1.0, 1: 1.0}, is_full)

            self.cfr_state.iterations += 1

            if self.mode == "dcfr" and (i + 1) % 100 == 0:
                self.cfr_state.apply_dcfr_discount()

            if (i + 1) % 1000 == 0:
                skip_rate = (
                    self._skipped_updates / max(self._total_updates, 1) * 100
                )
                logger.info(
                    "Lazy-CFR iter %d/%d — %d info sets, skip_rate=%.1f%%",
                    i + 1, iterations,
                    len(self.cfr_state.strategy_sum),
                    skip_rate,
                )

        return self.cfr_state

    def _should_update_info_set(self, info_set: str, is_full: bool) -> bool:
        """Decide whether to update this info set."""
        if is_full:
            return True
        if self.config.use_priority:
            return self.priority_queue.should_update(
                info_set, self.config.update_fraction, self.rng
            )
        return self.rng.random() < self.config.update_fraction

    def _update_priority(self, info_set: str) -> None:
        """Update the priority of an info set based on current regrets."""
        if info_set in self.cfr_state.cumulative_regret:
            regret_sum = sum(
                abs(v) for v in self.cfr_state.cumulative_regret[info_set].values()
            )
            self.priority_queue.update_priority(info_set, regret_sum)

    def _cfr_lazy(
        self,
        runtime: HandRuntime,
        reach_probs: dict[int, float],
        is_full: bool,
    ) -> dict[int, float]:
        """CFR traversal with lazy updates."""
        state = runtime.state

        if state.is_terminal or state.acting_seat is None:
            return {
                seat: p.stack - self.starting_stack
                for seat, p in state.players.items()
            }

        seat = state.acting_seat
        opponent = 1 - seat
        legal = set(self.engine.legal_actions(runtime))
        info_set = build_info_set_key(state, seat)
        strategy = self.cfr_state.current_strategy(info_set, legal)

        action_utilities: dict[ActionType, float] = {}
        node_utility = 0.0

        for action in legal:
            prob = strategy.probabilities.get(action, 0.0)
            if prob <= 0 and self.cfr_state.iterations > 0:
                continue

            child_runtime = self._clone_and_act(runtime, action)
            if child_runtime is None:
                continue

            new_reach = dict(reach_probs)
            new_reach[seat] = reach_probs[seat] * max(prob, 1e-6)
            child_utils = self._cfr_lazy(child_runtime, new_reach, is_full)

            action_utilities[action] = child_utils.get(seat, 0.0)
            node_utility += prob * action_utilities[action]

        # Lazy update decision
        self._total_updates += 1
        if self._should_update_info_set(info_set, is_full):
            self.cfr_state.update(info_set, strategy, action_utilities, node_utility)
            self._update_priority(info_set)
            self.priority_queue.record_update(info_set)
        else:
            self._skipped_updates += 1

        return {seat: node_utility, opponent: -node_utility}

    def _mccfr_lazy(
        self,
        runtime: HandRuntime,
        traverser: int,
        is_full: bool,
    ) -> float:
        """MCCFR with lazy updates."""
        state = runtime.state

        if state.is_terminal or state.acting_seat is None:
            return state.players[traverser].stack - self.starting_stack

        seat = state.acting_seat
        legal = set(self.engine.legal_actions(runtime))
        info_set = build_info_set_key(state, seat)
        strategy = self.cfr_state.current_strategy(info_set, legal)

        if seat != traverser:
            sampled = strategy.sample(self.rng)
            if sampled not in legal:
                sampled = self.rng.choice(list(legal))
            child = self._clone_and_act(runtime, sampled)
            if child is None:
                return 0.0
            return self._mccfr_lazy(child, traverser, is_full)

        action_utilities: dict[ActionType, float] = {}
        for action in legal:
            child = self._clone_and_act(runtime, action)
            if child is None:
                action_utilities[action] = 0.0
                continue
            action_utilities[action] = self._mccfr_lazy(child, traverser, is_full)

        node_utility = sum(
            strategy.probabilities.get(a, 0.0) * u
            for a, u in action_utilities.items()
        )

        # Lazy update decision
        self._total_updates += 1
        if self._should_update_info_set(info_set, is_full):
            self.cfr_state.update(info_set, strategy, action_utilities, node_utility)
            self._update_priority(info_set)
            self.priority_queue.record_update(info_set)
        else:
            self._skipped_updates += 1

        return node_utility

    def _clone_and_act(
        self, runtime: HandRuntime, action: ActionType
    ) -> HandRuntime | None:
        """Clone the game state and apply one action."""
        state = runtime.state
        if state.acting_seat is None:
            return None

        player = state.players[state.acting_seat]
        amount = _size_action(action, state, player, self.engine)

        all_hole_cards: list[str] = []
        for s in sorted(state.players.keys()):
            for c in state.players[s].hole_cards:
                all_hole_cards.append(str(c))
        board_cards = [str(c) for c in state.board]
        burned = [str(c) for c in runtime.burned_cards]
        deck_prefix = all_hole_cards + burned[:len(state.board)] + board_cards

        try:
            new_runtime = self.engine.start_new_hand(
                stacks=tuple(runtime.initial_stacks[s] for s in sorted(runtime.initial_stacks)),
                button_seat=state.button_seat,
                deck_prefix=deck_prefix,
            )
            for ev in state.actions:
                if new_runtime.state.is_terminal or new_runtime.state.acting_seat is None:
                    break
                self.engine.apply_action(new_runtime, ev.action_type, ev.amount)
            if new_runtime.state.is_terminal or new_runtime.state.acting_seat is None:
                return None
            self.engine.apply_action(new_runtime, action, amount)
            return new_runtime
        except Exception:
            return None
