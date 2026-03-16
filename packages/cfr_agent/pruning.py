"""Regret-based Pruning for CFR.

Skips traversal of actions with sufficiently negative cumulative regret,
providing ~10x speedup in later iterations with minimal convergence impact.

Reference: Brown & Sandholm (2015) "Regret-Based Pruning in Extensive-Form Games"
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
from packages.cfr_agent.trainer import CFRState, _size_action, CFR_ACTIONS

logger = logging.getLogger(__name__)


@dataclass
class PruningStats:
    """Tracks pruning statistics for monitoring."""
    total_actions_considered: int = 0
    actions_pruned: int = 0
    probe_rounds: int = 0

    @property
    def prune_rate(self) -> float:
        if self.total_actions_considered == 0:
            return 0.0
        return self.actions_pruned / self.total_actions_considered


class RegretPruningCFRTrainer:
    """CFR trainer with regret-based pruning.

    Actions with cumulative regret below `prune_threshold` are skipped
    during traversal. A probe round every `probe_interval` iterations
    re-evaluates all actions (including pruned ones) to avoid permanently
    discarding actions that may become relevant.

    This is the single most impactful optimization for tabular CFR,
    providing ~10x speedup with negligible loss in convergence quality.
    """

    def __init__(
        self,
        *,
        small_blind: int = 1,
        big_blind: int = 2,
        starting_stack: int = 100,
        seed: int = 42,
        mode: str = "dcfr",
        prune_threshold: float = -300_000_000,
        probe_interval: int = 10,
        min_iterations_before_pruning: int = 200,
    ) -> None:
        self.engine = GameEngine(small_blind=small_blind, big_blind=big_blind)
        self.starting_stack = starting_stack
        self.rng = random.Random(seed)
        self.cfr_state = CFRState()
        self.mode = mode
        self.prune_threshold = prune_threshold
        self.probe_interval = probe_interval
        self.min_iterations_before_pruning = min_iterations_before_pruning
        self.stats = PruningStats()

    def _should_prune(self, info_set: str, action: ActionType) -> bool:
        """Check if an action should be pruned based on cumulative regret."""
        if self.cfr_state.iterations < self.min_iterations_before_pruning:
            return False
        if info_set not in self.cfr_state.cumulative_regret:
            return False
        regret = self.cfr_state.cumulative_regret[info_set].get(action.value, 0.0)
        return regret < self.prune_threshold

    def _is_probe_round(self) -> bool:
        """Check if this is a probe round (explore all actions)."""
        return self.cfr_state.iterations % self.probe_interval == 0

    def train(self, iterations: int = 10000) -> CFRState:
        """Run CFR with regret-based pruning."""
        for i in range(iterations):
            seed = self.rng.randint(0, 2**31)
            runtime = self.engine.start_new_hand(
                stacks=(self.starting_stack, self.starting_stack),
                button_seat=i % 2,
                seed=seed,
            )

            is_probe = self._is_probe_round()
            if is_probe:
                self.stats.probe_rounds += 1

            if self.mode == "mccfr":
                traverser = i % 2
                self._mccfr_pruned(runtime, traverser, is_probe)
            else:
                self._cfr_pruned(runtime, {0: 1.0, 1: 1.0}, is_probe)

            self.cfr_state.iterations += 1

            if self.mode == "dcfr" and (i + 1) % 100 == 0:
                self.cfr_state.apply_dcfr_discount()

            if (i + 1) % 1000 == 0:
                logger.info(
                    "Pruned CFR iter %d/%d — %d info sets, prune_rate=%.1f%%",
                    i + 1, iterations, len(self.cfr_state.strategy_sum),
                    self.stats.prune_rate * 100,
                )

        return self.cfr_state

    def _cfr_pruned(
        self,
        runtime: HandRuntime,
        reach_probs: dict[int, float],
        is_probe: bool,
    ) -> dict[int, float]:
        """CFR traversal with regret-based pruning."""
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
            self.stats.total_actions_considered += 1
            prob = strategy.probabilities.get(action, 0.0)

            # Prune if not probe round and regret is sufficiently negative
            if not is_probe and self._should_prune(info_set, action):
                self.stats.actions_pruned += 1
                continue

            if prob <= 0 and self.cfr_state.iterations > 0 and not is_probe:
                continue

            child_runtime = self._clone_and_act(runtime, action)
            if child_runtime is None:
                continue

            new_reach = dict(reach_probs)
            new_reach[seat] = reach_probs[seat] * max(prob, 1e-6)
            child_utils = self._cfr_pruned(child_runtime, new_reach, is_probe)

            action_utilities[action] = child_utils.get(seat, 0.0)
            node_utility += prob * action_utilities[action]

        if action_utilities:
            self.cfr_state.update(info_set, strategy, action_utilities, node_utility)

        return {seat: node_utility, opponent: -node_utility}

    def _mccfr_pruned(
        self,
        runtime: HandRuntime,
        traverser: int,
        is_probe: bool,
    ) -> float:
        """MCCFR with external sampling + regret-based pruning."""
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
            return self._mccfr_pruned(child, traverser, is_probe)

        # Traverser: explore actions (with pruning)
        action_utilities: dict[ActionType, float] = {}
        for action in legal:
            self.stats.total_actions_considered += 1

            if not is_probe and self._should_prune(info_set, action):
                self.stats.actions_pruned += 1
                continue

            child = self._clone_and_act(runtime, action)
            if child is None:
                action_utilities[action] = 0.0
                continue
            action_utilities[action] = self._mccfr_pruned(child, traverser, is_probe)

        if not action_utilities:
            return 0.0

        node_utility = sum(
            strategy.probabilities.get(a, 0.0) * u
            for a, u in action_utilities.items()
        )

        self.cfr_state.update(info_set, strategy, action_utilities, node_utility)
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
