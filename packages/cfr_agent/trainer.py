"""CFR self-play trainer with Discounted CFR and Monte Carlo CFR.

Supports three CFR variants:
1. Vanilla CFR — full tree traversal (original, slow)
2. DCFR — Discounted CFR with temporal weighting (used by all pro solvers)
3. MCCFR — Monte Carlo CFR with external sampling (scalable to millions of iterations)

Over many iterations, the average strategy converges to a Nash equilibrium.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from packages.common.types import ActionType
from packages.engine.engine import GameEngine, HandRuntime
from packages.strategy.mixed import ActionDistribution, MixedStrategy

from packages.cfr_agent.info_set import build_info_set_key

logger = logging.getLogger(__name__)

# Actions the CFR agent can choose from (simplified action space)
CFR_ACTIONS = [
    ActionType.FOLD,
    ActionType.CHECK,
    ActionType.CALL,
    ActionType.BET,
    ActionType.RAISE,
    ActionType.ALL_IN,
]


@dataclass
class CFRState:
    """Persistent state of a trained CFR model."""

    cumulative_regret: dict[str, dict[str, float]] = field(default_factory=dict)
    strategy_sum: dict[str, dict[str, float]] = field(default_factory=dict)
    iterations: int = 0

    # DCFR parameters (standard from Brown et al.)
    dcfr_alpha: float = 1.5
    dcfr_beta: float = 0.5
    dcfr_gamma: float = 2.0

    # Linear CFR: weight iteration t by t (instead of uniform)
    linear_cfr: bool = True

    # VAD-CFR state: per-info-set regret volatility tracking
    # Maps info_set -> action_key -> (prev_regret, ema_volatility)
    _vad_volatility: dict[str, dict[str, tuple[float, float]]] = field(
        default_factory=dict
    )
    # VAD-CFR warm-start threshold: no strategy accumulation before this iteration
    vad_warmup: int = 50

    def average_strategy(self, info_set: str, legal: set[ActionType]) -> ActionDistribution:
        """Get the time-averaged strategy (the converged Nash approximation)."""
        if info_set not in self.strategy_sum:
            n = len(legal) or 1
            return ActionDistribution(probabilities={a: 1.0 / n for a in legal})

        sums = self.strategy_sum[info_set]
        total = sum(max(0.0, sums.get(a.value, 0.0)) for a in legal)

        if total > 0:
            probs = {a: max(0.0, sums.get(a.value, 0.0)) / total for a in legal}
        else:
            n = len(legal) or 1
            probs = {a: 1.0 / n for a in legal}

        return ActionDistribution(probabilities=probs)

    def current_strategy(self, info_set: str, legal: set[ActionType]) -> ActionDistribution:
        """Get the current strategy via regret matching (used during training)."""
        if info_set not in self.cumulative_regret:
            n = len(legal) or 1
            return ActionDistribution(probabilities={a: 1.0 / n for a in legal})

        regrets = self.cumulative_regret[info_set]
        positive = {a: max(0.0, regrets.get(a.value, 0.0)) for a in legal}
        total = sum(positive.values())

        if total > 0:
            probs = {a: v / total for a, v in positive.items()}
        else:
            n = len(legal) or 1
            probs = {a: 1.0 / n for a in legal}

        return ActionDistribution(probabilities=probs)

    def update(
        self,
        info_set: str,
        strategy: ActionDistribution,
        action_utilities: dict[ActionType, float],
        ev: float,
    ) -> None:
        """Update cumulative regrets and strategy sums with DCFR discounting."""
        if info_set not in self.cumulative_regret:
            self.cumulative_regret[info_set] = {}
        if info_set not in self.strategy_sum:
            self.strategy_sum[info_set] = {}

        # Linear CFR: weight strategy contribution by iteration number
        weight = max(1, self.iterations) if self.linear_cfr else 1

        for action, utility in action_utilities.items():
            regret = utility - ev
            key = action.value
            self.cumulative_regret[info_set][key] = (
                self.cumulative_regret[info_set].get(key, 0.0) + regret
            )
            prob = strategy.probabilities.get(action, 0.0)
            self.strategy_sum[info_set][key] = (
                self.strategy_sum[info_set].get(key, 0.0) + prob * weight
            )

    def update_vad(
        self,
        info_set: str,
        strategy: ActionDistribution,
        action_utilities: dict[ActionType, float],
        ev: float,
    ) -> None:
        """VAD-CFR update with volatility tracking and consistency-enforced optimism.

        Based on "Discovering Multiagent Learning Algorithms with LLMs" (Li et al., 2026).
        Key differences from DCFR:
        1. Tracks per-action regret volatility (EMA of |delta_regret|)
        2. Applies optimistic bonus to positive regrets proportional to consistency
        3. Strategy accumulation only starts after warm-up phase
        """
        if info_set not in self.cumulative_regret:
            self.cumulative_regret[info_set] = {}
        if info_set not in self.strategy_sum:
            self.strategy_sum[info_set] = {}
        if info_set not in self._vad_volatility:
            self._vad_volatility[info_set] = {}

        # EMA decay for volatility tracking
        vol_alpha = 0.1

        for action, utility in action_utilities.items():
            regret = utility - ev
            key = action.value

            prev_cumulative = self.cumulative_regret[info_set].get(key, 0.0)

            # Track volatility: EMA of absolute regret changes
            prev_regret, prev_vol = self._vad_volatility[info_set].get(key, (0.0, 0.0))
            delta = abs(regret - prev_regret)
            new_vol = (1.0 - vol_alpha) * prev_vol + vol_alpha * delta
            self._vad_volatility[info_set][key] = (regret, new_vol)

            # Consistency-enforced optimism: if regret is consistently positive
            # (low volatility, positive cumulative), add a small bonus
            optimism_bonus = 0.0
            if prev_cumulative > 0 and new_vol < abs(regret) * 0.5:
                # Low volatility + positive regret = consistent signal → boost
                consistency = max(0.0, 1.0 - new_vol / max(abs(regret), 1e-6))
                optimism_bonus = regret * 0.1 * consistency

            self.cumulative_regret[info_set][key] = prev_cumulative + regret + optimism_bonus

            # Hard warm-start: only accumulate strategy after warm-up
            if self.iterations >= self.vad_warmup:
                prob = strategy.probabilities.get(action, 0.0)
                weight = max(1, self.iterations - self.vad_warmup)
                self.strategy_sum[info_set].setdefault(key, 0.0)
                self.strategy_sum[info_set][key] += prob * weight

    def apply_vad_discount(self) -> None:
        """Apply volatility-adaptive discounting to regrets and strategy sums.

        Unlike DCFR's fixed discount schedule, VAD-CFR adapts per info-set:
        - High-volatility info sets get stronger discounting (less memory)
        - Low-volatility (stable) info sets retain more history
        - Strategy sums use iteration-weighted discounting after warm-up
        """
        t = max(1, self.iterations)
        base_alpha = self.dcfr_alpha
        base_beta = self.dcfr_beta
        base_gamma = self.dcfr_gamma

        for info_set in list(self.cumulative_regret.keys()):
            vol_data = self._vad_volatility.get(info_set, {})

            # Compute mean volatility for this info set
            vols = [v for _, v in vol_data.values()] if vol_data else [0.0]
            mean_vol = sum(vols) / len(vols) if vols else 0.0

            # Adaptive exponents: high volatility → stronger discount (lower exponent)
            # vol_factor in [0.5, 1.5]: >1 means stable, <1 means volatile
            vol_factor = max(0.5, min(1.5, 1.0 / max(mean_vol + 0.1, 0.1)))

            alpha = base_alpha * vol_factor
            beta = base_beta * vol_factor
            gamma = base_gamma * vol_factor

            pos_discount = (t ** alpha) / (t ** alpha + 1)
            neg_discount = (t ** beta) / (t ** beta + 1)

            for key in self.cumulative_regret[info_set]:
                val = self.cumulative_regret[info_set][key]
                if val > 0:
                    self.cumulative_regret[info_set][key] = val * pos_discount
                else:
                    self.cumulative_regret[info_set][key] = val * neg_discount

        # Strategy sum discount (global, but adjusted by iteration vs warm-up)
        if t > self.vad_warmup:
            effective_t = t - self.vad_warmup
            strat_discount = (effective_t / (effective_t + 1)) ** base_gamma
            for info_set in self.strategy_sum:
                for key in self.strategy_sum[info_set]:
                    self.strategy_sum[info_set][key] *= strat_discount

    def apply_dcfr_discount(self) -> None:
        """Apply DCFR temporal discounting to regrets and strategy sums.

        This is the key innovation over vanilla CFR — recent iterations
        are weighted more heavily, leading to 3-5x faster convergence.
        Based on Brown & Sandholm (2019) "Solving Imperfect-Information Games
        via Discounted Regret Minimization".
        """
        t = max(1, self.iterations)
        alpha, beta, gamma = self.dcfr_alpha, self.dcfr_beta, self.dcfr_gamma

        # Positive regret discount: t^alpha / (t^alpha + 1)
        pos_discount = (t ** alpha) / (t ** alpha + 1)
        # Negative regret discount: t^beta / (t^beta + 1)
        neg_discount = (t ** beta) / (t ** beta + 1)
        # Strategy sum discount: (t / (t + 1))^gamma
        strat_discount = (t / (t + 1)) ** gamma

        for info_set in self.cumulative_regret:
            for key in self.cumulative_regret[info_set]:
                val = self.cumulative_regret[info_set][key]
                if val > 0:
                    self.cumulative_regret[info_set][key] = val * pos_discount
                else:
                    self.cumulative_regret[info_set][key] = val * neg_discount

        for info_set in self.strategy_sum:
            for key in self.strategy_sum[info_set]:
                self.strategy_sum[info_set][key] *= strat_discount

    def save(self, path: str | Path) -> None:
        """Persist CFR state to JSON."""
        data = {
            "iterations": self.iterations,
            "cumulative_regret": self.cumulative_regret,
            "strategy_sum": self.strategy_sum,
            "dcfr_alpha": self.dcfr_alpha,
            "dcfr_beta": self.dcfr_beta,
            "dcfr_gamma": self.dcfr_gamma,
            "linear_cfr": self.linear_cfr,
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> CFRState:
        """Load CFR state from JSON."""
        data = json.loads(Path(path).read_text())
        return cls(
            cumulative_regret=data.get("cumulative_regret", {}),
            strategy_sum=data.get("strategy_sum", {}),
            iterations=data.get("iterations", 0),
            dcfr_alpha=data.get("dcfr_alpha", 1.5),
            dcfr_beta=data.get("dcfr_beta", 0.5),
            dcfr_gamma=data.get("dcfr_gamma", 2.0),
            linear_cfr=data.get("linear_cfr", True),
        )


def _size_action(
    action: ActionType, state: Any, player: Any, engine: GameEngine
) -> int:
    """Choose a reasonable sizing for the given action type."""
    if action in {ActionType.CHECK, ActionType.FOLD}:
        return 0
    if action == ActionType.CALL:
        return max(0, state.current_bet - player.invested_this_round)
    if action == ActionType.ALL_IN:
        return player.stack
    if action == ActionType.BET:
        size = max(engine.big_blind, int(max(state.pot, 2) * 0.5))
        return min(size, player.stack)
    if action == ActionType.RAISE:
        min_raise = state.min_raise_to or (state.current_bet + engine.big_blind)
        target = max(min_raise, state.current_bet + engine.big_blind * 2)
        max_raise = player.invested_this_round + player.stack
        # Clamp to [min_raise, max_raise] to avoid InvalidActionError
        target = max(target, min_raise)
        if target > max_raise:
            return max_raise
        return target
    return 0


class CFRTrainer:
    """Trains a CFR agent via self-play iterations.

    Supports four modes:
    - mode="vanilla": Full tree traversal (original CFR)
    - mode="dcfr": Discounted CFR with temporal weighting (default, best)
    - mode="mccfr": Monte Carlo CFR with external sampling (fastest)
    - mode="vadcfr": Volatility-Adaptive Discounted CFR (experimental, best convergence)
    """

    def __init__(
        self,
        *,
        small_blind: int = 1,
        big_blind: int = 2,
        starting_stack: int = 100,
        seed: int = 42,
        mode: str = "dcfr",
        warm_start_strategy: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.engine = GameEngine(small_blind=small_blind, big_blind=big_blind)
        self.starting_stack = starting_stack
        self.rng = random.Random(seed)
        self.cfr_state = CFRState()
        self.mode = mode

        # Warm start: initialize strategy_sum from provided strategy
        if warm_start_strategy is not None:
            for info_set, action_probs in warm_start_strategy.items():
                self.cfr_state.strategy_sum[info_set] = dict(action_probs)

    def train(self, iterations: int = 10000) -> CFRState:
        """Run CFR self-play for the given number of iterations."""
        for i in range(iterations):
            seed = self.rng.randint(0, 2**31)
            runtime = self.engine.start_new_hand(
                stacks=(self.starting_stack, self.starting_stack),
                button_seat=i % 2,
                seed=seed,
            )

            if self.mode == "mccfr":
                # External sampling MCCFR — sample opponent actions
                traverser = i % 2
                self._mccfr_external(runtime, traverser)
            elif self.mode == "vadcfr":
                # VAD-CFR — full traversal with volatility-adaptive updates
                self._cfr_iteration(runtime, reach_probs={0: 1.0, 1: 1.0}, use_vad=True)
            else:
                # Vanilla / DCFR — full tree traversal
                self._cfr_iteration(runtime, reach_probs={0: 1.0, 1: 1.0})

            self.cfr_state.iterations += 1

            # Apply discounting periodically
            if self.mode == "dcfr" and (i + 1) % 100 == 0:
                self.cfr_state.apply_dcfr_discount()
            elif self.mode == "vadcfr" and (i + 1) % 100 == 0:
                self.cfr_state.apply_vad_discount()

            if (i + 1) % 1000 == 0:
                n_info_sets = len(self.cfr_state.strategy_sum)
                logger.info(
                    "CFR iteration %d/%d — %d info sets (mode=%s)",
                    i + 1, iterations, n_info_sets, self.mode,
                )

        return self.cfr_state

    def _cfr_iteration(
        self,
        runtime: HandRuntime,
        reach_probs: dict[int, float],
        use_vad: bool = False,
    ) -> dict[int, float]:
        """Recursive CFR traversal. Returns expected utilities per seat."""
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

        # Regret-based pruning threshold
        prune_threshold = -300.0 * math.sqrt(max(1, self.cfr_state.iterations))
        regrets = self.cfr_state.cumulative_regret.get(info_set, {})

        for action in legal:
            prob = strategy.probabilities.get(action, 0.0)
            if prob <= 0 and self.cfr_state.iterations > 0:
                # Also check regret-based pruning
                cum_regret = regrets.get(action.value, 0.0)
                if cum_regret < prune_threshold:
                    continue

            child_runtime = self._clone_and_act(runtime, action)
            if child_runtime is None:
                continue

            new_reach = dict(reach_probs)
            new_reach[seat] = reach_probs[seat] * max(prob, 1e-6)
            child_utils = self._cfr_iteration(child_runtime, new_reach, use_vad=use_vad)

            action_utilities[action] = child_utils.get(seat, 0.0)
            node_utility += prob * action_utilities[action]

        if use_vad:
            self.cfr_state.update_vad(info_set, strategy, action_utilities, node_utility)
        else:
            self.cfr_state.update(info_set, strategy, action_utilities, node_utility)

        result = {seat: node_utility}
        result[opponent] = -node_utility
        return result

    def _mccfr_external(
        self,
        runtime: HandRuntime,
        traverser: int,
    ) -> float:
        """Monte Carlo CFR with external sampling.

        Only the traverser's actions are explored (all of them).
        Opponent actions are sampled from the current strategy.
        This is ~100x faster per iteration than vanilla CFR.
        """
        state = runtime.state

        if state.is_terminal or state.acting_seat is None:
            return state.players[traverser].stack - self.starting_stack

        seat = state.acting_seat
        legal = set(self.engine.legal_actions(runtime))
        info_set = build_info_set_key(state, seat)
        strategy = self.cfr_state.current_strategy(info_set, legal)

        if seat != traverser:
            # Opponent node: sample one action from current strategy
            sampled = strategy.sample(self.rng)
            if sampled not in legal:
                sampled = self.rng.choice(list(legal))
            child = self._clone_and_act(runtime, sampled)
            if child is None:
                return 0.0
            return self._mccfr_external(child, traverser)

        # Traverser node: explore ALL actions (with regret-based pruning)
        action_utilities: dict[ActionType, float] = {}
        prune_threshold = -300.0 * math.sqrt(max(1, self.cfr_state.iterations))
        regrets = self.cfr_state.cumulative_regret.get(info_set, {})

        for action in legal:
            # Regret-based pruning: skip actions with sufficiently negative regret
            if self.cfr_state.iterations > 0:
                cum_regret = regrets.get(action.value, 0.0)
                if cum_regret < prune_threshold:
                    continue

            child = self._clone_and_act(runtime, action)
            if child is None:
                action_utilities[action] = 0.0
                continue
            action_utilities[action] = self._mccfr_external(child, traverser)

        # If all actions were pruned, fall back to exploring all
        if not action_utilities:
            for action in legal:
                child = self._clone_and_act(runtime, action)
                if child is None:
                    action_utilities[action] = 0.0
                    continue
                action_utilities[action] = self._mccfr_external(child, traverser)

        # Compute node EV
        node_utility = sum(
            strategy.probabilities.get(a, 0.0) * u
            for a, u in action_utilities.items()
        )

        # Update regrets and strategy
        self.cfr_state.update(info_set, strategy, action_utilities, node_utility)
        return node_utility

    def _clone_and_act(
        self, runtime: HandRuntime, action: ActionType
    ) -> HandRuntime | None:
        """Clone the game state and apply one action. Returns None if illegal."""
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
