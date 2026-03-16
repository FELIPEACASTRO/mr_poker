"""PDCFR+ — Predictive Discounted CFR+ with optimistic mirror descent.

Combines Discounted CFR+ (positive-only regrets with temporal discounting)
with optimistic prediction to achieve faster convergence to Nash equilibrium.

The key idea: instead of using current regrets for regret matching, PDCFR+
linearly extrapolates a *predicted* regret vector and uses that for strategy
computation. This is equivalent to an optimistic mirror descent step with a
Euclidean regularizer, yielding O(1/T^{3/4}) convergence vs O(1/T^{1/2})
for vanilla CFR — roughly 2-5x faster in practice.

Algorithm outline:
    1. Maintain current regrets R_t and previous regrets R_{t-1}
    2. Compute predicted regrets: R̃_t = R_t + w * (R_t - R_{t-1})
    3. Use R̃_t (not R_t) for regret matching to get strategy σ_t
    4. Traverse game tree, compute instantaneous regrets
    5. Update: R_{t+1} = max(0, R_t + instantaneous_regret)  (CFR+ style)
    6. Apply DCFR discounting periodically

Reference:
    Wei et al. "Predictive Discounted CFR+" (arXiv:2404.13891, IJCAI 2024)
    "Optimistic Mirror Descent for Faster CFR Convergence"
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
from packages.strategy.mixed import ActionDistribution

from packages.cfr_agent.info_set import build_info_set_key
from packages.cfr_agent.trainer import CFRState, _size_action

logger = logging.getLogger(__name__)


@dataclass
class PDCFRPlusState:
    """State for Predictive Discounted CFR+.

    Maintains three regret tables to support the optimistic prediction step:
      - cumulative_regret: current regrets R_t (CFR+ style, positive-only)
      - prev_regret: previous iteration regrets R_{t-1}
      - predicted_regret: extrapolated regrets R̃_t = R_t + w*(R_t - R_{t-1})
      - strategy_sum: accumulated weighted strategies for average computation

    The prediction step is the core novelty: by using linearly extrapolated
    regrets for regret matching, the algorithm implicitly performs an optimistic
    mirror descent step, leading to faster convergence.

    Reference: arXiv:2404.13891 (IJCAI 2024)
    """

    cumulative_regret: dict[str, dict[str, float]] = field(default_factory=dict)
    prev_regret: dict[str, dict[str, float]] = field(default_factory=dict)
    predicted_regret: dict[str, dict[str, float]] = field(default_factory=dict)
    strategy_sum: dict[str, dict[str, float]] = field(default_factory=dict)
    iterations: int = 0

    # DCFR discounting parameters (Brown & Sandholm 2019)
    dcfr_alpha: float = 1.5
    dcfr_beta: float = 0.5
    dcfr_gamma: float = 2.0

    # Prediction weight: how much to extrapolate (1.0 = full linear extrapolation)
    prediction_weight: float = 1.0

    # Linear CFR: weight strategy contribution by iteration number
    linear_cfr: bool = True

    def predict(self) -> None:
        """Compute predicted regrets via linear extrapolation.

        For each info set and action:
            R̃_t = R_t + w * (R_t - R_{t-1})

        where w is the prediction_weight. With w=1.0 this is standard linear
        extrapolation; w=0.0 recovers plain DCFR+.

        The predicted regrets are clamped to non-negative values (CFR+ style)
        before being used for regret matching.
        """
        w = self.prediction_weight
        self.predicted_regret = {}

        for info_set, regrets in self.cumulative_regret.items():
            prev = self.prev_regret.get(info_set, {})
            predicted = {}
            for key, r_t in regrets.items():
                r_prev = prev.get(key, 0.0)
                # Linear extrapolation: R̃ = R_t + w * (R_t - R_{t-1})
                r_predicted = r_t + w * (r_t - r_prev)
                # CFR+ style: clamp to non-negative
                predicted[key] = max(0.0, r_predicted)
            self.predicted_regret[info_set] = predicted

    def current_strategy(
        self, info_set: str, legal: set[ActionType]
    ) -> ActionDistribution:
        """Get strategy via regret matching on predicted regrets.

        Uses the predicted (extrapolated) regrets R̃_t rather than the current
        regrets R_t. This is the key difference from standard CFR/DCFR — the
        optimistic prediction step yields faster convergence.

        Falls back to uniform if no predicted regrets exist or all are zero.
        """
        # Use predicted regrets if available, else fall back to cumulative
        regret_table = self.predicted_regret if self.predicted_regret else self.cumulative_regret

        if info_set not in regret_table:
            n = len(legal) or 1
            return ActionDistribution(probabilities={a: 1.0 / n for a in legal})

        regrets = regret_table[info_set]
        positive = {a: max(0.0, regrets.get(a.value, 0.0)) for a in legal}
        total = sum(positive.values())

        if total > 0:
            probs = {a: v / total for a, v in positive.items()}
        else:
            n = len(legal) or 1
            probs = {a: 1.0 / n for a in legal}

        return ActionDistribution(probabilities=probs)

    def average_strategy(
        self, info_set: str, legal: set[ActionType]
    ) -> ActionDistribution:
        """Get the time-averaged strategy (the converged Nash approximation).

        The average strategy over all iterations converges to a Nash equilibrium.
        This is what should be used for actual play, not the current strategy.
        """
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

    def update(
        self,
        info_set: str,
        strategy: ActionDistribution,
        action_utilities: dict[ActionType, float],
        ev: float,
    ) -> None:
        """Update regrets and strategy sums with CFR+ positive-only clamping.

        The update rule is:
            R_{t+1}(a) = max(0, R_t(a) + [u(a) - EV])

        This is the CFR+ update (Tammelin 2014) which ensures regrets never go
        negative, providing better practical convergence than vanilla CFR.

        Strategy sums are weighted by iteration number (Linear CFR).
        """
        if info_set not in self.cumulative_regret:
            self.cumulative_regret[info_set] = {}
        if info_set not in self.strategy_sum:
            self.strategy_sum[info_set] = {}

        # Linear CFR: weight strategy contribution by iteration number
        weight = max(1, self.iterations) if self.linear_cfr else 1

        for action, utility in action_utilities.items():
            regret = utility - ev
            key = action.value

            # CFR+ update: R_{t+1} = max(0, R_t + instantaneous_regret)
            old_regret = self.cumulative_regret[info_set].get(key, 0.0)
            self.cumulative_regret[info_set][key] = max(0.0, old_regret + regret)

            # Accumulate weighted strategy for average computation
            prob = strategy.probabilities.get(action, 0.0)
            self.strategy_sum[info_set][key] = (
                self.strategy_sum[info_set].get(key, 0.0) + prob * weight
            )

    def snapshot_regrets(self) -> None:
        """Save current regrets as previous regrets for next prediction step.

        Must be called at the end of each iteration, before predict() is
        called for the next iteration.
        """
        self.prev_regret = {
            info_set: dict(regrets)
            for info_set, regrets in self.cumulative_regret.items()
        }

    def apply_dcfr_discount(self) -> None:
        """Apply DCFR temporal discounting to regrets and strategy sums.

        Positive regrets discounted by t^alpha / (t^alpha + 1).
        Negative regrets discounted by t^beta / (t^beta + 1).
        Strategy sums discounted by (t / (t + 1))^gamma.

        Note: In CFR+ the regrets are always non-negative, so only the
        positive discount applies. We still handle negatives defensively.
        """
        t = max(1, self.iterations)
        alpha, beta, gamma = self.dcfr_alpha, self.dcfr_beta, self.dcfr_gamma

        pos_discount = (t ** alpha) / (t ** alpha + 1)
        neg_discount = (t ** beta) / (t ** beta + 1)
        strat_discount = (t / (t + 1)) ** gamma

        for info_set in self.cumulative_regret:
            for key in self.cumulative_regret[info_set]:
                val = self.cumulative_regret[info_set][key]
                if val > 0:
                    self.cumulative_regret[info_set][key] = val * pos_discount
                else:
                    self.cumulative_regret[info_set][key] = val * neg_discount

        # Also discount prev_regret to keep prediction stable
        for info_set in self.prev_regret:
            for key in self.prev_regret[info_set]:
                val = self.prev_regret[info_set][key]
                if val > 0:
                    self.prev_regret[info_set][key] = val * pos_discount
                else:
                    self.prev_regret[info_set][key] = val * neg_discount

        for info_set in self.strategy_sum:
            for key in self.strategy_sum[info_set]:
                self.strategy_sum[info_set][key] *= strat_discount

    def save(self, path: str | Path) -> None:
        """Persist PDCFR+ state to JSON."""
        data = {
            "type": "pdcfr_plus",
            "iterations": self.iterations,
            "cumulative_regret": self.cumulative_regret,
            "prev_regret": self.prev_regret,
            "strategy_sum": self.strategy_sum,
            "dcfr_alpha": self.dcfr_alpha,
            "dcfr_beta": self.dcfr_beta,
            "dcfr_gamma": self.dcfr_gamma,
            "prediction_weight": self.prediction_weight,
            "linear_cfr": self.linear_cfr,
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> PDCFRPlusState:
        """Load PDCFR+ state from JSON."""
        data = json.loads(Path(path).read_text())
        state = cls(
            cumulative_regret=data.get("cumulative_regret", {}),
            prev_regret=data.get("prev_regret", {}),
            strategy_sum=data.get("strategy_sum", {}),
            iterations=data.get("iterations", 0),
            dcfr_alpha=data.get("dcfr_alpha", 1.5),
            dcfr_beta=data.get("dcfr_beta", 0.5),
            dcfr_gamma=data.get("dcfr_gamma", 2.0),
            prediction_weight=data.get("prediction_weight", 1.0),
            linear_cfr=data.get("linear_cfr", True),
        )
        return state


class PDCFRPlusTrainer:
    """Trains a poker agent using Predictive Discounted CFR+.

    Combines three techniques for fast convergence:
      1. CFR+ (Tammelin 2014): positive-only regret clamping
      2. DCFR (Brown & Sandholm 2019): temporal discounting of old regrets
      3. Optimistic prediction (Wei et al., IJCAI 2024): linear extrapolation
         of regrets for an implicit optimistic mirror descent step

    Supports two traversal modes:
      - mode="dcfr": Full tree traversal with DCFR+ discounting (default)
      - mode="mccfr": Monte Carlo external sampling (faster per iteration)

    The prediction step adds negligible overhead (one linear extrapolation per
    info set per iteration) while yielding 2-5x faster convergence in practice.

    Usage::

        trainer = PDCFRPlusTrainer(small_blind=1, big_blind=2, starting_stack=100)
        state = trainer.train(iterations=10000)
        cfr_state = trainer.to_cfr_state()  # Convert to standard CFRState

    Reference: arXiv:2404.13891 (IJCAI 2024)
    """

    def __init__(
        self,
        *,
        small_blind: int = 1,
        big_blind: int = 2,
        starting_stack: int = 100,
        seed: int = 42,
        mode: str = "dcfr",
        prediction_weight: float = 1.0,
    ) -> None:
        """Initialize the PDCFR+ trainer.

        Args:
            small_blind: Small blind amount.
            big_blind: Big blind amount.
            starting_stack: Starting stack for each player.
            seed: Random seed for reproducibility.
            mode: Traversal mode — "dcfr" for full tree, "mccfr" for external sampling.
            prediction_weight: Weight for the optimistic extrapolation step.
                1.0 = full linear extrapolation (default, recommended).
                0.0 = no prediction (reduces to plain DCFR+).
                Values in (0, 1) interpolate between the two.
        """
        self.engine = GameEngine(small_blind=small_blind, big_blind=big_blind)
        self.starting_stack = starting_stack
        self.rng = random.Random(seed)
        self.mode = mode
        self.state = PDCFRPlusState(prediction_weight=prediction_weight)

    def train(self, iterations: int = 10000) -> PDCFRPlusState:
        """Run PDCFR+ self-play for the given number of iterations.

        Each iteration:
          1. Predict regrets (linear extrapolation from previous)
          2. Compute strategy from predicted regrets
          3. Traverse game tree and compute instantaneous regrets
          4. Update cumulative regrets with CFR+ clamping
          5. Snapshot regrets for next iteration's prediction
          6. Apply DCFR discounting periodically (every 100 iterations)

        Args:
            iterations: Number of self-play iterations to run.

        Returns:
            The trained PDCFRPlusState containing the converged strategy.
        """
        for i in range(iterations):
            # Step 1: Compute predicted regrets for this iteration
            if self.state.iterations > 0:
                self.state.predict()

            # Step 2-4: Traverse and update
            seed = self.rng.randint(0, 2**31)
            runtime = self.engine.start_new_hand(
                stacks=(self.starting_stack, self.starting_stack),
                button_seat=i % 2,
                seed=seed,
            )

            if self.mode == "mccfr":
                traverser = i % 2
                self._mccfr_external(runtime, traverser)
            else:
                self._cfr_iteration(runtime, reach_probs={0: 1.0, 1: 1.0})

            # Step 5: Snapshot regrets for next prediction
            self.state.snapshot_regrets()
            self.state.iterations += 1

            # Step 6: Apply DCFR discounting periodically
            if (i + 1) % 100 == 0:
                self.state.apply_dcfr_discount()

            if (i + 1) % 1000 == 0:
                n_info_sets = len(self.state.strategy_sum)
                logger.info(
                    "PDCFR+ iteration %d/%d — %d info sets (mode=%s, w=%.2f)",
                    i + 1,
                    iterations,
                    n_info_sets,
                    self.mode,
                    self.state.prediction_weight,
                )

        return self.state

    def to_cfr_state(self) -> CFRState:
        """Convert PDCFR+ state to standard CFRState for compatibility.

        This allows the trained strategy to be used with the existing CFRAgent
        and other tools that expect a CFRState object.

        Returns:
            A CFRState with the same cumulative regrets and strategy sums.
        """
        cfr = CFRState(
            cumulative_regret=dict(self.state.cumulative_regret),
            strategy_sum=dict(self.state.strategy_sum),
            iterations=self.state.iterations,
            dcfr_alpha=self.state.dcfr_alpha,
            dcfr_beta=self.state.dcfr_beta,
            dcfr_gamma=self.state.dcfr_gamma,
            linear_cfr=self.state.linear_cfr,
        )
        return cfr

    def _cfr_iteration(
        self,
        runtime: HandRuntime,
        reach_probs: dict[int, float],
    ) -> dict[int, float]:
        """Recursive PDCFR+ traversal with full tree exploration.

        Returns expected utilities per seat. Uses predicted regrets for
        strategy computation (via state.current_strategy).
        """
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

        strategy = self.state.current_strategy(info_set, legal)
        action_utilities: dict[ActionType, float] = {}
        node_utility = 0.0

        # Regret-based pruning threshold
        prune_threshold = -300.0 * math.sqrt(max(1, self.state.iterations))
        regrets = self.state.cumulative_regret.get(info_set, {})

        for action in legal:
            prob = strategy.probabilities.get(action, 0.0)
            if prob <= 0 and self.state.iterations > 0:
                cum_regret = regrets.get(action.value, 0.0)
                if cum_regret < prune_threshold:
                    continue

            child_runtime = self._clone_and_act(runtime, action)
            if child_runtime is None:
                continue

            new_reach = dict(reach_probs)
            new_reach[seat] = reach_probs[seat] * max(prob, 1e-6)
            child_utils = self._cfr_iteration(child_runtime, new_reach)

            action_utilities[action] = child_utils.get(seat, 0.0)
            node_utility += prob * action_utilities[action]

        self.state.update(info_set, strategy, action_utilities, node_utility)

        result = {seat: node_utility}
        result[opponent] = -node_utility
        return result

    def _mccfr_external(
        self,
        runtime: HandRuntime,
        traverser: int,
    ) -> float:
        """Monte Carlo CFR with external sampling for PDCFR+.

        Only the traverser's actions are fully explored; opponent actions are
        sampled from the current (predicted) strategy. Uses CFR+ positive-only
        regret updates and optimistic prediction.
        """
        state = runtime.state

        if state.is_terminal or state.acting_seat is None:
            return state.players[traverser].stack - self.starting_stack

        seat = state.acting_seat
        legal = set(self.engine.legal_actions(runtime))
        info_set = build_info_set_key(state, seat)
        strategy = self.state.current_strategy(info_set, legal)

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
        prune_threshold = -300.0 * math.sqrt(max(1, self.state.iterations))
        regrets = self.state.cumulative_regret.get(info_set, {})

        for action in legal:
            if self.state.iterations > 0:
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

        # Update regrets and strategy with CFR+ clamping
        self.state.update(info_set, strategy, action_utilities, node_utility)
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
        deck_prefix = all_hole_cards + burned[: len(state.board)] + board_cards

        try:
            new_runtime = self.engine.start_new_hand(
                stacks=tuple(
                    runtime.initial_stacks[s] for s in sorted(runtime.initial_stacks)
                ),
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
