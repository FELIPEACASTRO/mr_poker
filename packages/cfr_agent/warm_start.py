"""Warm Starting CFR — Initialize CFR with a prior strategy.

Instead of starting from uniform random, warm starting seeds the
cumulative regrets so the initial strategy matches a provided prior.
This accelerates convergence by ~2-5x when a reasonable prior exists.

Reference: Brown & Sandholm (2016) "Strategy-Based Warm Starting for
Regret Minimization in Games"
"""

from __future__ import annotations

import logging
from typing import Any

from packages.common.types import ActionType
from packages.strategy.mixed import ActionDistribution
from packages.cfr_agent.trainer import CFRState

logger = logging.getLogger(__name__)


def warm_start_from_strategy(
    cfr_state: CFRState,
    initial_strategy: dict[str, dict[ActionType, float]],
    *,
    regret_scale: float = 100.0,
) -> CFRState:
    """Initialize CFR state from a pre-existing strategy.

    Converts the initial strategy into positive cumulative regrets so that
    regret matching reproduces the given strategy. This costs essentially
    nothing (no traversal) and can significantly reduce iterations needed.

    Args:
        cfr_state: The CFR state to initialize (modified in place).
        initial_strategy: Mapping info_set -> {action: probability}.
        regret_scale: Scale factor for initial regrets. Higher values
            make the initial strategy "stickier" (harder to override).

    Returns:
        The modified CFR state.
    """
    for info_set, action_probs in initial_strategy.items():
        if not action_probs:
            continue

        # Convert probabilities to positive regrets that reproduce this strategy
        # via regret matching: P(a) = max(0, R(a)) / sum(max(0, R(a)))
        # Setting R(a) = prob(a) * scale achieves this
        regrets: dict[str, float] = {}
        strat_sums: dict[str, float] = {}

        for action, prob in action_probs.items():
            key = action.value if isinstance(action, ActionType) else str(action)
            regrets[key] = max(0.0, prob) * regret_scale
            strat_sums[key] = max(0.0, prob)

        cfr_state.cumulative_regret[info_set] = regrets
        cfr_state.strategy_sum[info_set] = strat_sums

    logger.info(
        "Warm started CFR with %d info sets (regret_scale=%.1f)",
        len(initial_strategy), regret_scale,
    )
    return cfr_state


def warm_start_from_cfr_state(
    target: CFRState,
    source: CFRState,
    *,
    decay: float = 0.5,
) -> CFRState:
    """Initialize CFR from a previously trained CFR state.

    Copies regrets and strategy sums from the source, optionally
    decaying them to allow the new training to adapt.

    Args:
        target: The new CFR state to initialize.
        source: A previously trained CFR state.
        decay: Decay factor applied to source values (0.0 = fresh start, 1.0 = full copy).

    Returns:
        The initialized target state.
    """
    for info_set, regrets in source.cumulative_regret.items():
        target.cumulative_regret[info_set] = {
            k: v * decay for k, v in regrets.items()
        }

    for info_set, strats in source.strategy_sum.items():
        target.strategy_sum[info_set] = {
            k: v * decay for k, v in strats.items()
        }

    logger.info(
        "Warm started from existing CFR state: %d info sets, decay=%.2f",
        len(source.cumulative_regret), decay,
    )
    return target


def extract_strategy_from_cfr(
    cfr_state: CFRState,
    legal_actions_map: dict[str, set[ActionType]] | None = None,
) -> dict[str, dict[ActionType, float]]:
    """Extract the average strategy from a trained CFR state.

    Useful for converting a trained model to a warm-start prior
    or for analysis.

    Args:
        cfr_state: Trained CFR state.
        legal_actions_map: Optional mapping of info_set -> legal actions.
            If not provided, all actions with positive strategy sum are included.

    Returns:
        Mapping info_set -> {action: probability}.
    """
    strategies: dict[str, dict[ActionType, float]] = {}

    for info_set, sums in cfr_state.strategy_sum.items():
        action_probs: dict[ActionType, float] = {}
        total = 0.0

        for key, val in sums.items():
            v = max(0.0, val)
            try:
                action = ActionType(key)
            except ValueError:
                continue

            if legal_actions_map and info_set in legal_actions_map:
                if action not in legal_actions_map[info_set]:
                    continue

            action_probs[action] = v
            total += v

        if total > 0:
            action_probs = {a: p / total for a, p in action_probs.items()}

        if action_probs:
            strategies[info_set] = action_probs

    return strategies
