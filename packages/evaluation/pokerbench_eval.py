"""PokerBench Evaluator — Measure agent accuracy against GTO solver decisions.

Evaluates any decision function against PokerBench scenarios (11K+ GTO-solver
optimal decisions for 6-handed NLHE).

Metrics computed:
- Overall action accuracy (exact match)
- Per-street accuracy (pre_flop, flop, turn, river)
- Per-action accuracy (fold, check, call, bet, raise, all_in)
- Confusion matrix (predicted vs actual)
- Aggression alignment (how often agent matches aggression direction)

Usage::

    evaluator = PokerBenchEvaluator()
    results = evaluator.evaluate(scenarios, decision_fn)
    print(results.accuracy, results.street_accuracy)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from packages.dataset_builder.pokerbench_adapter import (
    parse_pokerbench_instruction,
    _parse_action_output,
)


# Action groupings for aggression alignment
_AGGRESSIVE = frozenset({"bet", "raise", "all_in"})
_PASSIVE = frozenset({"check", "call", "fold"})


@dataclass
class PokerBenchResult:
    """Results from evaluating against PokerBench scenarios."""

    total: int = 0
    correct: int = 0
    accuracy: float = 0.0

    # Per-street breakdown
    street_accuracy: dict[str, float] = field(default_factory=dict)
    street_counts: dict[str, int] = field(default_factory=dict)

    # Per-action breakdown
    action_accuracy: dict[str, float] = field(default_factory=dict)
    action_counts: dict[str, int] = field(default_factory=dict)

    # Confusion matrix: actual -> predicted -> count
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)

    # Aggression alignment: does agent match aggressive/passive direction?
    aggression_alignment: float = 0.0

    # Parse failures
    skipped: int = 0

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"PokerBench Evaluation: {self.correct}/{self.total} = {self.accuracy:.1%}",
            f"  Aggression alignment: {self.aggression_alignment:.1%}",
            f"  Skipped (unparseable): {self.skipped}",
            "",
            "  Per-street:",
        ]
        for street in ("pre_flop", "flop", "turn", "river"):
            acc = self.street_accuracy.get(street, 0.0)
            cnt = self.street_counts.get(street, 0)
            lines.append(f"    {street:10s}: {acc:.1%} ({cnt} spots)")

        lines.append("")
        lines.append("  Per-action:")
        for action in ("fold", "check", "call", "bet", "raise", "all_in"):
            acc = self.action_accuracy.get(action, 0.0)
            cnt = self.action_counts.get(action, 0)
            if cnt > 0:
                lines.append(f"    {action:8s}: {acc:.1%} ({cnt} spots)")

        return "\n".join(lines)


# Type for decision functions: takes parsed game state, returns action string
DecisionFn = Callable[[dict[str, Any]], str]


class PokerBenchEvaluator:
    """Evaluates a decision function against PokerBench GTO scenarios.

    The decision function receives a parsed game state dict (from
    :func:`parse_pokerbench_instruction`) and returns an action string
    (one of: "fold", "check", "call", "bet", "raise", "all_in").

    Example::

        def my_agent(state: dict) -> str:
            if state["estimated_equity"] > 0.7:
                return "raise"
            if state["to_call"] == 0:
                return "check"
            return "call"

        evaluator = PokerBenchEvaluator()
        results = evaluator.evaluate(scenarios, my_agent)
    """

    def evaluate(
        self,
        scenarios: list[dict[str, str]],
        decision_fn: DecisionFn,
        *,
        max_scenarios: int | None = None,
    ) -> PokerBenchResult:
        """Evaluate decision function against PokerBench scenarios.

        Args:
            scenarios: List of dicts with "instruction" and "output" keys.
            decision_fn: Function that takes parsed state and returns action string.
            max_scenarios: Optional limit on number of scenarios to evaluate.

        Returns:
            PokerBenchResult with accuracy metrics.
        """
        result = PokerBenchResult()

        street_correct: Counter[str] = Counter()
        street_total: Counter[str] = Counter()
        action_correct: Counter[str] = Counter()
        action_total: Counter[str] = Counter()
        confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        aggression_matches = 0
        aggression_total = 0

        items = scenarios[:max_scenarios] if max_scenarios else scenarios

        for scenario in items:
            instruction = scenario.get("instruction", "")
            output = scenario.get("output", "")

            # Parse the scenario
            parsed = parse_pokerbench_instruction(instruction)
            if parsed is None:
                result.skipped += 1
                continue

            # Get solver's action
            solver_action, _ = _parse_action_output(output)

            # Get agent's decision
            try:
                agent_action = decision_fn(parsed)
            except Exception:
                agent_action = "check"  # fallback

            # Normalize
            solver_action = solver_action.lower().strip()
            agent_action = agent_action.lower().strip()

            result.total += 1
            street = parsed.get("street", "unknown")

            # Exact match
            matched = solver_action == agent_action
            if matched:
                result.correct += 1
                street_correct[street] += 1
                action_correct[solver_action] += 1

            street_total[street] += 1
            action_total[solver_action] += 1
            confusion[solver_action][agent_action] += 1

            # Aggression alignment
            solver_agg = solver_action in _AGGRESSIVE
            agent_agg = agent_action in _AGGRESSIVE
            aggression_total += 1
            if solver_agg == agent_agg:
                aggression_matches += 1

        # Compute final metrics
        if result.total > 0:
            result.accuracy = result.correct / result.total

        result.street_accuracy = {
            s: street_correct[s] / street_total[s]
            for s in street_total
            if street_total[s] > 0
        }
        result.street_counts = dict(street_total)

        result.action_accuracy = {
            a: action_correct[a] / action_total[a]
            for a in action_total
            if action_total[a] > 0
        }
        result.action_counts = dict(action_total)

        result.confusion = {k: dict(v) for k, v in confusion.items()}

        if aggression_total > 0:
            result.aggression_alignment = aggression_matches / aggression_total

        return result

    def evaluate_with_equity(
        self,
        scenarios: list[dict[str, str]],
        decision_fn: DecisionFn,
        *,
        equity_samples: int = 200,
        equity_seed: int = 42,
        max_scenarios: int | None = None,
    ) -> PokerBenchResult:
        """Like evaluate(), but enriches parsed state with Monte Carlo equity.

        Adds ``estimated_equity`` to each parsed state before passing to the
        decision function.
        """
        from packages.dataset_builder.pokerbench_adapter import PokerBenchAdapter

        adapter = PokerBenchAdapter(
            equity_samples=equity_samples, equity_seed=equity_seed
        )

        def enriched_fn(state: dict[str, Any]) -> str:
            if "estimated_equity" not in state or state.get("estimated_equity", 0) == 0:
                state["estimated_equity"] = adapter._estimate_equity(
                    state["hole_cards"], state["board"]
                )
            return decision_fn(state)

        return self.evaluate(
            scenarios, enriched_fn, max_scenarios=max_scenarios
        )


# ---------------------------------------------------------------------------
# Built-in decision functions for baseline comparisons
# ---------------------------------------------------------------------------


def random_decision(state: dict[str, Any], seed: int = 42) -> str:
    """Random baseline: pick uniformly from legal actions."""
    import random

    rng = random.Random(seed + hash(str(state.get("hole_cards", []))))
    return rng.choice(state.get("legal_actions", ["check"]))


def equity_threshold_decision(state: dict[str, Any]) -> str:
    """Simple equity-based decision for baseline comparison.

    - equity > 0.7 → raise/bet
    - equity > 0.4 and facing bet → call
    - facing no bet → check
    - else → fold
    """
    equity = state.get("estimated_equity", 0.5)
    to_call = state.get("to_call", 0)

    if equity > 0.7:
        return "raise" if to_call > 0 else "bet"
    if to_call == 0:
        return "check"
    if equity > 0.4:
        return "call"
    return "fold"


def tight_aggressive_decision(state: dict[str, Any]) -> str:
    """TAG-style decision for baseline.

    Preflop: raise strong hands, call medium, fold weak.
    Postflop: equity-driven with pot odds consideration.
    """
    equity = state.get("estimated_equity", 0.5)
    to_call = state.get("to_call", 0)
    pot = state.get("pot", 1)
    pot_odds = state.get("pot_odds", 0)
    street = state.get("street", "pre_flop")

    if street == "pre_flop":
        if equity > 0.6:
            return "raise"
        if equity > 0.45:
            return "call" if to_call > 0 else "check"
        return "fold" if to_call > 0 else "check"

    # Postflop
    if equity > 0.75:
        return "raise" if to_call > 0 else "bet"
    if equity > 0.55:
        return "bet" if to_call == 0 else "call"
    if to_call == 0:
        return "check"
    if pot_odds > 0 and equity > pot_odds:
        return "call"
    return "fold"
