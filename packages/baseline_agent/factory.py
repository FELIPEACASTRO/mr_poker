from __future__ import annotations

from packages.baseline_agent.strategy import (
    DecisionPolicyStrategy,
    EquityEstimatorStrategy,
    MonteCarloEquityEstimator,
    RuleBasedDecisionStrategy,
)


class BaselineAgentFactory:
    @staticmethod
    def create_equity_estimator() -> EquityEstimatorStrategy:
        return MonteCarloEquityEstimator()

    @staticmethod
    def create_decision_strategy() -> DecisionPolicyStrategy:
        return RuleBasedDecisionStrategy()
