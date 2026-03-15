from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class SagaStep(Generic[T]):
    """One synchronous in-process saga step."""

    name: str
    action: Callable[[], T]
    compensate: Callable[[], None] | None = None


class InProcessSaga:
    """Synchronous saga executor with reverse-order compensations."""

    def run(self, steps: list[SagaStep[Any]]) -> list[Any]:
        results: list[Any] = []
        completed: list[SagaStep[Any]] = []
        try:
            for step in steps:
                results.append(step.action())
                completed.append(step)
            return results
        except Exception:
            for step in reversed(completed):
                if step.compensate is None:
                    continue
                try:
                    step.compensate()
                except Exception:
                    # Compensations are best-effort; preserve original failure.
                    continue
            raise
