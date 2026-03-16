from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar
from uuid import uuid4

T = TypeVar("T")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SagaStep(Generic[T]):
    """One synchronous in-process saga step."""

    name: str
    action: Callable[[], T]
    compensate: Callable[[], None] | None = None
    timeout_seconds: float = 60.0
    max_retries: int = 0


class InProcessSaga:
    """Synchronous saga executor with reverse-order compensations and optional persistence."""

    def __init__(self, store: Any | None = None) -> None:
        self.store = store  # Optional SagaStore for persistence

    def run(
        self,
        steps: list[SagaStep[Any]],
        *,
        saga_type: str = "generic",
        idempotency_key: str | None = None,
    ) -> list[Any]:
        saga_id = str(uuid4())

        # Check idempotency
        if self.store and idempotency_key:
            existing = self.store.get_by_idempotency_key(idempotency_key)
            if existing and existing["status"] == "completed":
                logger.info("Saga already completed for key=%s", idempotency_key)
                return existing.get("results", [])

        # Persist initial state
        if self.store:
            self.store.create(
                saga_id, saga_type, len(steps), idempotency_key=idempotency_key
            )

        results: list[Any] = []
        completed: list[SagaStep[Any]] = []
        try:
            for idx, step in enumerate(steps):
                result = self._execute_step(step)
                results.append(result)
                completed.append(step)
                if self.store:
                    self.store.advance(saga_id, idx, result)
            if self.store:
                self.store.complete(saga_id)
            return results
        except Exception as exc:
            logger.error("Saga %s failed at step %d: %s", saga_id, len(completed), exc)
            if self.store:
                self.store.fail(saga_id, str(exc))
                self.store.compensating(saga_id)
            for step in reversed(completed):
                if step.compensate is None:
                    continue
                try:
                    step.compensate()
                except Exception as comp_exc:
                    logger.error(
                        "Compensation failed for step '%s': %s",
                        step.name,
                        comp_exc,
                        exc_info=True,
                    )
            raise

    def _execute_step(self, step: SagaStep[Any]) -> Any:
        last_exc: Exception | None = None
        for attempt in range(step.max_retries + 1):
            try:
                start = time.monotonic()
                result = step.action()
                elapsed = time.monotonic() - start
                if elapsed > step.timeout_seconds:
                    raise TimeoutError(
                        f"Step '{step.name}' took {elapsed:.1f}s (limit {step.timeout_seconds}s)"
                    )
                return result
            except Exception as exc:
                last_exc = exc
                if attempt < step.max_retries:
                    backoff = min(2**attempt * 0.1, 5.0)
                    logger.warning(
                        "Step '%s' attempt %d failed, retrying in %.1fs: %s",
                        step.name, attempt + 1, backoff, exc,
                    )
                    time.sleep(backoff)
        raise last_exc  # type: ignore[misc]
