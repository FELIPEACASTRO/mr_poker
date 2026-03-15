from __future__ import annotations

import pytest

from services.cqrs.saga import InProcessSaga, SagaStep


def test_saga_runs_all_steps_in_order() -> None:
    calls: list[str] = []
    saga = InProcessSaga()
    result = saga.run(
        [
            SagaStep(name="a", action=lambda: calls.append("a") or 1),
            SagaStep(name="b", action=lambda: calls.append("b") or 2),
        ]
    )
    assert calls == ["a", "b"]
    assert result == [1, 2]


def test_saga_compensates_completed_steps_in_reverse_order() -> None:
    calls: list[str] = []
    saga = InProcessSaga()
    with pytest.raises(RuntimeError):
        saga.run(
            [
                SagaStep(
                    name="a",
                    action=lambda: calls.append("run-a") or 1,
                    compensate=lambda: calls.append("undo-a"),
                ),
                SagaStep(
                    name="b",
                    action=lambda: calls.append("run-b") or 2,
                    compensate=lambda: calls.append("undo-b"),
                ),
                SagaStep(
                    name="c", action=lambda: (_ for _ in ()).throw(RuntimeError())
                ),
            ]
        )
    assert calls == ["run-a", "run-b", "undo-b", "undo-a"]


def test_saga_ignores_compensation_errors_and_preserves_original_failure() -> None:
    calls: list[str] = []
    saga = InProcessSaga()

    def broken_compensation() -> None:
        calls.append("undo-a")
        raise RuntimeError("compensation-failure")

    with pytest.raises(ValueError):
        saga.run(
            [
                SagaStep(
                    name="a",
                    action=lambda: calls.append("run-a") or 1,
                    compensate=broken_compensation,
                ),
                SagaStep(
                    name="b", action=lambda: (_ for _ in ()).throw(ValueError("boom"))
                ),
            ]
        )

    assert calls == ["run-a", "undo-a"]


def test_saga_skips_missing_compensation() -> None:
    calls: list[str] = []
    saga = InProcessSaga()

    with pytest.raises(RuntimeError):
        saga.run(
            [
                SagaStep(
                    name="a", action=lambda: calls.append("run-a") or 1, compensate=None
                ),
                SagaStep(
                    name="b", action=lambda: (_ for _ in ()).throw(RuntimeError("fail"))
                ),
            ]
        )

    assert calls == ["run-a"]
