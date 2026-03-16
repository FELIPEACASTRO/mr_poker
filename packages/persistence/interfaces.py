from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class HandRepository(ABC):
    """Abstract interface for hand persistence operations."""

    @abstractmethod
    def create_hand(
        self,
        *,
        hand_id: str,
        button_seat: int,
        stacks: tuple[int, int],
        seed: int | None,
        deck_prefix: list[str],
        initial_snapshot: dict[str, Any],
        session_id: str | None = None,
    ) -> None: ...

    @abstractmethod
    def append_action(
        self, *, hand_id: str, actor_seat: int, action_type: str, amount: int
    ) -> None: ...

    @abstractmethod
    def append_snapshot(
        self, *, hand_id: str, snapshot: dict[str, Any], label: str, snapshot_order: int | None = None
    ) -> None: ...

    @abstractmethod
    def append_decision_trace(
        self, *, session_id: str | None, hand_id: str, actor_seat: int, trace: dict[str, Any]
    ) -> None: ...

    @abstractmethod
    def get_hand(self, hand_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def get_actions(self, hand_id: str) -> list[dict[str, Any]]: ...

    @abstractmethod
    def get_snapshots(self, hand_id: str) -> list[dict[str, Any]]: ...

    @abstractmethod
    def get_latest_snapshot(self, hand_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def get_decision_traces(
        self, *, hand_id: str | None = None, session_id: str | None = None
    ) -> list[dict[str, Any]]: ...


class SessionRepository(ABC):
    """Abstract interface for session persistence operations."""

    @abstractmethod
    def create_session(
        self, *, session_id: str, session_type: str, config: dict[str, Any]
    ) -> None: ...

    @abstractmethod
    def get_session(self, session_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def update_session_summary(
        self, *, session_id: str, summary: dict[str, Any]
    ) -> None: ...

    @abstractmethod
    def delete_session_data(self, *, session_id: str) -> None: ...

    @abstractmethod
    def get_hands_for_session(self, session_id: str) -> list[dict[str, Any]]: ...
