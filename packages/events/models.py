from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class DomainEvent:
    """Base domain event."""

    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    event_type: str = ""
    aggregate_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HandStarted(DomainEvent):
    event_type: str = "hand.started"


@dataclass(frozen=True)
class ActionApplied(DomainEvent):
    event_type: str = "action.applied"


@dataclass(frozen=True)
class HandCompleted(DomainEvent):
    event_type: str = "hand.completed"
