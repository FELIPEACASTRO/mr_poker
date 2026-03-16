from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class PaginationParams:
    """Standard pagination parameters."""

    limit: int = 50
    offset: int = 0

    def __post_init__(self) -> None:
        self.limit = max(1, min(self.limit, 1000))
        self.offset = max(0, self.offset)


def paginate(
    items: list[Any],
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """Apply pagination to a list of items."""
    total = len(items)
    sliced = items[offset : offset + limit]
    return {
        "items": sliced,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": (offset + limit) < total,
    }
