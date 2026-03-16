"""Engine-specific exceptions for clear error handling."""

from __future__ import annotations


class EngineError(Exception):
    """Base exception for engine errors."""


class InvalidActionError(EngineError, ValueError):
    """Raised when an invalid action is attempted.

    Also inherits from ValueError for backward compatibility with callers
    that catch ValueError from engine operations.
    """


class EngineInvariantError(EngineError):
    """Raised when an engine invariant is violated."""


class DeckExhaustedError(EngineError):
    """Raised when the deck runs out of cards."""
