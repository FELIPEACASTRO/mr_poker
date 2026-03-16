"""Centralized database connection manager for SQLite.

All stores should use DatabaseManager.connect() to obtain connections,
ensuring consistent PRAGMA configuration (WAL, foreign keys, etc.).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


class DatabaseManager:
    """Single source of truth for SQLite connections and configuration."""

    def __init__(self, db_path: str = "var/poker_ai_local.db") -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._configure_db()

    def _configure_db(self) -> None:
        """Set database-level PRAGMAs (WAL persists across connections)."""
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")

    def connect(self) -> sqlite3.Connection:
        """Return a new connection with all per-connection PRAGMAs set."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn
