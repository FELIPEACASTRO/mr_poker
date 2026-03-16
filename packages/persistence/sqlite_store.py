from __future__ import annotations

import json
import sqlite3
from typing import Any

from packages.persistence.database import DatabaseManager

SCHEMA_VERSION = 2

MIGRATIONS: dict[int, list[str]] = {
    1: [
        # --- core tables ---
        """CREATE TABLE IF NOT EXISTS hands (
            hand_id TEXT PRIMARY KEY,
            session_id TEXT,
            button_seat INTEGER NOT NULL,
            stacks_json TEXT NOT NULL,
            seed INTEGER,
            deck_prefix_json TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            session_type TEXT NOT NULL,
            config_json TEXT NOT NULL,
            summary_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS action_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hand_id TEXT NOT NULL,
            actor_seat INTEGER NOT NULL,
            action_type TEXT NOT NULL,
            amount INTEGER NOT NULL,
            action_order INTEGER NOT NULL,
            FOREIGN KEY(hand_id) REFERENCES hands(hand_id) ON DELETE CASCADE
        )""",
        """CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hand_id TEXT NOT NULL,
            snapshot_order INTEGER NOT NULL,
            snapshot_json TEXT NOT NULL,
            label TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(hand_id) REFERENCES hands(hand_id) ON DELETE CASCADE
        )""",
        """CREATE TABLE IF NOT EXISTS decision_traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            hand_id TEXT NOT NULL,
            actor_seat INTEGER NOT NULL,
            trace_json TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(hand_id) REFERENCES hands(hand_id) ON DELETE CASCADE
        )""",
        # --- indexes ---
        "CREATE INDEX IF NOT EXISTS idx_action_log_hand_order ON action_log(hand_id, action_order)",
        "CREATE INDEX IF NOT EXISTS idx_snapshots_hand_order ON snapshots(hand_id, snapshot_order)",
        "CREATE INDEX IF NOT EXISTS idx_decision_traces_hand ON decision_traces(hand_id)",
        "CREATE INDEX IF NOT EXISTS idx_decision_traces_session ON decision_traces(session_id)",
        "CREATE INDEX IF NOT EXISTS idx_hands_session ON hands(session_id)",
        "CREATE INDEX IF NOT EXISTS idx_hands_created ON hands(created_at)",
        "CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at)",
        # --- schema version tracking ---
        """CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER NOT NULL,
            applied_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""",
    ],
    2: [
        # Migration v2: nothing extra yet — reserved for future use
    ],
}


class SqliteUnitOfWork:
    """Transactional write unit for batched hand persistence."""

    def __init__(self, store: SqliteHandStore) -> None:
        self._store = store
        self._conn: sqlite3.Connection | None = None

    def __enter__(self) -> SqliteUnitOfWork:
        self._conn = self._store._db.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        conn = self._conn
        if conn is None:
            return
        try:
            if exc_type is None:
                conn.commit()
            else:
                conn.rollback()
        finally:
            conn.execute("PRAGMA optimize")
            conn.close()
            self._conn = None

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("SqliteUnitOfWork must be used inside a context block")
        return self._conn

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
    ) -> None:
        conn = self._require_conn()
        self._store._create_hand_on_conn(
            conn,
            hand_id=hand_id,
            button_seat=button_seat,
            stacks=stacks,
            seed=seed,
            deck_prefix=deck_prefix,
            session_id=session_id,
        )
        self._store._append_snapshot_on_conn(
            conn,
            hand_id=hand_id,
            snapshot_order=0,
            snapshot=initial_snapshot,
            label="initial",
        )

    def append_hand_batch(
        self,
        *,
        hand_id: str,
        session_id: str | None,
        actions: list[tuple[int, str, int, int]],
        snapshots: list[tuple[int, dict[str, Any], str]],
        decision_traces: list[tuple[int, dict[str, Any]]],
    ) -> None:
        conn = self._require_conn()
        self._store._append_hand_batch_on_conn(
            conn,
            hand_id=hand_id,
            session_id=session_id,
            actions=actions,
            snapshots=snapshots,
            decision_traces=decision_traces,
        )


class SqliteHandStore:
    def __init__(
        self,
        db_path_or_manager: str | DatabaseManager | None = None,
        *,
        db: DatabaseManager | None = None,
    ) -> None:
        if db is not None:
            self._db = db
        elif isinstance(db_path_or_manager, DatabaseManager):
            self._db = db_path_or_manager
        else:
            self._db = DatabaseManager(db_path_or_manager or "var/poker_ai_local.db")
        self.db_path = self._db.db_path
        self._run_migrations()

    def _connect(self) -> sqlite3.Connection:
        return self._db.connect()

    def _current_version(self, conn: sqlite3.Connection) -> int:
        try:
            row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
            return int(row["v"]) if row and row["v"] is not None else 0
        except sqlite3.OperationalError:
            return 0

    def _run_migrations(self) -> None:
        with self._connect() as conn:
            # Backward compat: ensure session_id column exists before migrations
            # that create indexes on it (handles pre-migration legacy databases)
            try:
                existing = {
                    row["name"]
                    for row in conn.execute("PRAGMA table_info(hands)").fetchall()
                }
                if existing and "session_id" not in existing:
                    conn.execute("ALTER TABLE hands ADD COLUMN session_id TEXT")
            except sqlite3.OperationalError:
                pass  # hands table doesn't exist yet, migrations will create it

            current = self._current_version(conn)
            for version in sorted(MIGRATIONS.keys()):
                if version <= current:
                    continue
                for stmt in MIGRATIONS[version]:
                    conn.execute(stmt)
                conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)", (version,)
                )
            conn.execute("ANALYZE")

    def uow(self) -> SqliteUnitOfWork:
        return SqliteUnitOfWork(self)

    _ALLOWED_TABLES = frozenset({"hands", "sessions", "action_log", "snapshots", "decision_traces"})
    _ALLOWED_COLUMN_TYPES = frozenset({"TEXT", "INTEGER", "REAL", "BLOB", "NUMERIC"})

    def _ensure_column(
        self, conn: sqlite3.Connection, table: str, column: str, column_type: str
    ) -> None:
        if table not in self._ALLOWED_TABLES:
            raise ValueError(f"Table {table!r} not in allowlist")
        if column_type.upper() not in self._ALLOWED_COLUMN_TYPES:
            raise ValueError(f"Column type {column_type!r} not in allowlist")
        if not column.isidentifier():
            raise ValueError(f"Column name {column!r} is not a valid identifier")
        existing = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    # ── writes ──────────────────────────────────────────────────────

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
    ) -> None:
        with self._connect() as conn:
            self._create_hand_on_conn(
                conn,
                hand_id=hand_id,
                button_seat=button_seat,
                stacks=stacks,
                seed=seed,
                deck_prefix=deck_prefix,
                session_id=session_id,
            )
            self._append_snapshot_on_conn(
                conn,
                hand_id=hand_id,
                snapshot_order=0,
                snapshot=initial_snapshot,
                label="initial",
            )

    def _create_hand_on_conn(
        self,
        conn: sqlite3.Connection,
        *,
        hand_id: str,
        button_seat: int,
        stacks: tuple[int, int],
        seed: int | None,
        deck_prefix: list[str],
        session_id: str | None,
    ) -> None:
        conn.execute(
            "INSERT INTO hands (hand_id, session_id, button_seat, stacks_json, seed, deck_prefix_json) VALUES (?, ?, ?, ?, ?, ?)",
            (
                hand_id,
                session_id,
                button_seat,
                json.dumps(list(stacks)),
                seed,
                json.dumps(deck_prefix),
            ),
        )

    def append_action(
        self, *, hand_id: str, actor_seat: int, action_type: str, amount: int
    ) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(action_order), -1) AS max_order FROM action_log WHERE hand_id = ?",
                (hand_id,),
            ).fetchone()
            action_order = int(row["max_order"]) + 1
            conn.execute(
                "INSERT INTO action_log (hand_id, actor_seat, action_type, amount, action_order) VALUES (?, ?, ?, ?, ?)",
                (hand_id, actor_seat, action_type, amount, action_order),
            )

    def append_action_with_order(
        self,
        *,
        hand_id: str,
        actor_seat: int,
        action_type: str,
        amount: int,
        action_order: int,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO action_log (hand_id, actor_seat, action_type, amount, action_order) VALUES (?, ?, ?, ?, ?)",
                (hand_id, actor_seat, action_type, amount, action_order),
            )

    def append_snapshot(
        self,
        *,
        hand_id: str,
        snapshot_order: int | None = None,
        snapshot: dict[str, Any],
        label: str,
    ) -> None:
        with self._connect() as conn:
            self._append_snapshot_on_conn(
                conn,
                hand_id=hand_id,
                snapshot_order=snapshot_order,
                snapshot=snapshot,
                label=label,
            )

    def _append_snapshot_on_conn(
        self,
        conn: sqlite3.Connection,
        *,
        hand_id: str,
        snapshot_order: int | None,
        snapshot: dict[str, Any],
        label: str,
    ) -> None:
        if snapshot_order is None:
            row = conn.execute(
                "SELECT COALESCE(MAX(snapshot_order), -1) AS max_order FROM snapshots WHERE hand_id = ?",
                (hand_id,),
            ).fetchone()
            snapshot_order = int(row["max_order"]) + 1
        conn.execute(
            "INSERT INTO snapshots (hand_id, snapshot_order, snapshot_json, label) VALUES (?, ?, ?, ?)",
            (hand_id, snapshot_order, json.dumps(snapshot), label),
        )

    def append_hand_batch(
        self,
        *,
        hand_id: str,
        session_id: str | None,
        actions: list[tuple[int, str, int, int]],
        snapshots: list[tuple[int, dict[str, Any], str]],
        decision_traces: list[tuple[int, dict[str, Any]]],
    ) -> None:
        with self._connect() as conn:
            self._append_hand_batch_on_conn(
                conn,
                hand_id=hand_id,
                session_id=session_id,
                actions=actions,
                snapshots=snapshots,
                decision_traces=decision_traces,
            )

    def _append_hand_batch_on_conn(
        self,
        conn: sqlite3.Connection,
        *,
        hand_id: str,
        session_id: str | None,
        actions: list[tuple[int, str, int, int]],
        snapshots: list[tuple[int, dict[str, Any], str]],
        decision_traces: list[tuple[int, dict[str, Any]]],
    ) -> None:
        if actions:
            conn.executemany(
                "INSERT INTO action_log (hand_id, actor_seat, action_type, amount, action_order) VALUES (?, ?, ?, ?, ?)",
                [
                    (hand_id, actor_seat, action_type, amount, action_order)
                    for actor_seat, action_type, amount, action_order in actions
                ],
            )
        if snapshots:
            conn.executemany(
                "INSERT INTO snapshots (hand_id, snapshot_order, snapshot_json, label) VALUES (?, ?, ?, ?)",
                [
                    (hand_id, snapshot_order, json.dumps(snapshot), label)
                    for snapshot_order, snapshot, label in snapshots
                ],
            )
        if decision_traces:
            conn.executemany(
                "INSERT INTO decision_traces (session_id, hand_id, actor_seat, trace_json) VALUES (?, ?, ?, ?)",
                [
                    (session_id, hand_id, actor_seat, json.dumps(trace))
                    for actor_seat, trace in decision_traces
                ],
            )

    def append_decision_trace(
        self,
        *,
        session_id: str | None,
        hand_id: str,
        actor_seat: int,
        trace: dict[str, Any],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO decision_traces (session_id, hand_id, actor_seat, trace_json) VALUES (?, ?, ?, ?)",
                (session_id, hand_id, actor_seat, json.dumps(trace)),
            )

    def create_session(
        self, *, session_id: str, session_type: str, config: dict[str, Any]
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, session_type, config_json, summary_json, updated_at) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
                (session_id, session_type, json.dumps(config), None),
            )

    def delete_session_data(self, *, session_id: str) -> None:
        """Delete all data for a session using efficient subqueries."""
        with self._connect() as conn:
            sub = "SELECT hand_id FROM hands WHERE session_id = ?"
            conn.execute(f"DELETE FROM action_log WHERE hand_id IN ({sub})", (session_id,))
            conn.execute(f"DELETE FROM snapshots WHERE hand_id IN ({sub})", (session_id,))
            conn.execute(f"DELETE FROM decision_traces WHERE hand_id IN ({sub})", (session_id,))
            conn.execute("DELETE FROM decision_traces WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM hands WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    def update_session_summary(
        self, *, session_id: str, summary: dict[str, Any]
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET summary_json = ?, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                (json.dumps(summary), session_id),
            )

    # ── reads ───────────────────────────────────────────────────────

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return None
            item = dict(row)
            item["config"] = json.loads(item.pop("config_json"))
            item["summary"] = (
                json.loads(item.pop("summary_json"))
                if item.get("summary_json")
                else None
            )
            item.pop("updated_at", None)
            return item

    def get_hand(self, hand_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM hands WHERE hand_id = ?", (hand_id,)
            ).fetchone()
            if row is None:
                return None
            return {
                "hand_id": row["hand_id"],
                "session_id": row["session_id"],
                "button_seat": row["button_seat"],
                "stacks": tuple(json.loads(row["stacks_json"])),
                "seed": row["seed"],
                "deck_prefix": json.loads(row["deck_prefix_json"]),
            }

    def get_hands_for_session(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM hands WHERE session_id = ? ORDER BY created_at, hand_id",
                (session_id,),
            ).fetchall()
            result = []
            for row in rows:
                result.append(
                    {
                        "hand_id": row["hand_id"],
                        "session_id": row["session_id"],
                        "button_seat": row["button_seat"],
                        "stacks": tuple(json.loads(row["stacks_json"])),
                        "seed": row["seed"],
                        "deck_prefix": json.loads(row["deck_prefix_json"]),
                    }
                )
            return result

    def get_actions(self, hand_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM action_log WHERE hand_id = ? ORDER BY action_order",
                (hand_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_snapshots(self, hand_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM snapshots WHERE hand_id = ? ORDER BY snapshot_order",
                (hand_id,),
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["snapshot"] = json.loads(item.pop("snapshot_json"))
                result.append(item)
            return result

    def get_latest_snapshot(self, hand_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM snapshots WHERE hand_id = ? ORDER BY snapshot_order DESC LIMIT 1",
                (hand_id,),
            ).fetchone()
            if row is None:
                return None
            item = dict(row)
            item["snapshot"] = json.loads(item.pop("snapshot_json"))
            return item

    def get_decision_traces(
        self, *, hand_id: str | None = None, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM decision_traces"
        params: list[Any] = []
        clauses: list[str] = []
        if hand_id is not None:
            clauses.append("hand_id = ?")
            params.append(hand_id)
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["trace"] = json.loads(item.pop("trace_json"))
                result.append(item)
            return result

    # ── diagnostics ─────────────────────────────────────────────────

    def db_stats(self) -> dict[str, Any]:
        """Return database size and row counts for monitoring."""
        import os
        stats: dict[str, Any] = {"db_path": self.db_path}
        try:
            stats["size_bytes"] = os.path.getsize(self.db_path)
        except OSError:
            stats["size_bytes"] = 0
        with self._connect() as conn:
            for table in ["hands", "sessions", "action_log", "snapshots", "decision_traces"]:
                row = conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()
                stats[f"{table}_count"] = row["c"] if row else 0
            ver = conn.execute(
                "SELECT MAX(version) AS v FROM schema_version"
            ).fetchone()
            stats["schema_version"] = ver["v"] if ver and ver["v"] else 0
        return stats
