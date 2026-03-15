from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class SqliteHandStore:
    def __init__(self, db_path: str = 'var/poker_ai_local.db') -> None:
        self.db_path = db_path
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA synchronous=NORMAL')
            conn.execute('PRAGMA foreign_keys=ON')
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS hands (
                    hand_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    button_seat INTEGER NOT NULL,
                    stacks_json TEXT NOT NULL,
                    seed INTEGER,
                    deck_prefix_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS action_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hand_id TEXT NOT NULL,
                    actor_seat INTEGER NOT NULL,
                    action_type TEXT NOT NULL,
                    amount INTEGER NOT NULL,
                    action_order INTEGER NOT NULL,
                    FOREIGN KEY(hand_id) REFERENCES hands(hand_id)
                );
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hand_id TEXT NOT NULL,
                    snapshot_order INTEGER NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    label TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(hand_id) REFERENCES hands(hand_id)
                );
                CREATE TABLE IF NOT EXISTS decision_traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    hand_id TEXT NOT NULL,
                    actor_seat INTEGER NOT NULL,
                    trace_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    session_type TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    summary_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_action_log_hand_order ON action_log(hand_id, action_order);
                CREATE INDEX IF NOT EXISTS idx_snapshots_hand_order ON snapshots(hand_id, snapshot_order);
                CREATE INDEX IF NOT EXISTS idx_decision_traces_hand ON decision_traces(hand_id);
                CREATE INDEX IF NOT EXISTS idx_decision_traces_session ON decision_traces(session_id);
                CREATE INDEX IF NOT EXISTS idx_hands_session ON hands(session_id);
                """
            )
            self._ensure_column(conn, 'hands', 'session_id', 'TEXT')

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
        existing = {row['name'] for row in conn.execute(f'PRAGMA table_info({table})').fetchall()}
        if column not in existing:
            conn.execute(f'ALTER TABLE {table} ADD COLUMN {column} {column_type}')

    def create_hand(
        self,
        *,
        hand_id: str,
        button_seat: int,
        stacks: tuple[int, int],
        seed: int | None,
        deck_prefix: list[str],
        initial_snapshot: dict,
        session_id: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                'INSERT OR REPLACE INTO hands (hand_id, session_id, button_seat, stacks_json, seed, deck_prefix_json) VALUES (?, ?, ?, ?, ?, ?)',
                (hand_id, session_id, button_seat, json.dumps(list(stacks)), seed, json.dumps(deck_prefix)),
            )
        self.append_snapshot(hand_id=hand_id, snapshot_order=0, snapshot=initial_snapshot, label='initial')

    def append_action(self, *, hand_id: str, actor_seat: int, action_type: str, amount: int) -> None:
        with self._connect() as conn:
            row = conn.execute('SELECT COALESCE(MAX(action_order), -1) AS max_order FROM action_log WHERE hand_id = ?', (hand_id,)).fetchone()
            action_order = int(row['max_order']) + 1
            conn.execute(
                'INSERT INTO action_log (hand_id, actor_seat, action_type, amount, action_order) VALUES (?, ?, ?, ?, ?)',
                (hand_id, actor_seat, action_type, amount, action_order),
            )

    def append_snapshot(self, *, hand_id: str, snapshot_order: int | None = None, snapshot: dict, label: str) -> None:
        with self._connect() as conn:
            if snapshot_order is None:
                row = conn.execute('SELECT COALESCE(MAX(snapshot_order), -1) AS max_order FROM snapshots WHERE hand_id = ?', (hand_id,)).fetchone()
                snapshot_order = int(row['max_order']) + 1
            conn.execute(
                'INSERT INTO snapshots (hand_id, snapshot_order, snapshot_json, label) VALUES (?, ?, ?, ?)',
                (hand_id, snapshot_order, json.dumps(snapshot), label),
            )

    def append_decision_trace(self, *, session_id: str | None, hand_id: str, actor_seat: int, trace: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                'INSERT INTO decision_traces (session_id, hand_id, actor_seat, trace_json) VALUES (?, ?, ?, ?)',
                (session_id, hand_id, actor_seat, json.dumps(trace)),
            )

    def get_decision_traces(self, *, hand_id: str | None = None, session_id: str | None = None) -> list[dict[str, Any]]:
        query = 'SELECT * FROM decision_traces'
        params: list[Any] = []
        clauses: list[str] = []
        if hand_id is not None:
            clauses.append('hand_id = ?')
            params.append(hand_id)
        if session_id is not None:
            clauses.append('session_id = ?')
            params.append(session_id)
        if clauses:
            query += ' WHERE ' + ' AND '.join(clauses)
        query += ' ORDER BY id'
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item['trace'] = json.loads(item.pop('trace_json'))
                result.append(item)
            return result

    def create_session(self, *, session_id: str, session_type: str, config: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                'INSERT OR REPLACE INTO sessions (session_id, session_type, config_json, summary_json, updated_at) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)',
                (session_id, session_type, json.dumps(config), None),
            )

    def update_session_summary(self, *, session_id: str, summary: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                'UPDATE sessions SET summary_json = ?, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?',
                (json.dumps(summary), session_id),
            )

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute('SELECT * FROM sessions WHERE session_id = ?', (session_id,)).fetchone()
            if row is None:
                return None
            item = dict(row)
            item['config'] = json.loads(item.pop('config_json'))
            item['summary'] = json.loads(item.pop('summary_json')) if item.get('summary_json') else None
            item.pop('updated_at', None)
            return item

    def get_hand(self, hand_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute('SELECT * FROM hands WHERE hand_id = ?', (hand_id,)).fetchone()
            if row is None:
                return None
            return {
                'hand_id': row['hand_id'],
                'session_id': row['session_id'],
                'button_seat': row['button_seat'],
                'stacks': tuple(json.loads(row['stacks_json'])),
                'seed': row['seed'],
                'deck_prefix': json.loads(row['deck_prefix_json']),
            }

    def get_hands_for_session(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute('SELECT * FROM hands WHERE session_id = ? ORDER BY created_at, hand_id', (session_id,)).fetchall()
            result = []
            for row in rows:
                result.append({
                    'hand_id': row['hand_id'],
                    'session_id': row['session_id'],
                    'button_seat': row['button_seat'],
                    'stacks': tuple(json.loads(row['stacks_json'])),
                    'seed': row['seed'],
                    'deck_prefix': json.loads(row['deck_prefix_json']),
                })
            return result

    def get_actions(self, hand_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute('SELECT * FROM action_log WHERE hand_id = ? ORDER BY action_order', (hand_id,)).fetchall()
            return [dict(row) for row in rows]

    def get_snapshots(self, hand_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute('SELECT * FROM snapshots WHERE hand_id = ? ORDER BY snapshot_order', (hand_id,)).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item['snapshot'] = json.loads(item.pop('snapshot_json'))
                result.append(item)
            return result

    def get_latest_snapshot(self, hand_id: str) -> dict[str, Any] | None:
        snapshots = self.get_snapshots(hand_id)
        return snapshots[-1] if snapshots else None
