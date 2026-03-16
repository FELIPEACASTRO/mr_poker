# ADR-001: SQLite as Primary Database

## Status
Accepted

## Context
mr_poker is a local-first application. We need a database that:
- Works without external services
- Supports concurrent reads
- Has zero configuration
- Is portable across platforms

## Decision
Use SQLite as the primary and only database engine.

## Consequences
- **Positive**: Zero setup, single file, excellent read performance, WAL mode for concurrency
- **Positive**: Easy to backup (copy file), easy to inspect (sqlite3 CLI)
- **Negative**: Write concurrency limited (one writer at a time)
- **Negative**: No built-in replication or clustering
- **Mitigation**: Use Unit of Work pattern for batch writes, WAL mode for read concurrency
