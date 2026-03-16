# ADR-004: JWT Authentication

## Status
Accepted

## Context
The API needs authentication for write operations while remaining easy to use in development.

## Decision
- Custom JWT implementation using HMAC-SHA256 (no external JWT library dependency)
- Configurable via `POKER_AUTH_ENABLED` environment variable
- Disabled by default in development (`auth_enabled: false`)
- Supports both Bearer JWT and X-API-Key header
- Permission model: READ, WRITE, ADMIN

## Consequences
- **Positive**: Zero-dependency auth (stdlib only)
- **Positive**: Completely optional — disabled by default
- **Positive**: Simple permission model sufficient for local use
- **Negative**: No refresh tokens, no OAuth2 flows
- **Mitigation**: This is a local-first tool; production deployment would add proper auth proxy
