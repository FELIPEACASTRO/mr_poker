from __future__ import annotations

import os
from typing import Callable

from fastapi import Depends, HTTPException, Request

from packages.auth.jwt_handler import decode_token
from packages.auth.models import Permission, User

_DEFAULT_USER = User(
    username="anonymous",
    permissions=[Permission.READ, Permission.WRITE, Permission.ADMIN],
)

_API_KEYS: dict[str, User] = {}


def _auth_enabled() -> bool:
    return os.getenv("POKER_AUTH_ENABLED", "false").lower() == "true"


def get_current_user(request: Request) -> User:
    """FastAPI dependency to extract the current user.

    When POKER_AUTH_ENABLED=false (default), returns a default admin user.
    When enabled, checks Authorization header (Bearer JWT) or X-API-Key.
    """
    if not _auth_enabled():
        return _DEFAULT_USER

    # Try API key first
    api_key = request.headers.get("X-API-Key")
    if api_key and api_key in _API_KEYS:
        return _API_KEYS[api_key]

    # Try Bearer token
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="missing or invalid authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = auth_header[7:]
    try:
        payload = decode_token(token)
    except ValueError as exc:
        raise HTTPException(
            status_code=401,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    permissions = []
    for p in payload.permissions:
        try:
            permissions.append(Permission(p))
        except ValueError:
            continue

    return User(username=payload.sub, permissions=permissions)


def require_permission(permission: Permission) -> Callable[..., User]:
    """Factory that returns a FastAPI dependency enforcing a specific permission."""

    def _check(user: User = Depends(get_current_user)) -> User:
        if permission not in user.permissions:
            raise HTTPException(
                status_code=403,
                detail=f"permission '{permission.value}' required",
            )
        return user

    return _check


require_write = require_permission(Permission.WRITE)
require_admin = require_permission(Permission.ADMIN)
