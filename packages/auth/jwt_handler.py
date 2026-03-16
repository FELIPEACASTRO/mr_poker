from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import base64
from typing import Any

from packages.auth.models import TokenPayload

_DEFAULT_SECRET = "dev-secret-change-me"
_SECRET = os.getenv("POKER_JWT_SECRET", _DEFAULT_SECRET)
_ALGORITHM = "HS256"

if _SECRET == _DEFAULT_SECRET:
    import logging as _logging

    _logging.getLogger(__name__).warning(
        "POKER_JWT_SECRET not set — using insecure default. "
        "Set POKER_JWT_SECRET env var for production."
    )


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)


def create_access_token(
    subject: str,
    permissions: list[str] | None = None,
    expires_minutes: int = 60,
) -> str:
    """Create a JWT access token."""
    header = {"alg": _ALGORITHM, "typ": "JWT"}
    payload = {
        "sub": subject,
        "permissions": permissions or [],
        "exp": int(time.time()) + expires_minutes * 60,
        "iat": int(time.time()),
    }
    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
    message = f"{header_b64}.{payload_b64}"
    signature = hmac.new(
        _SECRET.encode(), message.encode(), hashlib.sha256
    ).digest()
    sig_b64 = _b64url_encode(signature)
    return f"{message}.{sig_b64}"


def decode_token(token: str) -> TokenPayload:
    """Decode and verify a JWT token."""
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("invalid token format")

    header_b64, payload_b64, sig_b64 = parts
    message = f"{header_b64}.{payload_b64}"
    expected_sig = hmac.new(
        _SECRET.encode(), message.encode(), hashlib.sha256
    ).digest()
    actual_sig = _b64url_decode(sig_b64)

    if not hmac.compare_digest(expected_sig, actual_sig):
        raise ValueError("invalid token signature")

    payload_bytes = _b64url_decode(payload_b64)
    payload: dict[str, Any] = json.loads(payload_bytes)

    if payload.get("exp", 0) < time.time():
        raise ValueError("token expired")

    return TokenPayload(
        sub=payload["sub"],
        exp=payload["exp"],
        permissions=payload.get("permissions", []),
    )
