from __future__ import annotations

import enum

from pydantic import BaseModel


class Permission(str, enum.Enum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class User(BaseModel):
    username: str
    permissions: list[Permission] = []


class TokenPayload(BaseModel):
    sub: str
    exp: int
    permissions: list[str] = []
