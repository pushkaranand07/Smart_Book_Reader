"""Authentication: API keys (default) or JWT bearer tokens."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from functools import lru_cache

import jwt
from fastapi import Depends, Request

from api.config import Settings, get_settings
from api.errors import ApiError


@dataclass(frozen=True)
class Principal:
    owner_id: str


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@lru_cache(maxsize=8)
def _key_table(raw: str) -> dict[str, str]:
    """Map sha256(key) -> owner_id. Keys are never kept in plain text after parsing."""
    table: dict[str, str] = {}
    for entry in (e.strip() for e in raw.split(",")):
        if not entry:
            continue
        owner, sep, key = entry.partition(":")
        if not sep:
            owner, key = f"key-{_digest(entry)[:12]}", entry
        table[_digest(key.strip())] = owner.strip()
    return table


def _unauthorized(message: str = "A valid API key is required.") -> ApiError:
    return ApiError(401, "UNAUTHORIZED", message)


def principal_from_request(request: Request, settings: Settings) -> Principal | None:
    if settings.auth_mode == "none":
        return Principal(owner_id="local")
    if settings.auth_mode == "api_key":
        supplied = request.headers.get("x-api-key", "")
        if not supplied:
            return None
        supplied_digest = _digest(supplied)
        for digest, owner in _key_table(settings.api_keys).items():
            if hmac.compare_digest(digest, supplied_digest):
                return Principal(owner_id=owner)
        return None
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token or not settings.jwt_secret:
        return None
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            audience=settings.jwt_audience,
            options={"require": ["sub", "exp"], "verify_aud": settings.jwt_audience is not None},
        )
    except jwt.PyJWTError:
        return None
    return Principal(owner_id=str(claims["sub"]))


def require_principal(request: Request, settings: Settings = Depends(get_settings)) -> Principal:
    principal = principal_from_request(request, settings)
    if principal is None:
        raise _unauthorized()
    request.state.owner_id = principal.owner_id
    return principal


def rate_limit_key(request: Request) -> str:
    """Rate-limit per credential when present, else per client address."""
    credential = request.headers.get("x-api-key") or request.headers.get("authorization")
    if credential:
        return "cred:" + _digest(credential)[:16]
    return "ip:" + (request.client.host if request.client else "unknown")
