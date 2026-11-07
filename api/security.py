"""Short-lived HMAC-signed media URLs (image tags cannot send auth headers)."""

from __future__ import annotations

import hashlib
import hmac
import time
from pathlib import Path
from urllib.parse import quote

from api.config import Settings


def _signature(secret: str, resource: str, expires: int) -> str:
    message = f"{resource}|{expires}".encode()
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()[:32]


def sign_path(settings: Settings, path: str, now: float | None = None) -> str:
    expires = int((now or time.time()) + settings.media_url_ttl_seconds)
    sig = _signature(settings.media_signing_secret, path, expires)
    return f"{quote(path)}?exp={expires}&sig={sig}"


def verify(settings: Settings, path: str, expires: int | None, sig: str | None, now: float | None = None) -> bool:
    if expires is None or not sig:
        return False
    if expires < int(now or time.time()):
        return False
    expected = _signature(settings.media_signing_secret, path, expires)
    return hmac.compare_digest(expected, sig)


def safe_data_path(settings: Settings, candidate: str | Path) -> Path | None:
    """Resolve a stored path and refuse anything outside the data directory."""
    try:
        path = Path(candidate)
        if not path.is_absolute():
            path = settings.data_dir.parent / path
        resolved = path.resolve()
        resolved.relative_to(settings.data_dir.resolve())
    except (OSError, ValueError):
        return None
    return resolved if resolved.is_file() else None
