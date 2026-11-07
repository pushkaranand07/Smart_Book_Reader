"""Shared rate limiter (slowapi, in-memory per process)."""

from slowapi import Limiter

from api.auth import rate_limit_key

limiter = Limiter(key_func=rate_limit_key)
