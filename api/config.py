"""Runtime configuration (environment variables, optionally from a .env file)."""

from __future__ import annotations

import os
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.qa.azure_llm import AzureConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # APP_ENV_FILE lets tests (and alternative deployments) point away from the developer's local .env.
    model_config = SettingsConfigDict(
        env_file=os.environ.get("APP_ENV_FILE", ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    environment: Literal["development", "production", "test"] = "development"
    port: int = 8000
    data_dir: Path = PROJECT_ROOT / "data"
    database_url: str = ""

    # Auth: "api_key" (X-API-Key header), "jwt" (Authorization: Bearer), or "none" (local development only).
    auth_mode: Literal["api_key", "jwt", "none"] = "api_key"
    # Comma-separated "owner:key" pairs, or bare keys (owner derived from the key hash).
    api_keys: str = ""
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_audience: str | None = None

    media_signing_secret: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    media_url_ttl_seconds: int = 3600

    max_upload_mb: int = 50
    max_pages: int = 1500
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    rate_limit_upload: str = "10/minute"
    rate_limit_ask: str = "60/minute"

    # "process" = low-priority worker process started by the API (default, single command),
    # "huey" = separate durable worker service (compose), "thread" = in-process thread, "immediate" = inline (tests).
    jobs_mode: Literal["process", "huey", "thread", "immediate"] = "process"
    book_cache_size: int = 8

    gemini_api_key: str | None = None
    # Optional Azure OpenAI synthesis; used instead of Gemini when key, endpoint and deployment are all set.
    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_deployment_name: str | None = None
    openai_api_version: str | None = None
    azure_openai_reasoning_effort: str | None = None
    # "required": /readyz fails without fine-tuned weights; "base": use Florence-2-base; "disabled": skip Florence.
    florence_mode: Literal["required", "base", "disabled"] = "disabled"
    warm_models_on_start: bool = False
    log_level: str = "INFO"
    log_json: bool = True

    @field_validator("data_dir")
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @property
    def azure_config(self) -> AzureConfig | None:
        return AzureConfig.from_values(
            self.azure_openai_api_key,
            self.azure_openai_endpoint,
            self.azure_openai_deployment_name,
            self.openai_api_version,
            self.azure_openai_reasoning_effort,
        )

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip().rstrip("/") for o in self.cors_origins.split(",") if o.strip()]

    @property
    def db_url(self) -> str:
        return self.database_url or f"sqlite:///{(self.data_dir / 'registry.db').as_posix()}"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / ".cache" / "qa_index"


@lru_cache
def get_settings() -> Settings:
    return Settings()
