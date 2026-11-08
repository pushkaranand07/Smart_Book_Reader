"""Optional Azure OpenAI answer synthesis (same contract and safety rules as the Gemini path).

Configured by AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME and OPENAI_API_VERSION.
Reasoning models (gpt-5, o-series) reject `temperature`; only `max_completion_tokens` and `reasoning_effort` are sent.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.qa.gemini import NOT_IN_BOOK, GeminiAnswer, GeminiUnavailable, build_prompt
from src.qa.types import ScoredChunk

logger = logging.getLogger(__name__)
DEFAULT_API_VERSION = "2025-04-01-preview"


@dataclass(frozen=True)
class AzureConfig:
    api_key: str
    endpoint: str
    deployment: str
    api_version: str = DEFAULT_API_VERSION
    reasoning_effort: str | None = "low"
    timeout_s: float = 45.0
    max_completion_tokens: int = 1200

    @classmethod
    def from_values(
        cls,
        api_key: str | None,
        endpoint: str | None,
        deployment: str | None,
        api_version: str | None = None,
        reasoning_effort: str | None = None,
    ) -> AzureConfig | None:
        """Return a config only when key, endpoint and deployment are all present."""
        if not (api_key and endpoint and deployment):
            return None
        effort = (reasoning_effort or os.environ.get("AZURE_OPENAI_REASONING_EFFORT") or "low").strip().lower()
        return cls(
            api_key=api_key.strip(),
            endpoint=endpoint.strip(),
            deployment=deployment.strip(),
            api_version=(api_version or DEFAULT_API_VERSION).strip(),
            reasoning_effort=None if effort in {"", "none", "off"} else effort,
        )


def _make_client(config: AzureConfig) -> Any:
    from openai import AzureOpenAI

    return AzureOpenAI(
        api_key=config.api_key,
        azure_endpoint=config.endpoint,
        api_version=config.api_version,
        timeout=config.timeout_s,
        max_retries=0,  # retries are handled below so only transient errors are retried
    )


def generate_answer(
    question: str,
    title: str,
    evidence: list[ScoredChunk],
    config: AzureConfig,
    client: Any | None = None,
) -> GeminiAnswer:
    """Ask the Azure deployment for an answer grounded in `evidence`. Raises GeminiUnavailable on failure."""
    import openai

    prompt = build_prompt(question, title, evidence)
    client = client or _make_client(config)
    transient = (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, openai.InternalServerError)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.0, max=8),
        retry=retry_if_exception_type(transient),
    )
    def _call() -> str:
        kwargs: dict[str, Any] = {
            "model": config.deployment,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": config.max_completion_tokens,
        }
        if config.reasoning_effort:
            kwargs["reasoning_effort"] = config.reasoning_effort
        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0] if response.choices else None
        return str(getattr(getattr(choice, "message", None), "content", "") or "").strip()

    try:
        text = _call()
    except openai.OpenAIError as exc:
        logger.warning("azure openai call failed, falling back to offline answer: %s", exc)
        raise GeminiUnavailable(f"azure openai: {exc.__class__.__name__}") from exc
    if not text:
        # Reasoning models can spend the whole token budget thinking and return no visible text.
        raise GeminiUnavailable("azure openai: empty response")
    in_scope = NOT_IN_BOOK not in text.upper()
    return GeminiAnswer(text=text if in_scope else "", in_scope=in_scope, model=f"azure:{config.deployment}")
