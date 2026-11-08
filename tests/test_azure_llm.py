"""Optional Azure OpenAI provider (fake client; no network)."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import openai
import pytest

from src.qa import azure_llm
from src.qa.azure_llm import AzureConfig
from src.qa.gemini import GeminiUnavailable
from src.qa.types import Chunk, ScoredChunk

EVIDENCE = [ScoredChunk(Chunk("c1", 12, "Oxidation is the gain of oxygen by a substance.", "1.2 Oxidation"), 5.0)]
CONFIG = AzureConfig(api_key="k", endpoint="https://x.cognitiveservices.azure.com/", deployment="gpt-5")


def test_config_requires_key_endpoint_and_deployment():
    assert AzureConfig.from_values(None, "https://e", "d") is None
    assert AzureConfig.from_values("k", "", "d") is None
    assert AzureConfig.from_values("k", "https://e", None) is None
    cfg = AzureConfig.from_values(" k ", "https://e", " gpt-5 ", None, "Medium")
    assert cfg is not None and cfg.api_key == "k" and cfg.deployment == "gpt-5"
    assert cfg.api_version == azure_llm.DEFAULT_API_VERSION and cfg.reasoning_effort == "medium"
    off = AzureConfig.from_values("k", "https://e", "d", "2025-04-01-preview", "none")
    assert off is not None and off.reasoning_effort is None


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=outcome))])


def fake_client(*outcomes):
    completions = FakeCompletions(outcomes)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://x.cognitiveservices.azure.com/openai")


def test_success_sends_reasoning_params_and_no_temperature():
    client, completions = fake_client("Oxidation is the gain of oxygen.")
    answer = azure_llm.generate_answer("What is oxidation?", "Book", EVIDENCE, CONFIG, client=client)
    assert answer.in_scope and answer.text == "Oxidation is the gain of oxygen." and answer.model == "azure:gpt-5"
    sent = completions.calls[0]
    assert sent["model"] == "gpt-5" and sent["reasoning_effort"] == "low" and "max_completion_tokens" in sent
    assert "temperature" not in sent
    assert '"Book"' in sent["messages"][0]["content"]


def test_not_in_book_reply_is_out_of_scope():
    client, _ = fake_client("NOT_IN_BOOK")
    answer = azure_llm.generate_answer("Who won?", "Book", EVIDENCE, CONFIG, client=client)
    assert not answer.in_scope and answer.text == ""


def test_empty_reply_raises_unavailable():
    client, _ = fake_client("")
    with pytest.raises(GeminiUnavailable, match="empty response"):
        azure_llm.generate_answer("q", "Book", EVIDENCE, CONFIG, client=client)


def test_transient_errors_are_retried(monkeypatch):
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda _s: None)
    client, completions = fake_client(openai.APIConnectionError(request=_request()), "Recovered.")
    answer = azure_llm.generate_answer("q", "Book", EVIDENCE, CONFIG, client=client)
    assert answer.text == "Recovered." and len(completions.calls) == 2


def test_permanent_error_raises_unavailable_without_retry():
    response = httpx.Response(403, request=_request())
    error = openai.PermissionDeniedError(
        "Access denied due to Virtual Network/Firewall rules.", response=response, body=None
    )
    client, completions = fake_client(error)
    with pytest.raises(GeminiUnavailable, match="PermissionDeniedError"):
        azure_llm.generate_answer("q", "Book", EVIDENCE, CONFIG, client=client)
    assert len(completions.calls) == 1


def test_reasoning_effort_omitted_when_disabled():
    client, completions = fake_client("Oxidation is the gain of oxygen.")
    config = AzureConfig(api_key="k", endpoint="https://e", deployment="gpt-4o-mini", reasoning_effort=None)
    azure_llm.generate_answer("q", "Book", EVIDENCE, config, client=client)
    assert "reasoning_effort" not in completions.calls[0]
