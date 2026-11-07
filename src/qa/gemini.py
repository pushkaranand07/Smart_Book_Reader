"""Optional Gemini synthesis with timeout, retries and a self-check for out-of-scope questions."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.qa.types import ScoredChunk

logger = logging.getLogger(__name__)

NOT_IN_BOOK = "NOT_IN_BOOK"
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
TIMEOUT_MS = int(os.environ.get("GEMINI_TIMEOUT_MS", "20000"))

PROMPT_TEMPLATE = """You answer questions about the document "{title}" using ONLY the excerpts below.

Rules:
- If the excerpts do not contain the answer, reply with exactly {not_in_book} and nothing else.
- Otherwise answer directly in 2-6 sentences of plain prose. Match the depth to the question.
- Do not restate the question. Do not add headings, page numbers or a sources section.
- Use LaTeX ($...$) only when the excerpts contain formulas or equations.

Excerpts:
{excerpts}

Question: {question}
Answer:"""


class GeminiUnavailable(RuntimeError):
    """Raised when Gemini cannot produce an answer; callers fall back to offline answering."""


@dataclass(frozen=True)
class GeminiAnswer:
    text: str
    in_scope: bool
    model: str


def build_prompt(question: str, title: str, evidence: list[ScoredChunk]) -> str:
    parts = []
    for item in evidence:
        section = f" | {item.chunk.section}" if item.chunk.section else ""
        parts.append(f"[page {item.chunk.page_number}{section}]\n{item.chunk.text}")
    return PROMPT_TEMPLATE.format(
        title=title or "the uploaded document",
        not_in_book=NOT_IN_BOOK,
        excerpts="\n\n".join(parts),
        question=question.strip(),
    )


def _transient_errors() -> tuple[type[BaseException], ...]:
    import httpx
    from google.genai import errors as genai_errors

    return (genai_errors.ServerError, httpx.TransportError, TimeoutError)


def _make_client(api_key: str) -> Any:
    from google import genai
    from google.genai import types

    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=TIMEOUT_MS))


def generate_answer(
    question: str,
    title: str,
    evidence: list[ScoredChunk],
    api_key: str,
    model: str = DEFAULT_MODEL,
    client: Any | None = None,
) -> GeminiAnswer:
    """Call Gemini with up to 3 attempts on transient errors. Raises GeminiUnavailable on failure."""
    import httpx
    from google.genai import errors as genai_errors

    prompt = build_prompt(question, title, evidence)
    client = client or _make_client(api_key)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
        retry=retry_if_exception_type(_transient_errors()),
    )
    def _call() -> str:
        response = client.models.generate_content(model=model, contents=prompt)
        return str(getattr(response, "text", "") or "").strip()

    try:
        text = _call()
    except (genai_errors.APIError, httpx.HTTPError, TimeoutError) as exc:
        logger.warning("gemini call failed, falling back to offline answer: %s", exc)
        raise GeminiUnavailable(str(exc)) from exc
    if not text:
        raise GeminiUnavailable("empty response")
    in_scope = NOT_IN_BOOK not in text.upper()
    return GeminiAnswer(text=text if in_scope else "", in_scope=in_scope, model=model)
