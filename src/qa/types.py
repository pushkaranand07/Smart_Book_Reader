"""Data types shared across the QA package."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Mode = Literal["offline", "gemini", "azure_openai"]


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage with its provenance."""

    chunk_id: str
    page_number: int
    text: str
    section: str | None = None
    kind: Literal["body", "exercise"] = "body"
    figure_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    rrf: float = 0.0


@dataclass
class Citation:
    page_number: int
    section: str | None
    chunk_id: str
    snippet: str
    figure_ids: list[str] = field(default_factory=list)


@dataclass
class AnswerResult:
    answer: str
    extractive_answer: str | None
    is_sufficient: bool
    confidence: float
    mode: Mode
    citations: list[Citation] = field(default_factory=list)
    evidence: list[ScoredChunk] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "extractive_answer": self.extractive_answer,
            "is_sufficient": self.is_sufficient,
            "confidence": round(self.confidence, 4),
            "mode": self.mode,
            "citations": [asdict(c) for c in self.citations],
            "diagnostics": self.diagnostics,
        }
