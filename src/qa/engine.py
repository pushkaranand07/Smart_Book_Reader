"""QA orchestration: retrieve -> rerank -> decide sufficiency -> answer (offline or Gemini)."""

from __future__ import annotations

import logging
import os
import re
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.qa.answer import compose_offline_answer
from src.qa.backends import Backend, get_backend, sigmoid
from src.qa.chunking import chunk_pages
from src.qa.gemini import GeminiUnavailable, generate_answer
from src.qa.retrieval import BookIndex
from src.qa.types import AnswerResult, Citation, ScoredChunk

logger = logging.getLogger(__name__)

NOT_COVERED = "This isn't covered in the book."


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


# Thresholds are calibrated on tests/eval/qa_dev.jsonl (scripts/eval/run_qa_eval.py --calibrate) and set to the
# middle of the observed gap: unanswerable top rerank <= -5.7, answerable >= +0.1 (MiniLM cross-encoder logits).
RERANK_THRESHOLD = _env_float("QA_RERANK_THRESHOLD", -2.5)
STRONG_RERANK = _env_float("QA_STRONG_RERANK", 3.0)
SPAN_MARGIN_THRESHOLD = _env_float("QA_SPAN_MARGIN_THRESHOLD", 0.0)
SENTENCE_FLOOR = _env_float("QA_SENTENCE_FLOOR", -3.0)
EXERCISE_PENALTY = 3.0
RERANK_POOL = 20
CITE_TOP = 3


class QAEngine:
    """Answers questions about books. Holds an LRU of built indexes keyed by book id."""

    def __init__(self, backend: Backend | None = None, cache_dir: Path | None = None, max_books: int = 16) -> None:
        self._backend = backend
        self.cache_dir = cache_dir
        self.max_books = max_books
        self._indexes: OrderedDict[str, BookIndex] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def backend(self) -> Backend:
        return self._backend or get_backend()

    def index_for(self, book_key: str, pages: Sequence[Mapping[str, Any]]) -> BookIndex:
        with self._lock:
            cached = self._indexes.get(book_key)
            if cached is not None:
                self._indexes.move_to_end(book_key)
                return cached
        chunks = chunk_pages(pages, document_id=re.sub(r"[^A-Za-z0-9_-]", "_", book_key)[:40] or "doc")
        index = BookIndex.build(chunks, self.backend, cache_dir=self.cache_dir)
        with self._lock:
            self._indexes[book_key] = index
            while len(self._indexes) > self.max_books:
                self._indexes.popitem(last=False)
        return index

    def retrieve(self, question: str, index: BookIndex) -> list[ScoredChunk]:
        candidates = index.search(question, self.backend, top_n=RERANK_POOL)
        if not candidates:
            return []
        texts = [f"{c.chunk.section}. {c.chunk.text}" if c.chunk.section else c.chunk.text for c in candidates]
        scores = self.backend.rerank(question, texts)
        rescored = [
            ScoredChunk(c.chunk, s - (EXERCISE_PENALTY if c.chunk.kind == "exercise" else 0.0), c.rrf)
            for c, s in zip(candidates, scores, strict=True)
        ]
        return sorted(rescored, key=lambda c: c.score, reverse=True)

    def answer(
        self,
        question: str,
        pages: Sequence[Mapping[str, Any]],
        book_key: str,
        title: str = "",
        gemini_api_key: str | None = None,
        figure_ids_by_page: Mapping[int, Sequence[str]] | None = None,
    ) -> AnswerResult:
        question = question.strip()
        if not question:
            return AnswerResult("Please ask a question.", None, False, 0.0, "offline")
        index = self.index_for(book_key, pages)
        ranked = self.retrieve(question, index)
        if not ranked:
            return AnswerResult(NOT_COVERED, None, False, 0.0, "offline", diagnostics={"reason": "no_candidates"})

        top_score = ranked[0].score
        span = self.backend.extract(question, [c.chunk.text for c in ranked[:3]])
        margin = span.margin if span else float("-inf")
        sufficient = top_score >= RERANK_THRESHOLD and (margin >= SPAN_MARGIN_THRESHOLD or top_score >= STRONG_RERANK)
        diagnostics: dict[str, Any] = {
            "top_rerank": round(top_score, 3),
            "span_margin": round(margin, 3) if span else None,
            "backend": self.backend.name,
            "candidates": len(ranked),
        }
        confidence = sigmoid(top_score / 2.0) * (sigmoid(margin / 2.0) if span else 0.5)

        if not sufficient:
            diagnostics["reason"] = "below_threshold"
            return AnswerResult(NOT_COVERED, None, False, confidence, "offline", diagnostics=diagnostics)

        answer_text, used = compose_offline_answer(question, ranked, span, self.backend, SENTENCE_FLOOR)
        extractive = span.text if span and margin >= SPAN_MARGIN_THRESHOLD else None
        mode: str = "offline"

        if gemini_api_key:
            try:
                generated = generate_answer(question, title, ranked[:5], gemini_api_key)
            except GeminiUnavailable as exc:
                diagnostics["gemini_error"] = str(exc)[:200]
            else:
                diagnostics["gemini_model"] = generated.model
                if not generated.in_scope:
                    diagnostics["reason"] = "gemini_self_check"
                    return AnswerResult(NOT_COVERED, None, False, confidence, "gemini", diagnostics=diagnostics)
                answer_text, mode, used = generated.text, "gemini", ranked[:CITE_TOP]

        if not answer_text:
            answer_text = extractive or ranked[0].chunk.text[:300]
        cited = used or ranked[:1]
        citations = [_citation(c, figure_ids_by_page) for c in cited[:CITE_TOP]]
        return AnswerResult(
            answer=answer_text,
            extractive_answer=extractive,
            is_sufficient=True,
            confidence=confidence,
            mode="gemini" if mode == "gemini" else "offline",
            citations=citations,
            evidence=ranked[:5],
            diagnostics=diagnostics,
        )


def _citation(item: ScoredChunk, figure_ids_by_page: Mapping[int, Sequence[str]] | None) -> Citation:
    chunk = item.chunk
    known = set(figure_ids_by_page.get(chunk.page_number, ())) if figure_ids_by_page else set()
    figure_ids = [ref for ref in chunk.figure_refs if not known or ref in known]
    snippet = chunk.text if len(chunk.text) <= 280 else chunk.text[:277].rsplit(" ", 1)[0] + "..."
    return Citation(
        page_number=chunk.page_number,
        section=chunk.section,
        chunk_id=chunk.chunk_id,
        snippet=snippet,
        figure_ids=figure_ids,
    )


_engine_lock = threading.Lock()
_engine: QAEngine | None = None


def get_engine() -> QAEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            cache = os.environ.get("QA_INDEX_DIR")
            _engine = QAEngine(cache_dir=Path(cache) if cache else None)
        return _engine
