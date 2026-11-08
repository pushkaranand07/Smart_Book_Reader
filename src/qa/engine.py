"""QA orchestration: retrieve -> rerank -> decide sufficiency -> answer (offline, or an optional LLM provider)."""

from __future__ import annotations

import logging
import os
import re
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from src.qa.answer import compose_offline_answer
from src.qa.backends import Backend, Span, get_backend, sigmoid
from src.qa.chunking import chunk_pages
from src.qa.gemini import GeminiAnswer, GeminiUnavailable, generate_answer
from src.qa.grounding import verify
from src.qa.query import split_comparison, sub_question
from src.qa.retrieval import BookIndex
from src.qa.types import AnswerResult, Citation, Mode, ScoredChunk

logger = logging.getLogger(__name__)

NOT_COVERED = "This isn't covered in the book."
NOT_STATED = "The book mentions this topic but doesn't state the answer."


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
LLM_EVIDENCE = 8
# Hard wall-clock limit for one LLM call, whatever the SDK's own timeout does (a stuck socket once took an hour).
LLM_DEADLINE_S = _env_float("QA_LLM_DEADLINE_S", 60.0)


@dataclass(frozen=True)
class LLMProvider:
    """An optional answer generator. `generate` raises GeminiUnavailable on any failure."""

    name: str
    generate: Callable[[str, str, list[ScoredChunk]], GeminiAnswer]


@dataclass
class _Analysis:
    ranked: list[ScoredChunk]
    span: Span | None
    top: float
    margin: float

    @property
    def sufficient(self) -> bool:
        return self.top >= RERANK_THRESHOLD and (self.margin >= SPAN_MARGIN_THRESHOLD or self.top >= STRONG_RERANK)


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

    def _analyse(self, question: str, index: BookIndex) -> _Analysis:
        ranked = self.retrieve(question, index)
        if not ranked:
            return _Analysis([], None, float("-inf"), float("-inf"))
        span = self.backend.extract(question, [c.chunk.text for c in ranked[:3]])
        return _Analysis(ranked, span, ranked[0].score, span.margin if span else float("-inf"))

    def answer(
        self,
        question: str,
        pages: Sequence[Mapping[str, Any]],
        book_key: str,
        title: str = "",
        gemini_api_key: str | None = None,
        figure_ids_by_page: Mapping[int, Sequence[str]] | None = None,
        llm: LLMProvider | None = None,
    ) -> AnswerResult:
        question = question.strip()
        if not question:
            return AnswerResult("Please ask a question.", None, False, 0.0, "offline")
        index = self.index_for(book_key, pages)

        # Comparison questions ("difference between X and Y") are answered per concept, so both sides are covered.
        concepts = split_comparison(question)
        parts = [(c, self._analyse(sub_question(c), index)) for c in concepts] if concepts else []
        # For comparisons the whole question is only used to widen retrieval, so skip the (slow) reader on it.
        whole = _Analysis(self.retrieve(question, index), None, 0.0, 0.0) if parts else self._analyse(question, index)
        if not whole.ranked and not parts:
            return AnswerResult(NOT_COVERED, None, False, 0.0, "offline", diagnostics={"reason": "no_candidates"})

        span: Span | None
        if parts:
            top = min(a.top for _, a in parts)
            margin = min(a.margin for _, a in parts)
            sufficient = all(a.sufficient for _, a in parts)
            ranked = _interleave([a.ranked for _, a in parts] + [whole.ranked])
            span = None
        else:
            top, margin, sufficient, ranked, span = whole.top, whole.margin, whole.sufficient, whole.ranked, whole.span

        has_margin = margin != float("-inf")
        diagnostics: dict[str, Any] = {
            "top_rerank": round(top, 3),
            "span_margin": round(margin, 3) if has_margin else None,
            "backend": self.backend.name,
            "candidates": len(ranked),
        }
        if parts:
            diagnostics["comparison"] = [
                {"concept": c, "top_rerank": round(a.top, 3), "sufficient": a.sufficient} for c, a in parts
            ]
        confidence = sigmoid(top / 2.0) * (sigmoid(margin / 2.0) if has_margin else 0.5)

        if not sufficient:
            diagnostics["reason"] = "below_threshold"
            return AnswerResult(NOT_COVERED, None, False, confidence, "offline", diagnostics=diagnostics)

        used: list[ScoredChunk]
        if parts:
            pieces: list[str] = []
            used = []
            for concept, analysis in parts:
                text, chunks = compose_offline_answer(
                    sub_question(concept), analysis.ranked, analysis.span, self.backend, SENTENCE_FLOOR
                )
                if text:
                    pieces.append(f"{concept[:1].upper()}{concept[1:]}: {text}")
                used += [c for c in chunks if c not in used]
            answer_text, extractive = "\n\n".join(pieces), None
        else:
            answer_text, used = compose_offline_answer(question, ranked, span, self.backend, SENTENCE_FLOOR)
            extractive = span.text if span and margin >= SPAN_MARGIN_THRESHOLD else None
        mode = "offline"

        provider = llm
        if provider is None and gemini_api_key:
            key = gemini_api_key
            provider = LLMProvider("gemini", lambda q, t, e: generate_answer(q, t, e, key))
        if provider is not None:
            # Long books need more context than the top 5: chapter-level questions span several passages.
            evidence = ranked[:LLM_EVIDENCE]
            try:
                generated = _call_with_deadline(provider, question, title, evidence)
            except GeminiUnavailable as exc:
                diagnostics[f"{provider.name}_error"] = str(exc)[:200]
            else:
                diagnostics[f"{provider.name}_model"] = generated.model
                if not generated.in_scope:
                    diagnostics["reason"] = f"{provider.name}_self_check"
                    return AnswerResult(
                        NOT_COVERED,
                        None,
                        False,
                        confidence,
                        cast(Mode, provider.name),
                        diagnostics=diagnostics,
                    )
                # The model may only say what the passages support: drop every sentence they do not entail.
                grounding = verify(generated.text, evidence, self.backend)
                diagnostics["grounding"] = {
                    "kept": len(grounding.kept),
                    "dropped": len(grounding.dropped),
                    "scores": grounding.scores,
                }
                if not grounding.kept:
                    diagnostics["reason"] = "llm_answer_not_supported_by_book"
                    return AnswerResult(
                        NOT_STATED,
                        None,
                        False,
                        confidence,
                        cast(Mode, provider.name),
                        diagnostics=diagnostics,
                    )
                answer_text, mode, used = grounding.text, provider.name, evidence[:CITE_TOP]

        if not answer_text:
            answer_text = extractive or ranked[0].chunk.text[:300]
        cited = used or ranked[:1]
        citations = [_citation(c, figure_ids_by_page) for c in cited[: CITE_TOP + (1 if parts else 0)]]
        return AnswerResult(
            answer=answer_text,
            extractive_answer=extractive,
            is_sufficient=True,
            confidence=confidence,
            mode=cast(Mode, mode),
            citations=citations,
            evidence=ranked[:5],
            diagnostics=diagnostics,
        )


_llm_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="llm-call")


def _call_with_deadline(provider: LLMProvider, question: str, title: str, evidence: list[ScoredChunk]) -> GeminiAnswer:
    """Run the provider call with a hard deadline; on expiry give up (the offline answer is used instead)."""
    future = _llm_pool.submit(provider.generate, question, title, evidence)
    try:
        return future.result(timeout=LLM_DEADLINE_S)
    except FutureTimeout as exc:
        future.cancel()
        logger.warning("%s call exceeded %.0fs; using the offline answer", provider.name, LLM_DEADLINE_S)
        raise GeminiUnavailable(f"{provider.name}: no response within {LLM_DEADLINE_S:.0f}s") from exc


def _interleave(rankings: list[list[ScoredChunk]], per_list: int = 4) -> list[ScoredChunk]:
    """Round-robin merge of several rankings, de-duplicated by chunk id."""
    merged: list[ScoredChunk] = []
    seen: set[str] = set()
    for position in range(per_list):
        for ranking in rankings:
            if position < len(ranking) and ranking[position].chunk.chunk_id not in seen:
                seen.add(ranking[position].chunk.chunk_id)
                merged.append(ranking[position])
    return merged


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
