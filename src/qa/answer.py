"""Offline answer composition from reranked evidence."""

from __future__ import annotations

import re

from src.qa.backends import Backend, Span, tokens
from src.qa.chunking import is_heading, split_sentences
from src.qa.types import ScoredChunk

MAX_SENTENCES = 3
_IMPERATIVE = re.compile(
    r"^(?:write|take|name|give|explain|list|identify|observe|note|collect|add|heat|pour|draw|find|state|"
    r"describe|compare|why|what|how|which|who|when|where|do you|can you|let us|try|record|repeat|discuss|"
    r"complete|fill|match|choose|mark|answer|activity|q\s?u\s?e\s?s|figure|fig\.|table)\b",
    re.IGNORECASE,
)
_EXERCISE_PREFIX = re.compile(r"^(?:\d+\.|\([a-z]\)|\([ivx]+\)|[a-z]\))\s")
_JUNK_PREFIX = re.compile(r"^(?:[nqlv•▪●■]\s+)+")


def clean_sentence(sentence: str) -> str:
    sentence = _JUNK_PREFIX.sub("", sentence.strip())
    return re.sub(r"\s+", " ", sentence).strip(" -–")


def echoes_question(sentence: str, question: str) -> bool:
    """True when the sentence mostly restates the question instead of answering it."""
    s, q = set(tokens(sentence)), set(tokens(question))
    if not s or not q:
        return False
    if s <= q:
        return True
    overlap = len(s & q) / len(s | q)
    return overlap >= 0.6 or sentence.lower().rstrip(" ?.").endswith(question.lower().rstrip(" ?."))


def is_answer_sentence(sentence: str, question: str) -> bool:
    words = sentence.split()
    if len(words) < 6 or len(sentence) > 450:
        return False
    if sentence.endswith("?") or is_heading(sentence):
        return False
    if _IMPERATIVE.match(sentence) or _EXERCISE_PREFIX.match(sentence):
        return False
    return not echoes_question(sentence, question)


def compose_offline_answer(
    question: str,
    evidence: list[ScoredChunk],
    span: Span | None,
    backend: Backend,
    sentence_floor: float,
) -> tuple[str, list[ScoredChunk]]:
    """Pick the best supporting sentences from the top evidence. Returns (answer, chunks_used)."""
    candidates: list[tuple[str, ScoredChunk, int]] = []
    seen: set[str] = set()
    for item in evidence[:3]:
        for pos, raw in enumerate(split_sentences(item.chunk.text)):
            sentence = clean_sentence(raw)
            key = " ".join(tokens(sentence))
            if key in seen or not is_answer_sentence(sentence, question):
                continue
            seen.add(key)
            candidates.append((sentence, item, pos))
    if not candidates:
        return (span.text if span else ""), evidence[:1]

    scores = backend.rerank(question, [c[0] for c in candidates])
    ranked = sorted(zip(scores, candidates, strict=True), key=lambda x: x[0], reverse=True)
    best_score = ranked[0][0]
    chosen = [c for s, c in ranked if s >= max(sentence_floor, best_score - 3.0)][:MAX_SENTENCES]

    if span is not None:
        span_hit = next((c for _, c in ranked if span.text and span.text in c[0]), None)
        if span_hit is not None and span_hit not in chosen:
            chosen = [span_hit, *chosen][:MAX_SENTENCES]

    # Keep reading order within the book so the answer flows naturally.
    order = {id(item): rank for rank, item in enumerate(evidence)}
    chosen.sort(key=lambda c: (c[1].chunk.page_number, order.get(id(c[1]), 0), c[2]))
    used: list[ScoredChunk] = []
    for _, item, _ in chosen:
        if item not in used:
            used.append(item)
    return " ".join(c[0] for c in chosen), used
