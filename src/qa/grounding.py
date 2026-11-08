"""Check that LLM answers are supported by the retrieved book passages.

Every sentence of a generated answer is scored with an NLI model (premise = book passage, hypothesis = the
sentence). Sentences the passages do not entail are removed, so the model cannot add outside knowledge.

NLI on long passages is slow on CPU (~0.3 s per pair), so each sentence is only checked against the two passages
that share the most words with it, plus those two joined together when neither supports it on its own.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from src.qa.backends import Backend, tokens
from src.qa.chunking import split_sentences
from src.qa.types import ScoredChunk

ENTAILMENT_THRESHOLD = float(os.environ.get("QA_ENTAILMENT_THRESHOLD", "0.5"))
CANDIDATES_PER_SENTENCE = 2
_MAX_PREMISE_WORDS = 320


@dataclass
class GroundingResult:
    kept: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(self.kept)

    @property
    def fully_grounded(self) -> bool:
        return not self.dropped


def _trim(text: str) -> str:
    return " ".join(text.split()[:_MAX_PREMISE_WORDS])


def _formula_supported(sentence: str, evidence_text: str) -> bool:
    """Equations and very short fragments: keep them only if every token appears in the passages."""
    toks = re.findall(r"[A-Za-z0-9]+", sentence)
    haystack = set(re.findall(r"[A-Za-z0-9]+", evidence_text))
    return bool(toks) and all(t in haystack for t in toks)


def _closest_passages(sentence: str, passages: list[str], token_sets: list[set[str]]) -> list[str]:
    words = set(tokens(sentence))
    order = sorted(range(len(passages)), key=lambda i: len(words & token_sets[i]), reverse=True)
    return [passages[i] for i in order[:CANDIDATES_PER_SENTENCE]]


def verify(answer: str, evidence: list[ScoredChunk], backend: Backend) -> GroundingResult:
    result = GroundingResult()
    sentences = [s.strip() for s in split_sentences(re.sub(r"\s+", " ", answer)) if s.strip()]
    if not sentences or not evidence:
        result.dropped = sentences
        return result
    passages = [c.chunk.text for c in evidence[:8]]
    token_sets = [set(tokens(p)) for p in passages]
    evidence_text = " ".join(passages)

    scores: dict[int, float] = {}
    candidates: dict[int, list[str]] = {}
    for i, sentence in enumerate(sentences):
        if len(tokens(sentence)) < 4:
            scores[i] = 1.0 if _formula_supported(sentence, evidence_text) else 0.0
        else:
            candidates[i] = _closest_passages(sentence, passages, token_sets)

    # Pass 1: each sentence against its closest single passages, in one batch.
    pairs = [(i, _trim(p)) for i, cands in candidates.items() for p in cands]
    for (i, _), prob in zip(pairs, backend.entailment([(p, sentences[i]) for i, p in pairs]), strict=True):
        scores[i] = max(scores.get(i, 0.0), prob)

    # Pass 2: sentences that combine two passages get the two closest passages joined as one premise.
    retry = [i for i in candidates if scores.get(i, 0.0) < ENTAILMENT_THRESHOLD and len(candidates[i]) > 1]
    if retry:
        joined = [(i, _trim(" ".join(candidates[i]))) for i in retry]
        for (i, _), prob in zip(joined, backend.entailment([(p, sentences[i]) for i, p in joined]), strict=True):
            scores[i] = max(scores[i], prob)

    for i, sentence in enumerate(sentences):
        score = scores.get(i, 0.0)
        result.scores.append(round(score, 3))
        (result.kept if score >= ENTAILMENT_THRESHOLD else result.dropped).append(sentence)
    return result
