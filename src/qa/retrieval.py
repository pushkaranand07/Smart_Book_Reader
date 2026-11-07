"""Hybrid retrieval: BM25 + dense embeddings fused with Reciprocal Rank Fusion."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import bm25s
import numpy as np

from src.qa.backends import Backend, tokens
from src.qa.types import Chunk, ScoredChunk

logger = logging.getLogger(__name__)

RRF_K = 60
_EQUATION_QUERY = re.compile(r"\b(equation|reaction|formula|react|reacts|balanced|chemical change)\b", re.IGNORECASE)
_EQUATION_TEXT = re.compile(r"(→|->|⟶|\+\s*[A-Z][a-z]?\d*\s*(?:\(|→)|=)")


def wants_equation(query: str) -> bool:
    return bool(_EQUATION_QUERY.search(query))


@dataclass
class BookIndex:
    """Searchable index over one book's chunks. Build once per book, reuse for every question."""

    chunks: list[Chunk]
    embeddings: np.ndarray
    bm25: bm25s.BM25 | None
    fingerprint: str

    @classmethod
    def build(cls, chunks: list[Chunk], backend: Backend, cache_dir: Path | None = None) -> BookIndex:
        fingerprint = _fingerprint(chunks, backend.name)
        embeddings = _load_cached(cache_dir, fingerprint, len(chunks))
        if embeddings is None:
            embeddings = (
                backend.embed([_index_text(c) for c in chunks]) if chunks else np.zeros((0, 1), dtype=np.float32)
            )
            _save_cached(cache_dir, fingerprint, embeddings)
        bm25 = None
        corpus = [tokens(_index_text(c)) or ["_empty_"] for c in chunks]
        if chunks:
            bm25 = bm25s.BM25()
            bm25.index(corpus, show_progress=False)
        return cls(chunks=chunks, embeddings=embeddings, bm25=bm25, fingerprint=fingerprint)

    def search(self, query: str, backend: Backend, top_n: int = 20, pool: int = 50) -> list[ScoredChunk]:
        """Return the top_n chunks by RRF over BM25 and dense rankings."""
        if not self.chunks:
            return []
        pool = min(pool, len(self.chunks))
        fused: dict[int, float] = {}

        q_tokens = tokens(query)
        if self.bm25 is not None and q_tokens:
            ids, scores = self.bm25.retrieve([q_tokens], k=pool, show_progress=False)
            for rank, (idx, score) in enumerate(zip(ids[0].tolist(), scores[0].tolist(), strict=True)):
                if score > 0:
                    fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank + 1)

        q_vec = backend.embed([query], is_query=True)[0]
        dense = self.embeddings @ q_vec
        for rank, idx in enumerate(np.argsort(-dense)[:pool].tolist()):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank + 1)

        if wants_equation(query):
            for idx in list(fused):
                if _EQUATION_TEXT.search(self.chunks[idx].text):
                    fused[idx] += 0.5 / RRF_K

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        return [ScoredChunk(chunk=self.chunks[i], score=s, rrf=s) for i, s in ranked]


def _index_text(chunk: Chunk) -> str:
    return f"{chunk.section}. {chunk.text}" if chunk.section else chunk.text


def _fingerprint(chunks: list[Chunk], backend_name: str) -> str:
    digest = hashlib.sha256(backend_name.encode())
    for chunk in chunks:
        digest.update(_index_text(chunk).encode("utf-8", "ignore"))
    return digest.hexdigest()[:24]


def _load_cached(cache_dir: Path | None, fingerprint: str, expected_rows: int) -> np.ndarray | None:
    if cache_dir is None:
        return None
    path = cache_dir / f"{fingerprint}.npy"
    if not path.exists():
        return None
    try:
        arr = np.load(path)
    except (OSError, ValueError) as exc:
        logger.warning("discarding unreadable embedding cache %s: %s", path, exc)
        return None
    return arr if arr.shape[0] == expected_rows else None


def _save_cached(cache_dir: Path | None, fingerprint: str, embeddings: np.ndarray) -> None:
    if cache_dir is None:
        return
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / f"{fingerprint}.npy", embeddings)
    except OSError as exc:
        logger.warning("could not write embedding cache: %s", exc)
