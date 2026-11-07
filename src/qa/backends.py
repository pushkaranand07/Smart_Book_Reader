"""Model backends for embedding, reranking and extractive QA.

`TransformerBackend` uses real models (production and eval). `LexicalBackend` is a deterministic,
dependency-free stand-in used by unit tests and as a degraded fallback when models cannot load.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)

EMBED_MODEL = os.environ.get("QA_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.environ.get("QA_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
READER_MODEL = os.environ.get("QA_READER_MODEL", "deepset/roberta-base-squad2")
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the of and or to in on for is are was were be been by with as at from that this these those it its "
    "what which who whom how why when where do does did can could should would will shall may might me you your "
    "we our they their them he she his her i give explain describe define tell about".split()
)


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


@dataclass(frozen=True)
class Span:
    text: str
    score: float
    null_score: float
    context_index: int

    @property
    def margin(self) -> float:
        """Positive when the reader prefers a real span over 'no answer'."""
        return self.score - self.null_score


class Backend(Protocol):
    name: str

    def embed(self, texts: list[str], *, is_query: bool = False) -> np.ndarray: ...

    def rerank(self, query: str, texts: list[str]) -> list[float]: ...

    def extract(self, question: str, contexts: list[str]) -> Span | None: ...


class LexicalBackend:
    """Hashing embeddings + token-overlap scoring. Fast, deterministic, no downloads."""

    name = "lexical"
    dim = 512

    def embed(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in tokens(text):
                digest = hashlib.blake2b(tok.encode(), digest_size=4).digest()
                out[row, int.from_bytes(digest, "little") % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        q = set(tokens(query))
        if not q:
            return [-10.0 for _ in texts]
        scores = []
        for text in texts:
            hits = len(q & set(tokens(text)))
            # Map overlap ratio onto a logit-like scale comparable to the MiniLM cross-encoder.
            scores.append(-10.0 + 18.0 * hits / len(q))
        return scores

    def extract(self, question: str, contexts: list[str]) -> Span | None:
        q = set(tokens(question))
        best: Span | None = None
        for idx, ctx in enumerate(contexts):
            for sentence in re.split(r"(?<=[.!?])\s+", ctx):
                hits = len(q & set(tokens(sentence)))
                if not hits:
                    continue
                score = 10.0 * hits / max(1, len(q))
                if best is None or score > best.score:
                    best = Span(sentence.strip(), score, 5.0, idx)
        return best


class TransformerBackend:
    """bge-small embeddings, MiniLM cross-encoder reranker, roberta-base-squad2 reader."""

    name = "transformers"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._embedder: Any = None
        self._reranker: Any = None
        self._reader: tuple[Any, Any] | None = None

    def _get_embedder(self) -> Any:
        with self._lock:
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
            return self._embedder

    def _get_reranker(self) -> Any:
        with self._lock:
            if self._reranker is None:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(RERANK_MODEL, device="cpu")
            return self._reranker

    def _get_reader(self) -> tuple[Any, Any]:
        with self._lock:
            if self._reader is None:
                from transformers import AutoModelForQuestionAnswering, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(READER_MODEL)
                model = AutoModelForQuestionAnswering.from_pretrained(READER_MODEL).eval()
                self._reader = (tokenizer, model)
            return self._reader

    def warm_up(self) -> None:
        self._get_embedder()
        self._get_reranker()
        self._get_reader()

    def embed(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        if is_query:
            texts = [_BGE_QUERY_PREFIX + t for t in texts]
        vectors = self._get_embedder().encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        )
        return np.asarray(vectors, dtype=np.float32)

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores = self._get_reranker().predict([(query, t) for t in texts], batch_size=32, show_progress_bar=False)
        return [float(s) for s in scores]

    def extract(self, question: str, contexts: list[str]) -> Span | None:
        import torch

        tokenizer, model = self._get_reader()
        best: Span | None = None
        for idx, context in enumerate(contexts):
            enc = tokenizer(
                question,
                context,
                return_tensors="pt",
                truncation="only_second",
                max_length=384,
                return_offsets_mapping=True,
            )
            offsets = enc.pop("offset_mapping")[0].tolist()
            sequence_ids = enc.sequence_ids(0)
            with torch.no_grad():
                out = model(**enc)
            start = out.start_logits[0]
            end = out.end_logits[0]
            null_score = float(start[0] + end[0])
            ctx_positions = [i for i, sid in enumerate(sequence_ids) if sid == 1]
            if not ctx_positions:
                continue
            lo, hi = ctx_positions[0], ctx_positions[-1]
            top_starts = torch.topk(start[lo : hi + 1], k=min(20, hi - lo + 1)).indices.tolist()
            top_ends = torch.topk(end[lo : hi + 1], k=min(20, hi - lo + 1)).indices.tolist()
            for s_rel in top_starts:
                for e_rel in top_ends:
                    s_i, e_i = s_rel + lo, e_rel + lo
                    if e_i < s_i or e_i - s_i > 40:
                        continue
                    score = float(start[s_i] + end[e_i])
                    if best is None or score - null_score > best.margin:
                        text = context[offsets[s_i][0] : offsets[e_i][1]].strip()
                        if text:
                            best = Span(text, score, null_score, idx)
        return best


_backend_lock = threading.Lock()
_backend: Backend | None = None


def get_backend() -> Backend:
    """Return the process-wide backend selected by QA_BACKEND (transformers | lexical)."""
    global _backend
    with _backend_lock:
        if _backend is None:
            choice = os.environ.get("QA_BACKEND", "transformers").lower()
            _backend = LexicalBackend() if choice == "lexical" else TransformerBackend()
            logger.info("qa backend selected: %s", _backend.name)
        return _backend


def set_backend(backend: Backend | None) -> None:
    """Override the backend (tests) or reset it so QA_BACKEND is re-read."""
    global _backend
    with _backend_lock:
        _backend = backend


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
