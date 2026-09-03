"""
semantic_search.py — Advanced semantic retrieval using sentence-transformers + FAISS.

Replaces keyword-based search_pages() with dense vector similarity search.
Model: BAAI/bge-large-en-v1.5 (state-of-art retrieval, works on CPU, no API key).
Index: FAISS FlatIP (cosine similarity via L2-normalised inner product).
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Lazy imports so the module loads even if packages are missing ─────────────
_st_lock = threading.Lock()
_embedder = None
_LOCAL_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "bge-large-en-v1.5"))
_MODEL_NAME = "BAAI/bge-large-en-v1.5"
_FALLBACK_MODEL = "all-MiniLM-L6-v2"


def _get_embedder():
    """Lazy-load the sentence-transformer model (thread-safe, loaded once)."""
    global _embedder
    if _embedder is not None:
        return _embedder
    with _st_lock:
        if _embedder is None:
            from sentence_transformers import SentenceTransformer
            # 1. Prefer local pre-downloaded weights
            if os.path.isdir(_LOCAL_MODEL_PATH):
                try:
                    _embedder = SentenceTransformer(_LOCAL_MODEL_PATH)
                    return _embedder
                except Exception:
                    pass
            # 2. Try huggingface hub model
            try:
                _embedder = SentenceTransformer(_MODEL_NAME)
                return _embedder
            except Exception:
                pass
            # 3. Fallback to lightweight model
            try:
                _embedder = SentenceTransformer(_FALLBACK_MODEL)
                return _embedder
            except Exception as e:
                raise RuntimeError(
                    f"Could not load any sentence-transformer model: {e}"
                )
    return _embedder


def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Encode a list of texts into L2-normalised embedding vectors.

    BGE models work best when you prepend 'Represent this sentence: ' for passages.
    For queries, no prefix is needed.
    """
    model = _get_embedder()
    # BGE passage prefix for document chunks
    prefixed = [f"Represent this sentence: {t}" for t in texts]
    embeddings = model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,   # L2 normalise → cosine = inner product
        convert_to_numpy=True,
    )
    return embeddings.astype("float32")


def embed_query(query: str) -> np.ndarray:
    """Encode a single search query (no passage prefix for BGE queries)."""
    model = _get_embedder()
    vec = model.encode(
        [query],
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vec.astype("float32")


# ── FAISS Index ───────────────────────────────────────────────────────────────

class SemanticIndex:
    """Wraps a FAISS flat index over page embeddings for fast cosine search."""

    def __init__(self):
        self.index = None
        self.pages: List[Dict[str, Any]] = []
        self._dim: Optional[int] = None

    def build(self, pages: List[Dict[str, Any]]) -> None:
        """Encode all page texts and build a FAISS inner-product index."""
        import faiss

        texts = []
        valid_pages = []
        for p in pages:
            p_dict = p.to_dict() if hasattr(p, "to_dict") else p
            text = p_dict.get("text", "").strip()
            if text:
                texts.append(text[:2000])   # cap at 2000 chars per page
                valid_pages.append(p_dict)

        if not texts:
            return

        embeddings = embed_texts(texts)
        dim = embeddings.shape[1]
        self._dim = dim

        # FlatIP = exact inner product (cosine on normalised vectors)
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.pages = valid_pages

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.35,
    ) -> List[Dict[str, Any]]:
        """Return top-k most semantically similar pages with cosine scores."""
        if self.index is None or not self.pages:
            return []

        q_vec = embed_query(query)
        k = min(top_k * 2, len(self.pages))
        scores, indices = self.index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if float(score) < score_threshold:
                continue
            page = dict(self.pages[idx])
            page["semantic_score"] = float(score)
            # Build a short snippet (first 300 chars of text)
            text = page.get("text", "")
            page["snippet"] = text[:300].strip() + ("..." if len(text) > 300 else "")
            results.append(page)
            if len(results) >= top_k:
                break

        return results

    @property
    def is_ready(self) -> bool:
        return self.index is not None and len(self.pages) > 0


# ── Module-level index singleton (one per uploaded book) ─────────────────────

_current_index: Optional[SemanticIndex] = None
_index_book_id: Optional[str] = None


def build_semantic_index(pages: List[Dict[str, Any]], book_id: str) -> SemanticIndex:
    """Build (or reuse if same book) the FAISS semantic index for a book's pages."""
    global _current_index, _index_book_id

    if _current_index is not None and _index_book_id == book_id and _current_index.is_ready:
        return _current_index

    idx = SemanticIndex()
    idx.build(pages)
    _current_index = idx
    _index_book_id = book_id
    return idx


def semantic_search_pages(
    pages: List[Dict[str, Any]],
    query: str,
    top_k: int = 5,
    book_id: str = "default",
    score_threshold: float = 0.35,
) -> List[Dict[str, Any]]:
    """Main entry point: semantic search over a list of pages.

    Builds the FAISS index on first call (cached for same book_id).
    Falls back to keyword search if sentence-transformers is unavailable.

    Returns list of page dicts with 'semantic_score' and 'snippet' added.
    """
    try:
        idx = build_semantic_index(pages, book_id)
        if not idx.is_ready:
            raise RuntimeError("FAISS index is empty.")
        return idx.search(query, top_k=top_k, score_threshold=score_threshold)
    except Exception:
        # Graceful fallback to keyword search
        from src.search import search_pages as keyword_search
        return keyword_search(pages, query, top_k=top_k)
