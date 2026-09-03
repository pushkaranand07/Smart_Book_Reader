import re
from typing import Any, Dict, List

STOP_WORDS = {
    "a", "an", "the", "what", "is", "are", "was", "were", "of", "in",
    "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "and", "or", "so", "but", "if",
    "then", "this", "that", "it", "its", "they", "them", "their",
    "i", "you", "he", "she", "we", "me", "my", "your", "his", "her",
    "do", "does", "did", "can", "could", "will", "would", "should",
    "give", "explain", "describe", "define", "list", "name", "show",
}

# Common science/chemistry stem suffixes for fuzzy matching
_STEM_SUFFIXES = (
    "ation", "ated", "ating", "ates", "ate",
    "ing", "ed", "s", "es", "er", "ion", "ions",
)


def clean_and_tokenize(text: str) -> List[str]:
    """Normalize text to lowercase, remove punctuation, and extract tokens."""
    tokens = re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())
    return tokens


def stem_word(word: str) -> str:
    """Lightweight suffix-stripping to improve recall for scientific terms.

    e.g. 'saturated' -> 'satur', 'unsaturated' -> 'unsatur', 'saturation' -> 'satur'
    """
    if len(word) <= 4:
        return word
    for suffix in _STEM_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: len(word) - len(suffix)]
    return word


def extract_meaningful_keywords(query: str) -> List[str]:
    """Normalize query, remove stop words, and return meaningful search terms."""
    tokens = clean_and_tokenize(query)
    keywords = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    # Fallback to original tokens if all were stop words
    return keywords if keywords else tokens


def extract_bigrams(tokens: List[str]) -> List[str]:
    """Generate adjacent word-pair bigrams from a token list."""
    return [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]


def extract_snippet(text: str, keywords: List[str], max_length: int = 300) -> str:
    """Find the most relevant text snippet containing matched keywords."""
    if not text:
        return ""

    lower_text = text.lower()
    best_pos = -1

    for kw in keywords:
        pos = lower_text.find(kw)
        if pos != -1 and (best_pos == -1 or pos < best_pos):
            best_pos = pos

    if best_pos == -1:
        return text[:max_length] + ("..." if len(text) > max_length else "")

    start = max(0, best_pos - 80)
    end = min(len(text), start + max_length)
    snippet = text[start:end].strip()

    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."

    return snippet


def _score_page(
    text: str,
    text_lower: str,
    tokens: List[str],
    keywords: List[str],
    query_lower: str,
) -> tuple:
    """Compute a multi-factor relevance score for a page.

    Scoring layers (highest → lowest priority):
      1. Exact token match          — weight ×3 per occurrence
      2. Stem match                 — weight ×2 per occurrence (handles OCR variants)
      3. Substring match (≥5 chars) — weight +2 (catches OCR-split words)
      4. Prefix match (≥8 chars)    — weight +1
      5. Exact phrase bonus         — +15
      6. Consecutive keyword bigram — +8 per bigram hit

    Returns (score, matched_terms).
    """
    score = 0
    matched_terms: List[str] = []
    stemmed_tokens = [stem_word(t) for t in tokens]

    for kw in keywords:
        kw_stem = stem_word(kw)

        # Layer 1: exact token match
        count = tokens.count(kw)
        if count > 0:
            score += count * 3
            matched_terms.append(kw)
            continue

        # Layer 2: stem match (e.g. "saturate" matches "saturated", "saturation")
        stem_count = stemmed_tokens.count(kw_stem)
        if stem_count > 0:
            score += stem_count * 2
            matched_terms.append(kw)
            continue

        # Layer 3: substring match for long keywords — handles OCR artifacts like "unsat urated"
        if len(kw) >= 5 and kw in text_lower:
            score += 2
            matched_terms.append(kw)
            continue

        # Layer 4: prefix match for very long scientific words
        if len(kw) >= 8:
            prefix = kw[:5]
            if any(t.startswith(prefix) for t in tokens):
                score += 1
                matched_terms.append(kw)

    # Layer 5: exact full-phrase bonus
    if query_lower in text_lower and len(keywords) > 1:
        score += 15

    # Layer 6: consecutive-keyword bigram bonus
    kw_bigrams = set(extract_bigrams(keywords))
    text_bigrams = set(extract_bigrams(tokens))
    for bg in kw_bigrams:
        if bg in text_bigrams:
            score += 8

    return score, matched_terms


def search_pages(
    pages: List[Dict[str, Any]],
    query: str,
    top_k: int = 5,
    book_id: str = "default",
) -> List[Dict[str, Any]]:
    """Hybrid search: combines dense semantic similarity (BGE-Large + FAISS) with keyword boosting.

    Search strategy:
        1. Semantic retrieval: encode query + page texts with BAAI/bge-large-en-v1.5,
           search FAISS index for top-k most similar pages by cosine similarity.
        2. Keyword boost: for each semantic result, compute a keyword overlap bonus
           and add it to the semantic score to promote exact-match hits.
        3. Rank by combined score and return top_k.

    Falls back to keyword-only search if semantic library is unavailable.

    Args:
        pages: List of page dictionaries or PageResult objects.
        query: User's question or search terms.
        top_k: Maximum number of ranked pages to return (default 5).
        book_id: Unique book identifier for FAISS index caching.

    Returns:
        List of ranked page dicts with 'semantic_score', 'score', 'snippet'.
    """
    if not query.strip() or not pages:
        return []

    # ── 1. Semantic Retrieval ──────────────────────────────────────────────
    try:
        from src.semantic_search import semantic_search_pages
        semantic_results = semantic_search_pages(
            pages, query, top_k=top_k * 2, book_id=book_id, score_threshold=0.20
        )
    except Exception:
        semantic_results = []

    # ── 2. Fallback: keyword-only if semantic unavailable ─────────────────
    if not semantic_results:
        keywords = extract_meaningful_keywords(query)
        if not keywords:
            return []
        query_lower = query.lower().strip()
        ranked_results = []
        for page in pages:
            p_dict = page.to_dict() if hasattr(page, "to_dict") else page
            text = p_dict.get("text", "")
            text_lower = text.lower()
            tokens = clean_and_tokenize(text)
            score, matched_terms = _score_page(text, text_lower, tokens, keywords, query_lower)
            if score > 0:
                snippet = extract_snippet(text, keywords)
                ranked_results.append({
                    "page_number": p_dict.get("page_number", 1),
                    "page_type": p_dict.get("page_type", "Digital"),
                    "score": score,
                    "semantic_score": 0.0,
                    "matched_terms": matched_terms,
                    "snippet": snippet,
                    "text": text,
                    "images": p_dict.get("images", []),
                })
        ranked_results.sort(key=lambda x: x["score"], reverse=True)
        return ranked_results[:top_k]

    # ── 3. Keyword Boost on Semantic Results ──────────────────────────────
    keywords = extract_meaningful_keywords(query)
    query_lower = query.lower().strip()
    boosted = []
    for page in semantic_results:
        text = page.get("text", "")
        text_lower = text.lower()
        tokens = clean_and_tokenize(text)
        kw_score, matched_terms = _score_page(text, text_lower, tokens, keywords, query_lower)

        # Normalise keyword score to [0, 1] range with a soft cap at 30
        kw_bonus = min(kw_score, 30) / 30.0 * 0.20   # max 0.20 boost

        semantic_score = page.get("semantic_score", 0.0)
        combined = semantic_score + kw_bonus

        boosted.append({
            "page_number": page.get("page_number", 1),
            "page_type": page.get("page_type", "Digital"),
            "score": round(combined, 4),
            "semantic_score": round(semantic_score, 4),
            "matched_terms": matched_terms,
            "snippet": page.get("snippet", text[:300]),
            "text": text,
            "images": page.get("images", []),
        })

    boosted.sort(key=lambda x: x["score"], reverse=True)
    return boosted[:top_k]
