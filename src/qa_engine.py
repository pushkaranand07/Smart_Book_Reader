"""Public QA entry point: grounded text answers from `src.qa` plus figure selection.

`answer_question` keeps the historical response shape (source_pages, evidence, figures, images, ...) and adds the
v2 fields (extractive_answer, citations, confidence, mode).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from src.normalized_document import parse_explicit_reference
from src.pdf_processor import extract_meaningful_terms, find_figures_for_query, is_figure_or_table
from src.qa.engine import LLMProvider, get_engine
from src.qa.types import AnswerResult

logger = logging.getLogger(__name__)


def _remove_page_citations(answer: str) -> str:
    """Keep source pages in structured metadata, but omit page labels from answer prose."""
    cleaned = re.sub(r"\s*\[(?:page|pages)\s+\d+(?:\s*[-,]\s*\d+)*\]", "", answer, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def _clean_answer_format(answer: str) -> str:
    """Remove stray heading markers that models sometimes emit inline."""
    cleaned = re.sub(r"(?m)^\s*#{1,6}\s*", "", answer)
    cleaned = re.sub(r"\s+#{1,6}\s+", "\n\n", cleaned)
    return cleaned.strip()


def _normalized_pages_for_retrieval(book_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Prefer the normalized document graph when present, but fall back to the legacy page list."""
    normalized_doc = book_data.get("normalized_document")
    if isinstance(normalized_doc, dict):
        pages: list[dict[str, Any]] = []
        for page in normalized_doc.get("pages", []) or []:
            page_text = (
                page.get("raw_text")
                or page.get("text")
                or " ".join(caption.get("text", "") for caption in page.get("captions", []))
            )
            pages.append(
                {
                    "page_number": page.get("page_number"),
                    "page_type": page.get("page_type", "Digital"),
                    "text": page_text,
                    "figures": page.get("figures", []),
                    "images": page.get("images", []),
                    "captions": page.get("captions", []),
                    "relationships": page.get("relationships", []),
                }
            )
        if pages:
            return pages
    return [p for p in book_data.get("pages", []) if isinstance(p, dict)]


def _explicit_reference_match(query: str, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return figures matching an explicitly requested reference like 'Figure 5.2'."""
    target_ref = None
    for pattern in [r"Figure\s+(\d+(?:\.\d+)?)", r"Fig\.?\s*(\d+(?:\.\d+)?)", r"Table\s+(\d+(?:\.\d+)?)"]:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            target_ref = match.group(1)
            break
    if not target_ref:
        return []

    matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    for page in pages:
        for fig in page.get("figures", []) or []:
            if not isinstance(fig, dict):
                continue
            refs = {
                str(fig.get("figure_id") or ""),
                parse_explicit_reference(str(fig.get("figure_label") or "")),
                parse_explicit_reference(str(fig.get("caption") or "")),
            }
            key = str(fig.get("figure_id") or fig.get("id") or id(fig))
            if target_ref in refs and key not in seen:
                seen.add(key)
                matches.append(fig)
    return matches


def _question_requests_visual(query: str) -> bool:
    """Return True only when the user asks for a visual or names one explicitly."""
    return bool(
        parse_explicit_reference(query)
        or re.search(
            r"\b(?:figure|fig\.?|diagram|image|picture|illustration|graph|chart|map|visual|shown|show|depict|draw|"
            r"setup|apparatus)\b",
            query,
            flags=re.IGNORECASE,
        )
    )


def _visual_fallbacks(
    query: str,
    evidence_pages: list[dict[str, Any]],
    selected_figures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pick the best figure on the evidence pages when ranking found none."""
    if selected_figures:
        return selected_figures
    query_terms = set(extract_meaningful_terms(query))
    for page in evidence_pages[:2]:
        if len(query_terms & set(extract_meaningful_terms(str(page.get("text") or "")))) < 2:
            continue
        page_figures = [f for f in page.get("figures", []) or [] if isinstance(f, dict) and is_figure_or_table(f)]

        def figure_score(figure: dict[str, Any]) -> int:
            text = " ".join(
                str(figure.get(k) or "") for k in ["figure_label", "caption", "surrounding_context"]
            ).lower()
            return sum(term in text for term in query_terms)

        if page_figures:
            best = max(page_figures, key=figure_score)
            if figure_score(best) > 0:
                return [best]
    return selected_figures


def _select_figures(
    query: str, pages: list[dict[str, Any]], evidence_pages: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], float, bool]:
    explicit = _explicit_reference_match(query, pages)
    if explicit:
        return explicit, 100.0, False
    if not _question_requests_visual(query):
        return [], 0.0, False

    ranked, margin, ambiguous = find_figures_for_query(page_results=pages, query=query, top_k=6, min_score=0.0)
    ranked.sort(
        key=lambda f: (
            f.get("relevance_components", {}).get("caption", 0.0),
            f.get("relevance_components", {}).get("semantic", 0.0),
            f.get("relevance_score", 0.0),
        ),
        reverse=True,
    )
    candidates: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for fig in ranked:
        if fig.get("figure_id") in seen:
            continue
        if is_figure_or_table(fig) and fig.get("relevance_components", {}).get("caption", 0.0) >= 0.2:
            seen.add(fig.get("figure_id"))
            candidates.append(fig)

    selected = candidates[:3]
    top_score = ranked[0].get("relevance_score", 0.0) if ranked else 0.0
    if ranked and margin >= 10.0 and top_score >= 25.0:
        selected = [ranked[0]]
    selected = [f for f in selected if is_figure_or_table(f)]
    return _visual_fallbacks(query, evidence_pages, selected), margin, ambiguous


def _book_key(book_data: dict[str, Any]) -> str:
    explicit = book_data.get("book_id") or book_data.get("source_file_hash")
    if explicit:
        return str(explicit)
    digest = hash(
        json.dumps(book_data.get("normalized_document") or book_data.get("pages"), sort_keys=True, default=str)
    )
    return f"anon:{digest}"


def answer_question(
    query: str,
    book_data: dict[str, Any],
    api_key: str | None = None,
    top_k: int = 5,
    llm: LLMProvider | None = None,
) -> dict[str, Any]:
    """Answer a question about a processed book. An optional LLM (Gemini or an injected provider) refines answers."""
    pages = _normalized_pages_for_retrieval(book_data)
    empty: dict[str, Any] = {
        "query": query,
        "answer": "Please upload a book and ask a question.",
        "extractive_answer": None,
        "source_pages": [],
        "citations": [],
        "evidence": [],
        "images": [],
        "figures": [],
        "confidence": 0.0,
        "confidence_margin": 0.0,
        "is_ambiguous": False,
        "is_sufficient": False,
        "mode": "offline",
    }
    if not query.strip() or not pages:
        return empty

    figure_ids_by_page = {
        int(p.get("page_number") or 0): [str(f.get("figure_id")) for f in p.get("figures", []) if isinstance(f, dict)]
        for p in pages
    }
    title = str(book_data.get("filename") or (book_data.get("normalized_document") or {}).get("filename") or "")
    resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    result: AnswerResult = get_engine().answer(
        query,
        pages,
        book_key=_book_key(book_data),
        title=title,
        gemini_api_key=resolved_key,
        llm=llm,
        figure_ids_by_page=figure_ids_by_page,
    )

    page_by_number = {p.get("page_number"): p for p in pages}
    cited_pages = list(dict.fromkeys(c.page_number for c in result.citations))
    evidence_pages = [{**page_by_number[n], "score": None} for n in cited_pages if n in page_by_number]
    figures, margin, ambiguous = _select_figures(query, pages, evidence_pages)

    answer_text = _clean_answer_format(_remove_page_citations(result.answer))
    is_sufficient = result.is_sufficient
    explicit = _explicit_reference_match(query, pages)
    if explicit:
        # The user named a figure that exists: answer from it even if the body text is thin.
        cited_pages = list(dict.fromkeys(int(f.get("page_number") or 0) for f in explicit))
        evidence_pages = [{**page_by_number[n], "score": None} for n in cited_pages if n in page_by_number]
        if not is_sufficient:
            is_sufficient = True
            caption = next((str(f.get("caption")) for f in explicit if f.get("caption")), "")
            label = str(explicit[0].get("figure_label") or f"Figure {explicit[0].get('figure_id')}")
            answer_text = f"{label}: {caption}" if caption else label
    elif not is_sufficient:
        figures = []

    images = [f["image_path"] for f in figures if f.get("image_path") and os.path.exists(str(f["image_path"]))]
    payload = result.to_dict()
    return {
        **empty,
        **payload,
        "query": query,
        "answer": answer_text if is_sufficient else result.answer,
        "is_sufficient": is_sufficient,
        "source_pages": cited_pages[:top_k] if is_sufficient else [],
        "evidence": evidence_pages[:top_k] if is_sufficient else [],
        "figures": figures,
        "images": list(dict.fromkeys(images)),
        "confidence_margin": margin,
        "is_ambiguous": ambiguous,
    }
