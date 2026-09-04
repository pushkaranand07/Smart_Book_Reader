import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from src.search import extract_meaningful_keywords, search_pages
from src.pdf_processor import find_figures_for_query, extract_meaningful_terms, is_figure_or_table
from src.normalized_document import parse_explicit_reference

BOOK_QA_SYSTEM_PROMPT = """You are an intelligent, expert Book AI tutor and assistant.
Your goal is to provide clear, thorough, accurate, and insightful answers to the user's questions based on the uploaded book excerpts.

Guidelines:
1. Grounding: Anchor your answers in the concepts, experiments, chemical reactions, activities, and topics present in the provided book excerpts.
2. Conceptual Inference & Reasoning: If the question asks "why", "how", or asks to solve an exercise/activity question (e.g. why magnesium ribbon is cleaned before burning, explaining underlying reactions, balancing equations), use your expert reasoning and scientific principles connected to the book's context to provide a complete, clear explanation.
3. Answer style: Explain the topic in a clear, well-structured way using a short definition, meaningful headings, concise paragraphs, and bullet lists where useful. Do not include page numbers or source citations in the answer text.
4. Out-of-Scope Topics: Only if the user's question is completely unrelated to anything covered in the book excerpts, state that the topic is not covered in the uploaded book.
"""


def format_evidence_prompt(query: str, evidence_pages: List[Dict[str, Any]]) -> str:
    """Format the retrieved evidence into a clean prompt for the LLM."""
    excerpts = []
    for item in evidence_pages:
        p_num = item["page_number"]
        p_type = item.get("page_type", "Digital")
        text = item.get("text", "").strip()
        excerpts.append(f"=== BOOK EVIDENCE: PAGE {p_num} ({p_type}) ===\n{text}")

    evidence_block = "\n\n".join(excerpts)
    prompt = f"""{BOOK_QA_SYSTEM_PROMPT}

BOOK EXCERPTS:
{evidence_block}

USER QUESTION:
{query}

ANSWER:"""
    return prompt


# Preferred model names — tried in order, first successful response wins
_PREFERRED_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
]


def _discover_models(client) -> list:
    """Dynamically list models that support generateContent from the API."""
    try:
        available = []
        for m in client.models.list():
            actions = getattr(m, "supported_actions", None) or []
            name = getattr(m, "name", "") or ""
            short = name.replace("models/", "")
            if "generateContent" in actions and "embedding" not in short:
                available.append(short)
        return available
    except Exception:
        return []


def llm_select_relevant_figures(
    query: str,
    candidate_figures: List[Dict[str, Any]],
    api_key: str,
    max_figures: int = 3,
) -> List[Dict[str, Any]]:
    """Use Gemini LLM to intelligently select and rank figures that illustrate the user's question."""
    if not candidate_figures or not api_key:
        return candidate_figures[:max_figures]

    # Deduplicate candidates by (page_number, figure_id, image_path)
    unique_candidates: List[Dict[str, Any]] = []
    seen = set()
    for f in candidate_figures:
        key = (f.get("page_number"), f.get("figure_id"), f.get("image_path"))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(f)

    if len(unique_candidates) <= 1:
        return unique_candidates

    try:
        from google import genai
        from google.genai import types as genai_types

        client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(api_version="v1"),
        )

        options_text = []
        for i, fig in enumerate(unique_candidates[:8]):
            fid = fig.get("figure_id", f"fig_{i+1}")
            flabel = fig.get("figure_label", f"Figure {fid}")
            p_num = fig.get("page_number", "?")
            cap = fig.get("caption", "No caption")
            labels = ", ".join(fig.get("labels_inside", [])[:6]) or "None"
            ctx = (fig.get("surrounding_context") or fig.get("context") or "")[:180]
            options_text.append(
                f"- Candidate [{i+1}]: ID={fid}, Page={p_num}, Label={flabel}\n"
                f"  Caption: {cap}\n"
                f"  Internal Labels: {labels}\n"
                f"  Context: {ctx}"
            )

        prompt = f"""You are an expert textbook visual selector.
User Question / Topic: "{query}"

Here are candidate figures and diagrams extracted from the textbook:
{chr(10).join(options_text)}

Task: Identify which candidate figure(s) directly illustrate, support, or explain the mechanism/activity/concept asked in the question.
If multiple candidates are relevant, list their candidate numbers in order of relevance (e.g., 1, 3).
If NONE of the figures are relevant to the user's question, respond with: NONE.

Respond ONLY in the format of comma-separated candidate numbers (e.g. 1, 2) or NONE."""

        for model_name in _PREFERRED_MODELS:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                if response and response.text:
                    resp_clean = response.text.strip().upper()
                    if "NONE" in resp_clean and not re.search(r'\b[1-8]\b', resp_clean):
                        return []

                    matched_indices = []
                    nums = re.findall(r'\b([1-8])\b', resp_clean)
                    for n in nums:
                        idx = int(n) - 1
                        if 0 <= idx < len(unique_candidates) and idx not in matched_indices:
                            matched_indices.append(idx)

                    if matched_indices:
                        selected = [unique_candidates[i] for i in matched_indices]
                        return selected[:max_figures]
                    break
            except Exception:
                continue
    except Exception:
        pass

    return unique_candidates[:max_figures]


def contrastive_disambiguate_figures(
    query: str,
    candidate_figures: List[Dict[str, Any]],
    api_key: str,
) -> Optional[str]:
    """Micro-LLM Contrastive Reranker to pick the most accurate figure when scores are close."""
    if len(candidate_figures) < 2:
        return None

    try:
        from google import genai
        from google.genai import types as genai_types

        client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(api_version="v1"),
        )

        options_text = []
        for i, fig in enumerate(candidate_figures[:3]):
            fid = fig.get("figure_id", f"candidate_{i+1}")
            flabel = fig.get("figure_label", f"Figure {fid}")
            cap = fig.get("caption", "No caption")
            labels = ", ".join(fig.get("labels_inside", [])[:6]) or "None"
            ctx = (fig.get("surrounding_context") or fig.get("context") or "")[:200]
            options_text.append(
                f"- Candidate ID: {fid}\n  Label: {flabel}\n  Caption: {cap}\n  Internal Labels: {labels}\n  Context: {ctx}"
            )

        prompt = f"""You are an expert textbook visual selector.
User Question/Topic: "{query}"

We have multiple candidate figures from the textbook:
{chr(10).join(options_text)}

Select the SINGLE candidate ID whose visual directly explains or illustrates the specific mechanism/process or subject asked.
Respond ONLY with the candidate ID (e.g., 5.11). Do not include any other words or punctuation."""

        for model_name in _PREFERRED_MODELS:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                if response and response.text:
                    selected_raw = response.text.strip().lower()
                    selected_clean = re.sub(r'[^a-z0-9\.]', ' ', selected_raw)
                    tokens = selected_clean.split()
                    for fig in candidate_figures:
                        fig_id = fig.get("figure_id", "").lower()
                        if fig_id in tokens or any(t == fig_id for t in tokens):
                            return fig.get("figure_id")
                    break
            except Exception:
                continue
    except Exception:
        pass
    return None


def call_gemini_llm(prompt: str, api_key: str) -> str:
    """Call Google Gemini LLM using the google-genai SDK (v1 API)."""
    try:
        from google import genai
        from google.genai import types as genai_types

        client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(api_version="v1"),
        )

        # 1. Try preferred models in order
        last_error = None
        for model_name in _PREFERRED_MODELS:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                last_error = e
                continue

        # 2. Fallback: auto-discover what models this key actually has
        discovered = _discover_models(client)
        for model_name in discovered:
            if model_name in _PREFERRED_MODELS:
                continue
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                last_error = e
                continue

        if last_error:
            return (
                f"LLM Error — no working model found for your API key.\n"
                f"Last error: {last_error}\n"
                f"Available models on your key: {discovered or 'could not list'}"
            )

        return "I couldn't find sufficient information in the uploaded book."
    except Exception as e:
        return f"Error connecting to LLM: {e}"


def synthesize_offline_evidence(query: str, evidence_pages: List[Dict[str, Any]]) -> str:
    """Synthesize an extractive answer when running in offline mode without an API key."""
    snippets = []
    for p in evidence_pages:
        snippet = p.get("snippet", "")
        snippets.append(f"> {snippet}")

    synthesis = "\n\n".join(snippets)
    return (
        f"**Extracted Evidence from Book:**\n\n{synthesis}\n\n"
        f"*(Add a Gemini API key for complete natural-language synthesis.)*"
    )


def _remove_page_citations(answer: str) -> str:
    """Keep source pages in structured metadata, but omit page labels from answer prose."""
    cleaned = re.sub(r"\s*\[(?:page|pages)\s+\d+(?:\s*[-,]\s*\d+)*\]", "", answer, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def _normalized_pages_for_retrieval(book_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Prefer the normalized document graph when present, but fall back to the legacy page list."""
    normalized_doc = book_data.get("normalized_document")
    if isinstance(normalized_doc, dict):
        pages: List[Dict[str, Any]] = []
        for page in normalized_doc.get("pages", []) or []:
            page_text = page.get("raw_text") or " ".join(
                [caption.get("text", "") for caption in page.get("captions", [])]
            )
            pages.append(
                {
                    "page_number": page.get("page_number"),
                    "page_type": page.get("page_type", "Digital"),
                    "text": page_text,
                    "figures": page.get("figures", []),
                    "captions": page.get("captions", []),
                    "relationships": page.get("relationships", []),
                }
            )
        if pages:
            return pages

    return book_data.get("pages", [])


def _explicit_reference_match(query: str, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prefer a single explicitly requested figure reference like 'Figure 5.2'."""
    target_ref = None
    for pattern in [r"Figure\s+(\d+(?:\.\d+)?)", r"Fig\.?\s*(\d+(?:\.\d+)?)", r"Table\s+(\d+(?:\.\d+)?)"]:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            target_ref = match.group(1)
            break

    if not target_ref:
        return []

    matches: List[Dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        for fig in page.get("figures", []) or []:
            if not isinstance(fig, dict):
                continue
            fig_label = str(fig.get("figure_label") or "")
            fig_id = str(fig.get("figure_id") or "")
            caption = str(fig.get("caption") or "")
            if target_ref in {fig_id, parse_explicit_reference(fig_label), parse_explicit_reference(caption)}:
                matches.append(fig)

    seen = set()
    deduped: List[Dict[str, Any]] = []
    for fig in matches:
        key = fig.get("figure_id") or fig.get("id")
        if key and key in seen:
            continue
        seen.add(key)
        deduped.append(fig)
    return deduped


def answer_question(
    query: str,
    book_data: Dict[str, Any],
    api_key: Optional[str] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Retrieve evidence, score & disambiguate figures, call LLM with strict prompt, and return structured answer."""
    pages_list = _normalized_pages_for_retrieval(book_data)
    if not query.strip() or not pages_list:
        return {
            "query": query,
            "answer": "Please upload a book and ask a question.",
            "source_pages": [],
            "evidence": [],
            "images": [],
            "figures": [],
            "is_sufficient": False,
        }

    # Use a per-book cache key so the semantic index does not leak between different books/tests.
    book_cache_key = f"book:{hash(json.dumps(book_data, sort_keys=True, default=str))}"

    # 1. Search & Rank (Retrieval)
    ranked_evidence = search_pages(pages_list, query, top_k=top_k, book_id=book_cache_key)

    # 2. Handle insufficient evidence
    if not ranked_evidence:
        return {
            "query": query,
            "answer": "I couldn't find sufficient information in the uploaded book.",
            "source_pages": [],
            "evidence": [],
            "images": [],
            "figures": [],
            "is_sufficient": False,
        }

    source_pages = [p["page_number"] for p in ranked_evidence]
    resolved_api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    # 3. Gather candidate figures from top evidence pages + global multi-signal search
    candidate_figures: List[Dict[str, Any]] = []

    # a) High-priority candidates from the retrieved evidence pages
    for ev in ranked_evidence:
        p_num = ev.get("page_number")
        for p in pages_list:
            p_curr_num = p.get("page_number") if isinstance(p, dict) else getattr(p, "page_number", None)
            if p_curr_num == p_num:
                p_figs = p.get("figures", []) if isinstance(p, dict) else getattr(p, "figures", [])
                for f in p_figs:
                    f_dict = dict(f) if isinstance(f, dict) else (f.to_dict() if hasattr(f, "to_dict") else f)
                    if is_figure_or_table(f_dict):
                        candidate_figures.append(f_dict)

    # b) Global multi-signal figure search across entire book
    global_ranked_figures, confidence_margin, is_ambiguous = find_figures_for_query(
        page_results=pages_list,
        query=query,
        top_k=6,
        min_score=15.0,
    )
    for f in global_ranked_figures:
        if is_figure_or_table(f):
            candidate_figures.append(f)

    # Deduplicate candidate figures
    deduped_candidates: List[Dict[str, Any]] = []
    seen_fids = set()
    for f in candidate_figures:
        fid = f.get("figure_id")
        if fid not in seen_fids:
            seen_fids.add(fid)
            deduped_candidates.append(f)
    candidate_figures = deduped_candidates

    explicit_matches = _explicit_reference_match(query, pages_list)
    if explicit_matches:
        candidate_figures = explicit_matches
        explicit_page_numbers = {
            f.get("page_number")
            for f in explicit_matches
            if isinstance(f, dict) and f.get("page_number") is not None
        }
        if explicit_page_numbers:
            ranked_evidence = [
                ev for ev in ranked_evidence if ev.get("page_number") in explicit_page_numbers
            ]
            if not ranked_evidence:
                for page in pages_list:
                    if not isinstance(page, dict):
                        continue
                    if page.get("page_number") in explicit_page_numbers:
                        snippet = page.get("text") or " ".join(
                            caption.get("text", "") for caption in page.get("captions", [])
                        )
                        ranked_evidence = [{
                            "page_number": page.get("page_number"),
                            "page_type": page.get("page_type", "Digital"),
                            "score": 100.0,
                            "semantic_score": 1.0,
                            "matched_terms": ["figure"],
                            "snippet": snippet[:300],
                            "text": snippet,
                            "images": page.get("images", []),
                        }]
                        break

    # 4. LLM Visual Selection & Ranking (or Heuristic Fallback)
    if resolved_api_key and candidate_figures:
        selected_figures = llm_select_relevant_figures(
            query=query,
            candidate_figures=candidate_figures,
            api_key=resolved_api_key,
            max_figures=3,
        )
        if not selected_figures and candidate_figures:
            selected_figures = candidate_figures[:2]
    else:
        selected_figures = candidate_figures[:3] if candidate_figures else [f for f in global_ranked_figures if is_figure_or_table(f)][:3]

    # Final strict verification: only retain visuals explicitly representing Figure, Table, or Image
    selected_figures = [f for f in selected_figures if is_figure_or_table(f)]

    # Collect valid distinct image paths strictly from verified figures/tables
    collected_images = []
    for fig in selected_figures:
        img_p = fig.get("image_path")
        if img_p and os.path.exists(img_p) and img_p not in collected_images:
            collected_images.append(img_p)

    # 5. Connect to LLM for Answer Generation
    if resolved_api_key:
        prompt = format_evidence_prompt(query, ranked_evidence)
        answer_text = _remove_page_citations(call_gemini_llm(prompt, resolved_api_key))
    else:
        answer_text = synthesize_offline_evidence(query, ranked_evidence)

    return {
        "query": query,
        "answer": answer_text,
        "source_pages": source_pages,
        "evidence": ranked_evidence,
        "images": collected_images,
        "figures": selected_figures,
        "confidence_margin": confidence_margin,
        "is_ambiguous": is_ambiguous,
        "is_sufficient": True,
    }
