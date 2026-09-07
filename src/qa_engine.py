import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from src.search import extract_meaningful_keywords, search_pages
from src.pdf_processor import find_figures_for_query, extract_meaningful_terms, is_figure_or_table
from src.normalized_document import parse_explicit_reference

BOOK_QA_SYSTEM_PROMPT = """You are an intelligent, expert Book AI tutor and assistant.
Your goal is to provide accurate, direct answers based only on the uploaded book excerpts.

Guidelines:
1. Grounding: Anchor your answers in the concepts, experiments, chemical reactions, activities, and topics present in the provided book excerpts.
2. Conceptual Inference & Reasoning: If the question asks "why", "how", or asks to solve an exercise/activity question (e.g. why magnesium ribbon is cleaned before burning, explaining underlying reactions, balancing equations), use your expert reasoning and scientific principles connected to the book's context to provide a complete, clear explanation.
3. Answer style: Match the depth to the question. Do not force a one-word or overly compressed answer. For equations and science questions, explain the relevant substances, components, symbols, coefficients, and reasoning needed to understand the answer, using only the book context and standard scientific reasoning. Keep the explanation focused and avoid unrelated background.
4. Formatting: Do not use Markdown heading markers such as # or ###. Use plain text. For chemical formulas and equations, use LaTeX math delimiters, for example: $$\\mathrm{2Mg + O_2 \\rightarrow 2MgO}$$. Use correct chemical subscripts and coefficients.
5. Do not include page numbers, source citations, or extra sections unless the user asks for them.
6. Out-of-Scope Topics: Only if the user's question is completely unrelated to anything covered in the book excerpts, state that the topic is not covered in the uploaded book.
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
    """Build a grounded answer from the most relevant extracted sentences."""
    query_terms = set(extract_meaningful_terms(query))
    candidates: List[Tuple[int, str]] = []
    for page in evidence_pages:
        text = str(page.get("text") or page.get("snippet") or "")
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
            sentence = re.sub(r"\s+", " ", sentence).strip(" -")
            if len(sentence) < 25:
                continue
            lower = sentence.lower()
            term_hits = sum(1 for term in query_terms if term in lower)
            equation_bonus = 4 if re.search(r"(?:→|->||=)", sentence) else 0
            if term_hits or equation_bonus:
                candidates.append((term_hits * 3 + equation_bonus, sentence))

    selected: List[str] = []
    for _, sentence in sorted(candidates, key=lambda item: item[0], reverse=True):
        if sentence not in selected:
            selected.append(sentence)
        if len(selected) >= 6:
            break

    if not selected:
        return "I couldn't find sufficient information in the uploaded book."
    return "\n\n".join(selected)


def _remove_page_citations(answer: str) -> str:
    """Keep source pages in structured metadata, but omit page labels from answer prose."""
    cleaned = re.sub(r"\s*\[(?:page|pages)\s+\d+(?:\s*[-,]\s*\d+)*\]", "", answer, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def _clean_answer_format(answer: str) -> str:
    """Remove stray heading markers that models sometimes emit inline."""
    cleaned = re.sub(r"(?m)^\s*#{1,6}\s*", "", answer)
    cleaned = re.sub(r"\s+#{1,6}\s+", "\n\n", cleaned)
    return cleaned.strip()


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


def _question_requests_visual(query: str) -> bool:
    """Return True only when the user asks for a visual or names one explicitly."""
    return bool(
        parse_explicit_reference(query)
        or re.search(
            r"\b(?:figure|fig\.?|diagram|image|picture|illustration|graph|chart|map|visual|shown|show|depict|draw|setup|apparatus)\b",
            query,
            flags=re.IGNORECASE,
        )
    )


def _visual_fallbacks(
    query: str,
    evidence_pages: List[Dict[str, Any]],
    selected_figures: List[Dict[str, Any]],
    all_pages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Use a relevant page crop only when extraction has no usable figure record."""
    if selected_figures:
        return selected_figures

    query_terms = set(extract_meaningful_terms(query))
    candidate_pages = evidence_pages[:2]
    if all_pages:
        best_page = max(
            all_pages,
            key=lambda page: len(query_terms & set(extract_meaningful_terms(str(page.get("text") or "")))),
        )
        best_number = best_page.get("page_number")
        candidate_pages = [
            page for page in all_pages
            if abs((page.get("page_number") or 0) - (best_number or 0)) <= 1
        ]

    for page in candidate_pages:
        text = str(page.get("text") or "")
        text_terms = set(extract_meaningful_terms(text))
        if len(query_terms & text_terms) < 2:
            continue

        page_figures = [
            figure for figure in page.get("figures", []) or []
            if isinstance(figure, dict) and is_figure_or_table(figure)
        ]
        if page_figures:
            def figure_score(figure: Dict[str, Any]) -> int:
                figure_text = " ".join(
                    str(figure.get(key) or "")
                    for key in ["figure_label", "caption", "surrounding_context", "associated_keywords"]
                ).lower()
                return sum(term in figure_text for term in query_terms)

            best_figure = max(page_figures, key=figure_score)
            if figure_score(best_figure) > 0:
                return [best_figure]

        for image_path in page.get("images", []) or []:
            if image_path and os.path.exists(image_path):
                return [{
                    "figure_id": f"page-{page.get('page_number')}-visual",
                    "figure_label": f"Page {page.get('page_number')} visual",
                    "caption": "Relevant visual extracted from the source page",
                    "page_number": page.get("page_number"),
                    "image_path": image_path,
                    "source_type": "page_image_fallback",
                }]
    return selected_figures

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
    book_cache_key = str(book_data.get("book_id") or f"book:{hash(json.dumps(book_data, sort_keys=True, default=str))}")

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

    candidate_figures: List[Dict[str, Any]] = []
    visual_requested = _question_requests_visual(query)

    if visual_requested:
        global_ranked_figures, confidence_margin, is_ambiguous = find_figures_for_query(
            page_results=pages_list,
            query=query,
            top_k=6,
            min_score=0.0,
        )
        global_ranked_figures.sort(
            key=lambda item: (
                item.get("relevance_components", {}).get("caption", 0.0),
                item.get("relevance_components", {}).get("semantic", 0.0),
                item.get("relevance_score", 0.0),
            ),
            reverse=True,
        )
        for f in global_ranked_figures:
            components = f.get("relevance_components", {})
            if is_figure_or_table(f) and (
                parse_explicit_reference(query)
                or components.get("caption", 0.0) >= 0.2
            ):
                candidate_figures.append(f)
    else:
        global_ranked_figures, confidence_margin, is_ambiguous = [], 0.0, False

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
        selected_figures = candidate_figures[:3]

    # A clear global winner should not be diluted by other figures from the same page.
    top_figure_score = global_ranked_figures[0].get("relevance_score", 0.0) if global_ranked_figures else 0.0
    if global_ranked_figures and confidence_margin >= 10.0 and top_figure_score >= 25.0:
        selected_figures = [global_ranked_figures[0]]

    # Final strict verification: only retain visuals explicitly representing Figure, Table, or Image
    selected_figures = [f for f in selected_figures if is_figure_or_table(f)]
    if visual_requested:
        selected_figures = _visual_fallbacks(query, ranked_evidence, selected_figures, pages_list)

    # Collect valid distinct image paths strictly from verified figures/tables
    collected_images = []
    for fig in selected_figures:
        img_p = fig.get("image_path")
        if img_p and os.path.exists(img_p) and img_p not in collected_images:
            collected_images.append(img_p)

    # 5. Connect to LLM for Answer Generation
    if resolved_api_key:
        prompt = format_evidence_prompt(query, ranked_evidence)
        answer_text = _clean_answer_format(_remove_page_citations(call_gemini_llm(prompt, resolved_api_key)))
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
