"""Unit tests for src/qa (lexical backend; real models are covered by `make eval`)."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors as genai_errors

from src.qa import gemini
from src.qa.answer import echoes_question, is_answer_sentence
from src.qa.backends import LexicalBackend
from src.qa.chunking import chunk_pages, clean_line
from src.qa.engine import NOT_COVERED, QAEngine
from src.qa.retrieval import BookIndex, wants_equation
from src.qa.types import Chunk, ScoredChunk

PAGES = [
    {
        "page_number": 1,
        "text": "Reprint 2026-27\n1.1 Corrosion\nn Corrosion is the slow destruction of metals by air and moisture.\n"
        "Rusting of iron is a common example of corrosion and it weakens bridges and ships over time.",
    },
    {
        "page_number": 2,
        "text": "Reprint 2026-27\n1.2 Rancidity\nRancidity is the oxidation of fats and oils in food, which changes "
        "their smell and taste. Keeping food in airtight containers slows down rancidity.",
    },
    {
        "page_number": 3,
        "text": "Reprint 2026-27\n1.3 Combination\nMagnesium burns in oxygen. 2Mg + O2 → 2MgO is the balanced "
        "equation for burning magnesium ribbon in air.",
    },
    {
        "page_number": 4,
        "text": "Reprint 2026-27\nEXERCISES\n1. What is corrosion? 2. What is rancidity? 3. Why is rusting harmful?",
    },
]


def make_engine() -> QAEngine:
    return QAEngine(backend=LexicalBackend())


def test_chunking_strips_bullets_boilerplate_and_tracks_sections():
    chunks = chunk_pages(PAGES, document_id="t")
    texts = " ".join(c.text for c in chunks)
    assert "Reprint" not in texts
    assert not any(c.text.startswith("n ") for c in chunks)
    assert chunks[0].section == "1.1 Corrosion"
    assert {c.page_number for c in chunks} == {1, 2, 3, 4}
    assert next(c for c in chunks if c.page_number == 4).kind == "exercise"


def test_clean_line_collapses_repeated_phrases():
    assert clean_line("Activity 1.4 Activity 1.4 Activity 1.4 Take a beaker") == "Activity 1.4 Take a beaker"


def test_equation_boost_only_when_question_asks_for_one():
    assert wants_equation("Write the balanced equation for burning magnesium")
    assert not wants_equation("Who won the 2022 football world cup?")


def test_answerable_question_is_cited_and_does_not_echo():
    result = make_engine().answer("What is rancidity?", PAGES, book_key="b1", title="Science")
    assert result.is_sufficient
    assert result.citations and result.citations[0].page_number == 2
    assert "oxidation of fats" in result.answer
    assert not echoes_question(result.answer, "What is rancidity?")
    assert result.mode == "offline"


def test_off_topic_question_is_refused_without_citations():
    result = make_engine().answer("Who won the 2022 football world cup?", PAGES, book_key="b1")
    assert not result.is_sufficient
    assert result.answer == NOT_COVERED
    assert result.citations == []


def test_exercise_prompts_are_not_used_as_answers():
    assert not is_answer_sentence("Write the balanced chemical equation for the following reactions.", "q")
    assert not is_answer_sentence("1. What is corrosion?", "What is corrosion?")
    assert not is_answer_sentence("What is corrosion and why is it harmful to iron bridges", "What is corrosion?")


def test_index_build_uses_disk_cache(tmp_path):
    backend = LexicalBackend()
    chunks = chunk_pages(PAGES)
    first = BookIndex.build(chunks, backend, cache_dir=tmp_path)
    assert list(tmp_path.glob("*.npy"))
    second = BookIndex.build(chunks, backend, cache_dir=tmp_path)
    assert first.fingerprint == second.fingerprint
    assert (first.embeddings == second.embeddings).all()


def test_empty_question_and_empty_book():
    engine = make_engine()
    assert not engine.answer("   ", PAGES, book_key="b").is_sufficient
    assert not engine.answer("What is corrosion?", [], book_key="empty").is_sufficient


def test_engine_lru_evicts_old_books():
    engine = QAEngine(backend=LexicalBackend(), max_books=1)
    engine.index_for("a", PAGES)
    engine.index_for("b", PAGES)
    assert list(engine._indexes) == ["b"]


# ---------------------------------------------------------------- Gemini (mocked client)


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def generate_content(self, model, contents):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return SimpleNamespace(text=outcome)


def fake_client(*outcomes):
    return SimpleNamespace(models=FakeModels(outcomes))


EVIDENCE = [ScoredChunk(Chunk("c1", 2, "Rancidity is the oxidation of fats.", "1.2 Rancidity"), 5.0)]


def test_prompt_is_document_agnostic_and_injects_title_and_section():
    prompt = gemini.build_prompt("What is rancidity?", "My Book", EVIDENCE)
    assert '"My Book"' in prompt and "1.2 Rancidity" in prompt and "page 2" in prompt
    assert "magnesium" not in prompt.lower()


def test_gemini_success():
    client = fake_client("Rancidity is when fats oxidise.")
    answer = gemini.generate_answer("What is rancidity?", "Book", EVIDENCE, "key", client=client)
    assert answer.in_scope and "oxidise" in answer.text


def test_gemini_self_check_marks_out_of_scope():
    answer = gemini.generate_answer("Who won?", "Book", EVIDENCE, "key", client=fake_client("NOT_IN_BOOK"))
    assert not answer.in_scope and answer.text == ""


def test_gemini_retries_transient_errors(monkeypatch):
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda _s: None)
    client = fake_client(httpx.ConnectError("boom"), "Recovered answer.")
    answer = gemini.generate_answer("q", "Book", EVIDENCE, "key", client=client)
    assert answer.text == "Recovered answer." and client.models.calls == 2


def test_gemini_client_error_raises_unavailable():
    error = genai_errors.ClientError(400, {"error": {"message": "bad key"}})
    with pytest.raises(gemini.GeminiUnavailable):
        gemini.generate_answer("q", "Book", EVIDENCE, "key", client=fake_client(error))


def test_engine_falls_back_to_offline_when_gemini_fails(monkeypatch):
    def broken(*_a, **_k):
        raise gemini.GeminiUnavailable("timeout")

    monkeypatch.setattr("src.qa.engine.generate_answer", broken)
    result = make_engine().answer("What is rancidity?", PAGES, book_key="g1", gemini_api_key="key")
    assert result.is_sufficient and result.mode == "offline"
    assert result.diagnostics["gemini_error"] == "timeout"


def test_engine_uses_gemini_answer_and_self_check(monkeypatch):
    monkeypatch.setattr(
        "src.qa.engine.generate_answer",
        lambda *a, **k: gemini.GeminiAnswer("Rancidity is the oxidation of fats and oils in food.", True, "m"),
    )
    result = make_engine().answer("What is rancidity?", PAGES, book_key="g2", gemini_api_key="key")
    assert result.mode == "gemini" and result.answer.startswith("Rancidity is the oxidation") and result.citations

    monkeypatch.setattr("src.qa.engine.generate_answer", lambda *a, **k: gemini.GeminiAnswer("", False, "m"))
    refused = make_engine().answer("What is rancidity?", PAGES, book_key="g3", gemini_api_key="key")
    assert not refused.is_sufficient and refused.diagnostics["reason"] == "gemini_self_check"


def test_stale_section_heading_is_not_carried_through_the_book():
    pages = [
        {"page_number": 1, "text": "VIKING\nA publisher imprint line that is long enough to form a chunk of text."}
    ]
    pages += [
        {"page_number": n, "text": f"Body text on page {n} about people and their emotions in everyday life."}
        for n in range(2, 8)
    ]
    chunks = chunk_pages(pages)
    by_page = {c.page_number: c.section for c in chunks}
    assert by_page[2] == "VIKING" and by_page[3] == "VIKING"
    assert by_page[4] is None and by_page[7] is None
