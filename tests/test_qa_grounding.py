"""Comparison questions and grounding of LLM answers (lexical backend; real models are covered by make eval)."""

from __future__ import annotations

import pytest

from src.qa.backends import LexicalBackend
from src.qa.engine import NOT_COVERED, NOT_STATED, LLMProvider, QAEngine
from src.qa.gemini import GeminiAnswer, GeminiUnavailable
from src.qa.grounding import verify
from src.qa.query import split_comparison
from src.qa.types import Chunk, ScoredChunk

PAGES = [
    {"page_number": 1, "text": "Clean a magnesium ribbon by rubbing it with sandpaper. Burn it using a spirit lamp."},
    {
        "page_number": 10,
        "text": "1.2.3 Displacement Reaction\nIn a displacement reaction a more reactive element displaces a less "
        "reactive element from its compound. Iron displaces copper from copper sulphate solution.",
    },
    {
        "page_number": 11,
        "text": "1.2.4 Double Displacement Reaction\nIn a double displacement reaction two compounds exchange ions. "
        "Sodium sulphate and barium chloride exchange ions to form barium sulphate.",
    },
    {
        "page_number": 12,
        "text": "1.2.5 Oxidation and Reduction\nOxidation is the gain of oxygen by a substance. "
        "Reduction is the loss of oxygen by a substance during a reaction.",
    },
]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "What is the difference between a displacement and a double displacement reaction?",
            ["displacement reaction", "double displacement reaction"],
        ),
        ("Compare oxidation and reduction.", ["oxidation", "reduction"]),
        (
            "What is the difference between aerobic and anaerobic respiration?",
            ["aerobic respiration", "anaerobic respiration"],
        ),
        ("How are soaps different from detergents?", ["soaps", "detergents"]),
        ("Xylem vs phloem", ["Xylem", "phloem"]),
        ("How do metals differ from non-metals?", ["metals", "non-metals"]),
    ],
)
def test_split_comparison(question, expected):
    assert split_comparison(question) == expected


@pytest.mark.parametrize("question", ["What is corrosion?", "What do xylem and phloem transport?", "Why is it hot?"])
def test_non_comparisons_are_not_split(question):
    assert split_comparison(question) is None


def test_comparison_answer_covers_both_concepts():
    result = QAEngine(backend=LexicalBackend()).answer(
        "What is the difference between a displacement and a double displacement reaction?", PAGES, book_key="cmp"
    )
    assert result.is_sufficient
    assert "more reactive element displaces" in result.answer
    assert "exchange ions" in result.answer
    assert {c.page_number for c in result.citations} >= {10, 11}
    assert [p["sufficient"] for p in result.diagnostics["comparison"]] == [True, True]


def test_comparison_with_one_unknown_side_is_refused():
    result = QAEngine(backend=LexicalBackend()).answer(
        "What is the difference between oxidation and photosynthesis?", PAGES, book_key="cmp2"
    )
    assert not result.is_sufficient and result.answer == NOT_COVERED


EVIDENCE = [ScoredChunk(Chunk("c1", 12, PAGES[3]["text"]), 5.0)]


def test_verify_keeps_supported_and_drops_invented_sentences():
    answer = (
        "Oxidation is the gain of oxygen by a substance. "
        "Antoine Lavoisier discovered oxidation in Paris during the eighteenth century."
    )
    result = verify(answer, EVIDENCE, LexicalBackend())
    assert result.kept == ["Oxidation is the gain of oxygen by a substance."]
    assert len(result.dropped) == 1 and not result.fully_grounded


def test_verify_keeps_equations_present_in_the_passage():
    chunk = ScoredChunk(Chunk("c2", 3, "Magnesium burns: 2Mg + O2 → 2MgO in air."), 4.0)
    assert verify("2Mg + O2 → 2MgO.", [chunk], LexicalBackend()).kept


def _provider(text: str, in_scope: bool = True) -> LLMProvider:
    return LLMProvider("fake_llm", lambda q, t, e: GeminiAnswer(text, in_scope, "fake"))


def test_llm_outside_knowledge_is_removed_from_the_answer():
    engine = QAEngine(backend=LexicalBackend())
    llm = _provider(
        "Oxidation is the gain of oxygen by a substance. "
        "It was first described by Antoine Lavoisier in seventeen seventy eight."
    )
    result = engine.answer("What is oxidation?", PAGES, book_key="g1", llm=llm)
    assert result.is_sufficient and result.mode == "fake_llm"
    assert result.answer == "Oxidation is the gain of oxygen by a substance."
    assert result.diagnostics["grounding"]["dropped"] == 1


def test_llm_answer_entirely_from_outside_knowledge_becomes_not_stated():
    engine = QAEngine(backend=LexicalBackend())
    llm = _provider("The thin oxide layer on magnesium prevents it from igniting easily, so it must be removed.")
    result = engine.answer("Why should a magnesium ribbon be cleaned with sandpaper?", PAGES, book_key="g2", llm=llm)
    assert not result.is_sufficient and result.answer == NOT_STATED
    assert result.diagnostics["reason"] == "llm_answer_not_supported_by_book"


def test_llm_failure_falls_back_to_offline_and_self_check_refuses():
    engine = QAEngine(backend=LexicalBackend())

    def broken(*_a):
        raise GeminiUnavailable("timeout")

    offline = engine.answer("What is oxidation?", PAGES, book_key="g3", llm=LLMProvider("fake_llm", broken))
    assert offline.is_sufficient and offline.mode == "offline" and offline.diagnostics["fake_llm_error"] == "timeout"

    refused = engine.answer("What is oxidation?", PAGES, book_key="g4", llm=_provider("", in_scope=False))
    assert not refused.is_sufficient and refused.answer == NOT_COVERED


def test_llm_call_that_hangs_is_abandoned_and_offline_answer_used(monkeypatch):
    import threading

    import src.qa.engine as engine_module

    monkeypatch.setattr(engine_module, "LLM_DEADLINE_S", 0.2)
    release = threading.Event()

    def hangs(*_a):
        release.wait(5)
        return GeminiAnswer("late", True, "slow")

    result = QAEngine(backend=LexicalBackend()).answer(
        "What is oxidation?", PAGES, book_key="slow", llm=LLMProvider("slow_llm", hangs)
    )
    release.set()
    assert result.is_sufficient and result.mode == "offline"
    assert "no response within" in result.diagnostics["slow_llm_error"]
