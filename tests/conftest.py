"""Shared test configuration.

Unit tests use the deterministic lexical QA backend so they run offline without downloading models.
The real-model path is exercised by `make eval` (scripts/eval/run_qa_eval.py).
"""

import os

import pytest

os.environ.setdefault("QA_BACKEND", "lexical")
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("GOOGLE_API_KEY", None)


@pytest.fixture(autouse=True)
def _fresh_qa_engine():
    import src.qa.backends as backends
    import src.qa.engine as engine

    backends.set_backend(None)
    engine._engine = None
    yield
    backends.set_backend(None)
    engine._engine = None
