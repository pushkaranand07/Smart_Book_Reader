PYTHON ?= python

.PHONY: test eval eval-dev eval-ci lint format typecheck cov

test:
	$(PYTHON) -m pytest -q

# Full golden set on the real models (needs the cached books in data/extracted).
eval:
	$(PYTHON) scripts/eval/run_qa_eval.py --set tests/eval/qa_golden.jsonl --strict

# Threshold calibration set (never tune on the golden set).
eval-dev:
	$(PYTHON) scripts/eval/run_qa_eval.py --set tests/eval/qa_dev.jsonl --calibrate

# Small committed fixture book; runs in CI.
eval-ci:
	$(PYTHON) scripts/eval/run_qa_eval.py --set tests/eval/qa_ci.jsonl --books-dir tests/fixtures/books --strict

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

typecheck:
	$(PYTHON) -m mypy

cov:
	$(PYTHON) -m pytest -q --cov=api --cov=src/qa --cov-report=term-missing --cov-fail-under=85
