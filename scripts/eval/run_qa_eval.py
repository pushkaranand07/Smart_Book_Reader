"""Evaluate the QA engine against a golden JSONL set.

Usage:
    python scripts/eval/run_qa_eval.py                                  # golden set, real models
    python scripts/eval/run_qa_eval.py --set tests/eval/qa_dev.jsonl --calibrate
    python scripts/eval/run_qa_eval.py --set tests/eval/qa_ci.jsonl --books-dir tests/fixtures/books

Each JSONL row: {"id", "book", "question", "answerable", "expected_pages"}.
Books are read from <books-dir>/<book>_extracted.json (the pipeline's cached output).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.qa.answer import echoes_question
from src.qa.engine import QAEngine

TARGETS = {"recall@5": 0.90, "unanswerable_acc": 0.90, "false_refusal": 0.10, "echo": 0, "citation_cov": 1.0}


def load_pages(books_dir: Path, book: str) -> list[dict[str, Any]]:
    data = json.loads((books_dir / f"{book}_extracted.json").read_text(encoding="utf-8"))
    normalized = data.get("normalized_document") or {}
    return list(normalized.get("pages") or data.get("pages") or [])


def run(rows: list[dict[str, Any]], books_dir: Path, cache_dir: Path | None) -> list[dict[str, Any]]:
    engine = QAEngine(cache_dir=cache_dir, max_books=32)
    pages_by_book: dict[str, list[dict[str, Any]]] = {}
    results = []
    for row in rows:
        book = row["book"]
        if book not in pages_by_book:
            pages_by_book[book] = load_pages(books_dir, book)
            # Index build is a one-off cost paid by the worker at upload time; keep it out of answer latency.
            engine.index_for(book, pages_by_book[book])
            if not results:  # load reranker/reader weights once, outside the timed region
                engine.answer("warm up", pages_by_book[book], book_key=book)
        start = time.perf_counter()
        result = engine.answer(row["question"], pages_by_book[book], book_key=book, title=book)
        elapsed = time.perf_counter() - start
        ranked = engine.retrieve(row["question"], engine.index_for(book, pages_by_book[book]))
        top_pages: list[int] = []
        for item in ranked:
            if item.chunk.page_number not in top_pages:
                top_pages.append(item.chunk.page_number)
            if len(top_pages) == 5:
                break
        results.append(
            {
                **row,
                "is_sufficient": result.is_sufficient,
                "answer": result.answer,
                "extractive_answer": result.extractive_answer,
                "citations": [c.page_number for c in result.citations],
                "top_pages": top_pages,
                "top_rerank": result.diagnostics.get("top_rerank"),
                "span_margin": result.diagnostics.get("span_margin"),
                "latency_s": round(elapsed, 3),
            }
        )
    return results


def metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]
    recall = sum(bool(set(r["expected_pages"]) & set(r["top_pages"])) for r in answerable) / max(1, len(answerable))
    unans_acc = sum(not r["is_sufficient"] for r in unanswerable) / max(1, len(unanswerable))
    false_refusal = sum(not r["is_sufficient"] for r in answerable) / max(1, len(answerable))
    answered = [r for r in results if r["is_sufficient"]]
    echo = sum(
        echoes_question(r["answer"], r["question"]) or r["question"].lower().rstrip("?") in r["answer"].lower()
        for r in answered
    )
    citation_cov = sum(bool(r["citations"]) for r in answered) / max(1, len(answered))
    cited_correct = sum(bool(set(r["expected_pages"]) & set(r["citations"])) for r in answerable if r["is_sufficient"])
    latencies = sorted(r["latency_s"] for r in results)
    return {
        "questions": len(results),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "recall@5": recall,
        "unanswerable_acc": unans_acc,
        "false_refusal": false_refusal,
        "echo": echo,
        "citation_cov": citation_cov,
        "citation_page_precision": cited_correct / max(1, sum(r["is_sufficient"] for r in answerable)),
        "p50_latency_s": latencies[len(latencies) // 2] if latencies else 0.0,
        "p95_latency_s": latencies[int(len(latencies) * 0.95) - 1] if latencies else 0.0,
    }


def print_table(m: dict[str, float]) -> bool:
    print("\n| Metric | Value | Target | Pass |\n| --- | --- | --- | --- |")
    ok = True
    for key, value in m.items():
        target = TARGETS.get(key)
        shown = f"{value:.3f}" if isinstance(value, float) else str(value)
        if target is None:
            print(f"| {key} | {shown} | | |")
            continue
        passed = value <= target if key in {"false_refusal", "echo"} else value >= target
        ok &= passed
        op = "<=" if key in {"false_refusal", "echo"} else ">="
        print(f"| {key} | {shown} | {op} {target} | {'yes' if passed else 'NO'} |")
    return ok


def calibrate(results: list[dict[str, Any]]) -> None:
    """Grid-search the sufficiency thresholds on this set (use the dev set, never the golden set)."""
    best = None
    for rerank_t, strong_t, margin_t in itertools.product(
        [x / 2 for x in range(-8, 9)], [x / 2 for x in range(17)], [x / 2 for x in range(-10, 11)]
    ):
        if strong_t < rerank_t:
            continue
        correct = 0
        for r in results:
            top, margin = r["top_rerank"] or -99.0, r["span_margin"] if r["span_margin"] is not None else -99.0
            pred = top >= rerank_t and (margin >= margin_t or top >= strong_t)
            correct += pred == r["answerable"]
        acc = correct / len(results)
        # Prefer the most permissive thresholds among ties (fewer false refusals on unseen data).
        key = (acc, -rerank_t, -margin_t, -strong_t)
        if best is None or key > best[0]:
            best = (key, rerank_t, strong_t, margin_t)
    assert best is not None
    print(
        f"\nCalibrated: accuracy={best[0][0]:.3f} QA_RERANK_THRESHOLD={best[1]} QA_STRONG_RERANK={best[2]} "
        f"QA_SPAN_MARGIN_THRESHOLD={best[3]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default=str(ROOT / "tests" / "eval" / "qa_golden.jsonl"))
    parser.add_argument("--books-dir", default=str(ROOT / "data" / "extracted"))
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / ".cache" / "qa_index"))
    parser.add_argument("--dump", help="write per-question results to this JSON file")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--strict", action="store_true", help="exit 1 if any target is missed")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.set).read_text(encoding="utf-8").splitlines() if line.strip()]
    results = run(rows, Path(args.books_dir), Path(args.cache_dir) if args.cache_dir else None)
    if args.dump:
        Path(args.dump).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    m = metrics(results)
    ok = print_table(m)
    if args.calibrate:
        calibrate(results)
    if args.strict and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
