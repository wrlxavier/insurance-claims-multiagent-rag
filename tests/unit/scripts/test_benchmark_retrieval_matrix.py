"""Unit tests for the [M3-08] retrieval benchmark matrix runner.

Pure functions and the timing proxy only -- the database-backed matrix is run by
hand via ``make eval-retrieval-matrix`` and every configuration it drives is
already covered by ``tests/eval/test_*_retrieval_baseline.py``.
"""

import pytest
from scripts.benchmark_retrieval_matrix import (
    MATRIX,
    MatrixConfig,
    _TimingRetriever,
    render_matrix_report,
)

from infrastructure.rag.retrieval_filter import RetrievalFilter


@pytest.mark.unit
def test_matrix_covers_the_dod_configurations_plus_the_settled_extras() -> None:
    keys = [config.key for config in MATRIX]
    assert keys == [
        "random",
        "lexical",
        "dense",
        "hybrid_rrf",
        "hybrid_rrf_rerank",
        "hybrid_rrf_rerank_co_retrieval",
        "hybrid_weighted_rerank_co_retrieval",
    ]
    # Every real configuration runs on the filtered default path; only the
    # random self-test row is unfiltered.
    for config in MATRIX:
        expected = "none" if config.key == "random" else "default"
        assert config.filter_mode == expected


@pytest.mark.unit
def test_namespace_carries_every_field_the_harness_helpers_read() -> None:
    namespace = MatrixConfig(
        "hybrid_weighted_rerank_co_retrieval",
        "hybrid weighted + rerank + co-retrieval",
        "hybrid",
        "weighted",
        True,
        True,
        "default",
    ).namespace()
    assert namespace.retriever == "hybrid"
    assert namespace.fusion == "weighted"
    assert namespace.filter_mode == "default"
    assert namespace.rerank is True
    assert namespace.co_retrieval is True
    assert isinstance(namespace.seed, int)


class _FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, bool]] = []

    def retrieve(
        self,
        question: str,
        *,
        k: int,
        metadata_filter: RetrievalFilter | None = None,
    ) -> list[str]:
        self.calls.append((question, k, metadata_filter is not None))
        return ["1:a", "1:b"][:k]


@pytest.mark.unit
def test_timing_retriever_delegates_and_records_one_sample_per_call() -> None:
    inner = _FakeRetriever()
    timed = _TimingRetriever(inner)

    assert timed.retrieve("q1", k=1) == ["1:a"]
    assert timed.retrieve(
        "q2", k=2, metadata_filter=RetrievalFilter(susep_process="p", cnpj="c")
    ) == ["1:a", "1:b"]

    assert inner.calls == [("q1", 1, False), ("q2", 2, True)]
    assert len(timed.samples_ms) == 2
    assert all(sample >= 0.0 for sample in timed.samples_ms)


def _report(key: str, label: str, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "matrix_key": key,
        "matrix_label": label,
        "config": {"retriever_name": key, "filter_mode": "default"},
        "overall": {
            "n": 117.0,
            "recall@1": 0.65,
            "recall@5": 0.87,
            "recall@10": 0.92,
            "mrr": 0.80,
            "ndcg@10": 0.82,
        },
        "by_question_type": {
            "direct_lookup": {"n": 64, "recall@10": 0.95, "mrr": 0.85},
            "coverage_with_exclusion": {"n": 19, "recall@10": 0.84, "mrr": 0.65},
            "cross_document": {"n": 16, "recall@10": 0.94, "mrr": 0.80},
            "definition": {"n": 18, "recall@10": 0.89, "mrr": 0.81},
        },
        "exclusion_clause_recall": {"recall": 1.0, "hits": 27, "total": 27},
        "foreign_document_rate": {"rate": 0.0},
        "latency_ms": {"n": 117, "p50": 12.0, "p95": 40.0, "mean": 15.0},
    }
    base.update(overrides)
    return base


@pytest.mark.unit
def test_render_matrix_report_has_every_section_and_the_best_row() -> None:
    document = {
        "run_config": {
            "golden_set_dir": "data/golden_set",
            "golden_set_question_count": 140,
            "corpus_path": "build/parsed_clauses.jsonl",
            "corpus_clause_count": 4925,
            "filter_mode": "default",
            "platform": "Linux",
            "run_at_utc": "2026-08-30T00:00:00+00:00",
        },
        "configurations": [
            _report(
                "random",
                "random (self-test)",
                overall={
                    "n": 117.0,
                    "recall@1": 0.0,
                    "recall@5": 0.0,
                    "recall@10": 0.0,
                    "mrr": 0.0,
                    "ndcg@10": 0.0,
                },
                exclusion_clause_recall={"recall": 0.0, "hits": 0, "total": 27},
                foreign_document_rate={"rate": 0.95},
                by_question_type={},
            ),
            _report("lexical", "lexical"),
            _report(
                "hybrid_rrf_rerank_co_retrieval",
                "hybrid RRF + rerank + co-retrieval",
            ),
        ],
    }

    markdown = render_matrix_report(document)

    for header in (
        "# Retrieval benchmark matrix",
        "## The matrix",
        "## By question type",
        "## Latency and cost per query",
        "## Exact configuration per run",
    ):
        assert header in markdown
    assert "hybrid RRF + rerank + co-retrieval" in markdown
    assert "100.0% (27/27)" in markdown  # exclusion recall cell
    assert "$0.00" in markdown
    assert "12.0 / 15.0" in markdown  # latency p50 / mean cell
    assert "117 scorable" in markdown
