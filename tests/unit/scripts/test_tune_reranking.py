"""Tests for the [M3-05] reranker candidate-depth sweep script."""

import pytest
from scripts.tune_reranking import (
    RERANK_CANDIDATE_DEPTHS,
    _ReplayRetriever,
    _rerank_order,
    render_markdown_report,
    summarise_latency,
)


@pytest.mark.unit
def test_candidate_depths_start_at_ten_and_ascend() -> None:
    # 10 is the pure-reorder point (Recall@10 cannot move); the one rerank pass
    # runs at max(depths), so shallower depths must be prefixes.
    assert RERANK_CANDIDATE_DEPTHS[0] == 10
    assert list(RERANK_CANDIDATE_DEPTHS) == sorted(RERANK_CANDIDATE_DEPTHS)


@pytest.mark.unit
def test_summarise_latency_handles_empty_and_populated() -> None:
    assert summarise_latency([]) == {"n": 0, "p50": 0.0, "p95": 0.0, "mean": 0.0}
    stats = summarise_latency([10.0, 20.0, 30.0, 40.0])
    assert stats["n"] == 4
    assert stats["mean"] == 25.0


@pytest.mark.unit
def test_rerank_order_sorts_by_score_desc_and_keeps_ties_in_candidate_order() -> None:
    assert _rerank_order(["a", "b", "c"], [0.1, 0.9, 0.5]) == ["b", "c", "a"]
    # All tied -> candidate order survives.
    assert _rerank_order(["c", "a", "b"], [1.0, 1.0, 1.0]) == ["c", "a", "b"]


@pytest.mark.unit
def test_replay_retriever_returns_the_precomputed_list_cut_to_k() -> None:
    retriever = _ReplayRetriever({"q": ["a", "b", "c", "d"]})
    assert retriever.retrieve("q", k=2) == ["a", "b"]
    assert retriever.retrieve("q", k=10) == ["a", "b", "c", "d"]


@pytest.mark.unit
def test_render_markdown_report_has_the_baseline_row_and_a_depth_row() -> None:
    report = {
        "config": {
            "golden_set_dir": "data/golden_set",
            "golden_set_question_count": 140,
            "corpus_path": "build/parsed_clauses.jsonl",
            "corpus_clause_count": 4925,
            "chunk_corpus_chunk_count": 4540,
            "reranker_model_id": "Alibaba-NLP/gte-multilingual-reranker-base",
            "reranker_model_revision": "8215cf04",
            "reranker_config_fingerprint": "777c0503f1073d52",
            "embedding_config_fingerprint": "7ea39a621eaee88e",
            "lexical_config_fingerprint": "ef0a2dd0c1dfb4e4",
            "hybrid_config_fingerprint": "279ed8ee0a668227",
            "scoring_device": "cuda:0",
            "reranker_device": "cpu",
            "latency_probe_questions": 20,
            "platform": "Linux",
            "candidate_depths": [10, 20, 30, 50],
            "max_depth": 50,
            "chosen_candidate_depth": 10,
            "run_at_utc": "2026-08-28T00:00:00+00:00",
        },
        "rows": [
            {
                "candidate_depth": None,
                "recall@1": 0.645,
                "recall@5": 0.836,
                "recall@10": 0.915,
                "mrr": 0.788,
                "ndcg@10": 0.803,
                "exclusion_recall": 0.925,
                "exclusion_hits": 25,
                "exclusion_total": 27,
                "latency_ms": {"n": 0, "p50": 0.0, "p95": 0.0, "mean": 0.0},
            },
            {
                "candidate_depth": 30,
                "recall@1": 0.70,
                "recall@5": 0.85,
                "recall@10": 0.92,
                "mrr": 0.84,
                "ndcg@10": 0.83,
                "exclusion_recall": 0.925,
                "exclusion_hits": 25,
                "exclusion_total": 27,
                "latency_ms": {"n": 40, "p50": 900.0, "p95": 1800.0, "mean": 1000.0},
            },
        ],
    }

    markdown = render_markdown_report(report)

    assert "hybrid RRF (no rerank)" in markdown
    assert "| 30 " in markdown
    assert "900.0 / 1800.0 / 1000.0" in markdown
    assert "777c0503f1073d52" in markdown
    assert "one rerank pass at depth 50" in markdown
    assert "`cuda:0` (scoring)" in markdown
    assert "`cpu` (latency probe)" in markdown
