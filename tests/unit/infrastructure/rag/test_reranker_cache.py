"""The reranker cache and its fingerprint scoping -- [M3-05].

Mirrors ``test_embedding_cache.py``.
"""

from collections.abc import Sequence
from pathlib import Path

import pytest

from infrastructure.rag.reranker_cache import CachingReranker


def _score(query: str, passage: str) -> float:
    """A deterministic, distinct pseudo-score per pair."""
    return float(len(query) * 100 + len(passage))


class CountingReranker:
    """Records every batch of passages it was asked to score."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls.append((query, list(passages)))
        return [_score(query, passage) for passage in passages]

    @property
    def call_count(self) -> int:
        return len(self.calls)


@pytest.mark.unit
def test_cache_miss_calls_inner_and_persists(tmp_path: Path) -> None:
    inner = CountingReranker()
    cache_path = tmp_path / "cache.jsonl"

    result = CachingReranker(inner, cache_path=cache_path).rerank("q", ["a", "b"])

    assert result == [_score("q", "a"), _score("q", "b")]
    assert inner.call_count == 1
    assert len(cache_path.read_text(encoding="utf-8").strip().splitlines()) == 2


@pytest.mark.unit
def test_cache_hit_skips_inner(tmp_path: Path) -> None:
    inner = CountingReranker()
    cache_path = tmp_path / "cache.jsonl"
    reranker = CachingReranker(inner, cache_path=cache_path)

    first = reranker.rerank("q", ["a", "b"])
    second = reranker.rerank("q", ["a", "b"])

    assert first == second
    assert inner.call_count == 1


@pytest.mark.unit
def test_batch_scores_only_missing_pairs_and_preserves_order(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.jsonl"
    CachingReranker(CountingReranker(), cache_path=cache_path).rerank("q", ["a", "c"])

    inner = CountingReranker()
    result = CachingReranker(inner, cache_path=cache_path).rerank(
        "q", ["a", "b", "c", "d"]
    )

    assert result == [_score("q", p) for p in ("a", "b", "c", "d")]
    assert inner.calls == [("q", ["b", "d"])]  # only the two not already cached


@pytest.mark.unit
def test_same_passage_different_query_is_a_distinct_key(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.jsonl"
    CachingReranker(CountingReranker(), cache_path=cache_path).rerank("q1", ["a"])

    inner = CountingReranker()
    CachingReranker(inner, cache_path=cache_path).rerank("q2", ["a"])

    assert inner.calls == [("q2", ["a"])]


@pytest.mark.unit
def test_duplicate_passages_in_one_call_score_once(tmp_path: Path) -> None:
    inner = CountingReranker()
    cache_path = tmp_path / "cache.jsonl"

    result = CachingReranker(inner, cache_path=cache_path).rerank("q", ["x", "x", "x"])

    assert result == [_score("q", "x")] * 3
    assert inner.calls == [("q", ["x"])]


@pytest.mark.unit
def test_hit_and_miss_counters_track_every_pair(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.jsonl"

    cold = CachingReranker(CountingReranker(), cache_path=cache_path)
    cold.rerank("q", ["a", "b", "c"])
    assert (cold.hits, cold.misses) == (0, 3)

    warm = CachingReranker(CountingReranker(), cache_path=cache_path)
    warm.rerank("q", ["a", "b", "new"])
    assert (warm.hits, warm.misses) == (2, 1)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("attr", "value"),
    [
        ("RERANKER_MODEL_ID", "other/model"),
        ("RERANKER_MODEL_REVISION", "0" * 40),
        ("RERANK_CANDIDATE_DEPTH", 999),
    ],
)
def test_config_change_invalidates_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attr: str, value: object
) -> None:
    cache_path = tmp_path / "cache.jsonl"
    warm_inner = CountingReranker()
    CachingReranker(warm_inner, cache_path=cache_path).rerank("q", ["a"])
    assert warm_inner.call_count == 1

    # Same pair, but a different reranker contract -- must NOT reuse the score.
    monkeypatch.setattr(f"infrastructure.rag.reranker_config.{attr}", value)

    cold_inner = CountingReranker()
    CachingReranker(cold_inner, cache_path=cache_path).rerank("q", ["a"])

    assert cold_inner.calls == [("q", ["a"])]
