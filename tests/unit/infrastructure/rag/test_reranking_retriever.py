"""Cross-encoder reranking of a base retriever's candidates -- [M3-05]."""

from collections.abc import Sequence

import pytest

from infrastructure.rag.reranking_retriever import RerankingRetriever
from infrastructure.rag.retrieval_filter import RetrievalFilter


class FakeBase:
    """A base retriever double: returns a fixed clause-id list, records the call."""

    def __init__(self, clause_ids: list[str]) -> None:
        self._clause_ids = clause_ids
        self.seen_k: int | None = None
        self.seen_filter: RetrievalFilter | None = None

    def retrieve(
        self,
        question: str,
        *,
        k: int,
        metadata_filter: RetrievalFilter | None = None,
    ) -> list[str]:
        del question
        self.seen_k = k
        self.seen_filter = metadata_filter
        return self._clause_ids[:k]


class FakeReranker:
    """Scores each passage by a fixed lookup, records the query it was given."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores
        self.seen_query: str | None = None

    def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        self.seen_query = query
        return [self._scores[passage] for passage in passages]


_TEXT = {"a": "pa", "b": "pb", "c": "pc", "d": "pd"}


@pytest.mark.unit
def test_reorders_candidates_by_score_and_cuts_to_k() -> None:
    base = FakeBase(["a", "b", "c"])
    reranker = FakeReranker({"pa": 0.1, "pb": 0.9, "pc": 0.5})
    retriever = RerankingRetriever(base, reranker, _TEXT, candidate_depth=10)

    assert retriever.retrieve("q", k=2) == ["b", "c"]


@pytest.mark.unit
def test_asks_the_base_for_the_candidate_depth_not_k() -> None:
    base = FakeBase(["a", "b", "c", "d"])
    reranker = FakeReranker(dict.fromkeys(_TEXT.values(), 0.0))
    retriever = RerankingRetriever(base, reranker, _TEXT, candidate_depth=3)

    retriever.retrieve("q", k=1)

    assert base.seen_k == 3


@pytest.mark.unit
def test_equal_scores_keep_the_base_order() -> None:
    base = FakeBase(["c", "a", "b"])
    reranker = FakeReranker({"pa": 1.0, "pb": 1.0, "pc": 1.0})
    retriever = RerankingRetriever(base, reranker, _TEXT, candidate_depth=10)

    # All tied -> the base retriever's order survives unchanged.
    assert retriever.retrieve("q", k=3) == ["c", "a", "b"]


@pytest.mark.unit
def test_the_metadata_filter_reaches_the_base() -> None:
    base = FakeBase(["a"])
    reranker = FakeReranker({"pa": 1.0})
    retriever = RerankingRetriever(base, reranker, _TEXT)
    filt = RetrievalFilter(susep_process="P", cnpj="C")

    retriever.retrieve("q", k=1, metadata_filter=filt)

    assert base.seen_filter is filt


@pytest.mark.unit
def test_empty_candidate_set_returns_empty_without_calling_the_reranker() -> None:
    reranker = FakeReranker({})
    retriever = RerankingRetriever(FakeBase([]), reranker, _TEXT)

    assert retriever.retrieve("q", k=10) == []
    assert reranker.seen_query is None


@pytest.mark.unit
def test_non_positive_k_returns_empty() -> None:
    base = FakeBase(["a", "b"])
    retriever = RerankingRetriever(base, FakeReranker({"pa": 1.0}), _TEXT)

    assert retriever.retrieve("q", k=0) == []
    assert base.seen_k is None


@pytest.mark.unit
def test_a_candidate_missing_from_the_text_map_scores_as_empty_string() -> None:
    base = FakeBase(["a", "unknown"])
    reranker = FakeReranker({"pa": 0.2, "": 0.9})
    retriever = RerankingRetriever(base, reranker, _TEXT, candidate_depth=10)

    # "unknown" has no text -> scored as "" -> here that wins.
    assert retriever.retrieve("q", k=2) == ["unknown", "a"]
