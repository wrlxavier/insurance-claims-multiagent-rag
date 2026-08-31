"""Hybrid fusion of the lexical and dense legs -- [M3-04]."""

import pytest

from infrastructure.rag.hybrid_config import FusionStrategy
from infrastructure.rag.hybrid_retriever import HybridRetriever
from infrastructure.rag.retrieval_filter import RetrievalFilter


class FakeLeg:
    """A retrieval leg double: returns a fixed scored list, records the filter."""

    def __init__(self, scored: list[tuple[str, float]]) -> None:
        self._scored = scored
        self.seen_filter: RetrievalFilter | None = None
        self.seen_k: int | None = None

    def retrieve_scored(
        self,
        question: str,
        *,
        k: int,
        metadata_filter: RetrievalFilter | None = None,
    ) -> list[tuple[str, float]]:
        del question
        self.seen_filter = metadata_filter
        self.seen_k = k
        return self._scored[:k]


@pytest.mark.unit
def test_rrf_fuses_both_legs_and_truncates_to_k() -> None:
    lexical = FakeLeg([("a", 5.0), ("b", 4.0), ("c", 3.0)])
    dense = FakeLeg([("c", 0.9), ("a", 0.8), ("d", 0.7)])
    hybrid = HybridRetriever(lexical, dense, fusion=FusionStrategy.RRF)

    # `a` (rank 1 + 2) and `c` (rank 3 + 1) beat the singletons `b`, `d`.
    assert hybrid.retrieve("q", k=2) == ["a", "c"]


@pytest.mark.unit
def test_weighted_fusion_uses_the_leg_scores() -> None:
    lexical = FakeLeg([("a", 100.0), ("b", 0.0)])
    dense = FakeLeg([("b", 0.9), ("a", 0.1)])
    hybrid = HybridRetriever(lexical, dense, fusion=FusionStrategy.WEIGHTED)

    # Balanced weights, each leg normalised to [0,1]: a and b both sum to 1.0,
    # tie -> id order.
    assert hybrid.retrieve("q", k=2) == ["a", "b"]


@pytest.mark.unit
def test_the_metadata_filter_reaches_both_legs() -> None:
    lexical = FakeLeg([("a", 1.0)])
    dense = FakeLeg([("a", 1.0)])
    hybrid = HybridRetriever(lexical, dense)
    filt = RetrievalFilter(susep_process="P", cnpj="C")

    hybrid.retrieve("q", k=5, metadata_filter=filt)

    assert lexical.seen_filter is filt
    assert dense.seen_filter is filt


@pytest.mark.unit
def test_each_leg_is_asked_for_candidate_depth_not_k() -> None:
    lexical = FakeLeg([("a", 1.0)])
    dense = FakeLeg([("a", 1.0)])
    hybrid = HybridRetriever(lexical, dense)

    hybrid.retrieve("q", k=3)

    assert lexical.seen_k == 100  # hybrid_config.CANDIDATE_DEPTH
    assert dense.seen_k == 100


@pytest.mark.unit
def test_returns_fewer_than_k_without_padding() -> None:
    hybrid = HybridRetriever(FakeLeg([("a", 1.0)]), FakeLeg([]))
    assert hybrid.retrieve("q", k=10) == ["a"]


@pytest.mark.unit
def test_non_positive_k_returns_empty() -> None:
    hybrid = HybridRetriever(FakeLeg([("a", 1.0)]), FakeLeg([("a", 1.0)]))
    assert hybrid.retrieve("q", k=0) == []


@pytest.mark.unit
def test_default_strategy_is_rrf() -> None:
    hybrid = HybridRetriever(FakeLeg([("a", 1.0)]), FakeLeg([("b", 1.0)]))
    # A run that would differ under weighted (disjoint singletons, equal RRF).
    assert hybrid.retrieve("q", k=2) == ["a", "b"]
