"""Unit tests for the graph retrieval adapter ([M4-04]).

Fakes for the hybrid retriever, the reranker and the clause graph -- no DB, no
models. End-to-end retrieval quality is a separate ``eval``-marked measurement.
"""

from collections.abc import Sequence
from typing import cast

import pytest

from domain.clause_classification import ClauseType
from infrastructure.rag.exclusion_co_retrieval import (
    ClauseGraph,
    ExclusionEdge,
    LinkedExclusion,
)
from infrastructure.rag.graph_retrieval_adapter import (
    GraphRetrievalAdapter,
    IndexedClause,
    build_clause_index,
)
from infrastructure.rag.retrieval_filter import RetrievalFilter


class _FakeHybrid:
    def __init__(self, clause_ids: Sequence[str]) -> None:
        self._clause_ids = list(clause_ids)
        self.k: int | None = None
        self.metadata_filter: RetrievalFilter | None = None

    def retrieve(
        self,
        question: str,
        *,
        k: int,
        metadata_filter: RetrievalFilter | None = None,
    ) -> list[str]:
        self.k = k
        self.metadata_filter = metadata_filter
        return self._clause_ids[:k]


class _FakeReranker:
    """Scores each passage by a per-clause-id lookup keyed on a marker in the text."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        return [self._scores[passage] for passage in passages]


def _index(*clause_ids: str) -> dict[str, IndexedClause]:
    return {
        cid: IndexedClause(
            embed_text=cid,
            excerpt=f"excerpt {cid}",
            document_id="doc-1",
            susep_process="15414.900000/2013-00",
            clause_type=ClauseType.COVERAGE,
        )
        for cid in clause_ids
    }


@pytest.mark.unit
def test_adapter_reranks_and_hydrates() -> None:
    adapter = GraphRetrievalAdapter(
        _FakeHybrid(["a", "b", "c"]),
        _FakeReranker({"a": 0.2, "b": 0.9, "c": 0.5}),
        _index("a", "b", "c"),
        candidate_depth=3,
    )
    hits = adapter.retrieve("q", k=2)

    assert [h.clause_id for h in hits] == ["b", "c"]
    assert [h.score for h in hits] == [pytest.approx(0.9), pytest.approx(0.5)]
    assert hits[0].excerpt == "excerpt b"
    assert hits[0].document_id == "doc-1"


@pytest.mark.unit
def test_adapter_passes_candidate_depth_and_filter_to_the_hybrid_leg() -> None:
    hybrid = _FakeHybrid(["a", "b"])
    adapter = GraphRetrievalAdapter(
        hybrid, _FakeReranker({"a": 0.7, "b": 0.6}), _index("a", "b"), candidate_depth=8
    )
    filt = RetrievalFilter(susep_process="15414.900000/2013-00")
    adapter.retrieve("q", k=5, metadata_filter=filt)

    assert hybrid.k == 8
    assert hybrid.metadata_filter is filt


@pytest.mark.unit
def test_adapter_returns_empty_when_the_hybrid_leg_finds_nothing() -> None:
    adapter = GraphRetrievalAdapter(
        _FakeHybrid([]), _FakeReranker({}), _index(), candidate_depth=3
    )
    assert adapter.retrieve("q", k=5) == []


@pytest.mark.unit
def test_adapter_drops_candidates_missing_from_the_clause_index() -> None:
    adapter = GraphRetrievalAdapter(
        _FakeHybrid(["a", "ghost"]),
        _FakeReranker({"a": 0.8, "": 0.9}),
        _index("a"),
        candidate_depth=3,
    )
    hits = adapter.retrieve("q", k=5)
    assert [h.clause_id for h in hits] == ["a"]


@pytest.mark.unit
def test_adapter_injects_a_linked_exclusion_with_zero_score() -> None:
    class _FakeGraph:
        def is_coverage(self, clause_id: str) -> bool:
            return clause_id == "cov"

        def is_exclusion(self, clause_id: str) -> bool:
            return clause_id == "exc"

        def ranked_linked_exclusions(
            self, coverage_clause_ids: Sequence[str]
        ) -> list[LinkedExclusion]:
            return [LinkedExclusion(ExclusionEdge.SAME_SECTION, 0, 0, "exc")]

    adapter = GraphRetrievalAdapter(
        _FakeHybrid(["cov", "x"]),
        _FakeReranker({"cov": 0.8, "x": 0.4}),
        {
            **_index("cov", "x"),
            "exc": IndexedClause(
                embed_text="exc",
                excerpt="excerpt exc",
                document_id="doc-1",
                susep_process="15414.900000/2013-00",
                clause_type=ClauseType.EXCLUSION,
            ),
        },
        candidate_depth=3,
        co_retrieval=cast(ClauseGraph, _FakeGraph()),
        reserved_exclusion_slots=1,
    )
    hits = adapter.retrieve("q", k=2)

    by_id = {h.clause_id: h for h in hits}
    assert "exc" in by_id
    assert by_id["exc"].score == 0.0


@pytest.mark.unit
def test_build_clause_index_rejoins_split_chunks_and_keeps_provenance() -> None:
    class _Chunk:
        def __init__(
            self,
            *,
            source_clause_ids: list[str],
            chunk_index: int,
            text: str,
            display_text: str,
        ) -> None:
            self.source_clause_ids = source_clause_ids
            self.chunk_index = chunk_index
            self.text = text
            self.display_text = display_text
            self.document_id = "doc-1"
            self.susep_process = "15414.900000/2013-00"
            self.clause_type = ClauseType.CONDITION

    chunks = [
        _Chunk(
            source_clause_ids=["c1"], chunk_index=1, text="segunda", display_text="B"
        ),
        _Chunk(
            source_clause_ids=["c1"], chunk_index=0, text="primeira", display_text="A"
        ),
    ]
    index = build_clause_index(chunks)

    assert index["c1"].embed_text == "primeira\n\nsegunda"
    assert index["c1"].excerpt == "A\n\nB"
    assert index["c1"].clause_type is ClauseType.CONDITION
