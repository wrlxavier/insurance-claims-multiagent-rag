"""Structural exclusion co-retrieval and its reserved-budget merge -- [M3-06]."""

import pytest

from infrastructure.parsing.clause_schema import ParsedClauseRecord
from infrastructure.rag.exclusion_co_retrieval import (
    ClauseGraph,
    ExclusionCoRetrievalRetriever,
    ExclusionEdge,
    extract_cross_references,
)
from infrastructure.rag.retrieval_filter import RetrievalFilter


def _clause(
    clause_id: str,
    clause_type: str,
    *,
    parent_id: str | None = None,
    text: str = "corpo da cláusula",
    page_start: int = 1,
    page_end: int | None = None,
    bundle_section: str | None = None,
    document_id: str = "1",
) -> ParsedClauseRecord:
    """A minimal ParsedClauseRecord; ``path`` is the id's segment after the doc."""
    return ParsedClauseRecord.model_validate(
        {
            "schema_version": "v1",
            "clause_id": clause_id,
            "document_id": document_id,
            "parent_id": parent_id,
            "path": clause_id.split(":", 1)[1],
            "title": "Título",
            "text": text,
            "clause_type": clause_type,
            "type_source": "rule",
            "confidence": 1.0,
            "bundle_section": bundle_section,
            "page_start": page_start,
            "page_end": page_start if page_end is None else page_end,
            "source": "text",
            "susep_process": "123",
            "insurer": "Insurer",
            "cnpj": "C1",
            "product_line": "CASCO",
            "indemnity_regime": "VD",
            "filing_year": "2020",
        }
    )


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


# --- extract_cross_references -------------------------------------------------


@pytest.mark.unit
def test_extract_cross_references_finds_plain_and_decimal_numbers() -> None:
    assert extract_cross_references(
        "conforme a Cláusula 12 e a cláusula 4.2 acima"
    ) == {"12", "4.2"}


@pytest.mark.unit
def test_extract_cross_references_is_accent_and_case_insensitive() -> None:
    assert extract_cross_references("ver CLAUSULA 26 - PREJUÍZOS") == {"26"}


@pytest.mark.unit
def test_extract_cross_references_no_match_returns_empty_set() -> None:
    assert extract_cross_references("nenhuma referência de numeração aqui") == set()


# --- ClauseGraph edges ------------------------------------------------------


@pytest.mark.unit
def test_same_section_links_a_sibling_exclusion() -> None:
    graph = ClauseGraph(
        [
            _clause("1:root", "other"),
            _clause("1:root/1", "coverage", parent_id="1:root"),
            _clause("1:root/10", "exclusion", parent_id="1:root"),
            _clause("1:root/2", "condition", parent_id="1:root"),
        ]
    )
    links = graph.linked_exclusions("1:root/1")

    assert [link.clause_id for link in links] == ["1:root/10"]
    assert links[0].edge is ExclusionEdge.SAME_SECTION
    assert links[0].tree_distance == 2  # sibling: up to the parent and back down


@pytest.mark.unit
def test_same_section_links_a_descendant_exclusion() -> None:
    graph = ClauseGraph(
        [
            _clause("1:cov", "coverage"),
            _clause("1:cov/2", "condition", parent_id="1:cov"),
            _clause("1:cov/2/2.1", "exclusion", parent_id="1:cov/2"),
        ]
    )
    links = graph.linked_exclusions("1:cov")

    assert [link.clause_id for link in links] == ["1:cov/2/2.1"]
    assert links[0].tree_distance == 2


@pytest.mark.unit
def test_adjacent_section_links_a_nearby_exclusion_in_another_section() -> None:
    graph = ClauseGraph(
        [
            _clause("1:1", "coverage", page_start=43, page_end=43),
            _clause("1:2", "exclusion", page_start=42, page_end=42),
            _clause("1:99", "exclusion", page_start=80, page_end=80),
        ]
    )
    links = graph.linked_exclusions("1:1")

    assert [link.clause_id for link in links] == ["1:2"]
    assert links[0].edge is ExclusionEdge.ADJACENT_SECTION
    assert links[0].page_gap == 1


@pytest.mark.unit
def test_cross_reference_edge_outranks_same_section() -> None:
    graph = ClauseGraph(
        [
            _clause("1:root", "other"),
            _clause(
                "1:root/1",
                "coverage",
                parent_id="1:root",
                text="indeniza salvo o disposto na cláusula 10.2",
            ),
            _clause("1:root/9", "exclusion", parent_id="1:root"),
            _clause("1:root/x/10.2", "exclusion", parent_id="1:root"),
        ]
    )
    links = graph.linked_exclusions("1:root/1")

    # Both are same-section; the cross-referenced one is promoted and sorts first.
    assert links[0].clause_id == "1:root/x/10.2"
    assert links[0].edge is ExclusionEdge.CROSS_REFERENCE
    assert {link.clause_id for link in links} == {"1:root/9", "1:root/x/10.2"}


@pytest.mark.unit
def test_bundle_section_guard_excludes_a_different_bundle() -> None:
    graph = ClauseGraph(
        [
            _clause("1:root", "other"),
            _clause("1:root/1", "coverage", parent_id="1:root", bundle_section="MOTO"),
            _clause(
                "1:root/8", "exclusion", parent_id="1:root", bundle_section="CARGA"
            ),
            _clause("1:root/9", "exclusion", parent_id="1:root", bundle_section="MOTO"),
            _clause("1:root/10", "exclusion", parent_id="1:root"),
        ]
    )
    linked = {link.clause_id for link in graph.linked_exclusions("1:root/1")}

    assert linked == {"1:root/9", "1:root/10"}  # same bundle, or unknown


@pytest.mark.unit
def test_linked_exclusions_empty_for_non_coverage_or_unknown() -> None:
    graph = ClauseGraph(
        [
            _clause("1:e", "exclusion"),
            _clause("1:c", "condition"),
        ]
    )
    assert graph.linked_exclusions("1:e") == []
    assert graph.linked_exclusions("1:c") == []
    assert graph.linked_exclusions("1:missing") == []


@pytest.mark.unit
def test_ranking_and_dedup_are_deterministic_regardless_of_input_order() -> None:
    records = [
        _clause("1:root", "other"),
        _clause("1:root/1", "coverage", parent_id="1:root", page_start=1, page_end=1),
        _clause("1:root/10", "exclusion", parent_id="1:root", page_start=9, page_end=9),
        _clause("1:root/11", "exclusion", parent_id="1:root", page_start=5, page_end=5),
    ]
    forward = ClauseGraph(records).linked_exclusions("1:root/1")
    reverse = ClauseGraph(list(reversed(records))).linked_exclusions("1:root/1")

    # Same edge + tree distance -> ordered by page gap: /11 (gap 4) before /10 (gap 8).
    assert [link.clause_id for link in forward] == ["1:root/11", "1:root/10"]
    assert forward == reverse


@pytest.mark.unit
def test_ranked_linked_exclusions_merges_over_several_coverage_clauses() -> None:
    graph = ClauseGraph(
        [
            _clause("1:root", "other"),
            _clause("1:root/1", "coverage", parent_id="1:root"),
            _clause("1:root/2", "coverage", parent_id="1:root"),
            _clause("1:root/10", "exclusion", parent_id="1:root"),
        ]
    )
    links = graph.ranked_linked_exclusions(["1:root/1", "1:root/2"])

    assert [link.clause_id for link in links] == ["1:root/10"]  # kept once


# --- ExclusionCoRetrievalRetriever ------------------------------------------


def _graph_with_one_link() -> ClauseGraph:
    return ClauseGraph(
        [
            _clause("1:root", "other"),
            _clause("1:cov", "coverage", parent_id="1:root"),
            _clause("1:exc", "exclusion", parent_id="1:root"),
            _clause("1:n1", "condition", parent_id="1:root"),
            _clause("1:n2", "condition", parent_id="1:root"),
            _clause("1:n3", "condition", parent_id="1:root"),
        ]
    )


@pytest.mark.unit
def test_injects_a_linked_exclusion_and_evicts_the_lowest_non_exclusion() -> None:
    base = FakeBase(["1:cov", "1:n1", "1:n2", "1:n3"])
    retriever = ExclusionCoRetrievalRetriever(
        base, _graph_with_one_link(), reserved_slots=2
    )

    # k=4, one linked exclusion missing from the base -> it takes the last slot,
    # the lowest-ranked non-exclusion ("1:n3") is dropped.
    assert retriever.retrieve("q", k=4) == ["1:cov", "1:n1", "1:n2", "1:exc"]


@pytest.mark.unit
def test_appends_without_eviction_when_the_base_under_fills() -> None:
    base = FakeBase(["1:cov", "1:n1"])
    retriever = ExclusionCoRetrievalRetriever(
        base, _graph_with_one_link(), reserved_slots=2
    )

    assert retriever.retrieve("q", k=4) == ["1:cov", "1:n1", "1:exc"]


@pytest.mark.unit
def test_passthrough_when_no_coverage_clause_was_retrieved() -> None:
    base = FakeBase(["1:n1", "1:n2", "1:n3"])
    retriever = ExclusionCoRetrievalRetriever(
        base, _graph_with_one_link(), reserved_slots=2
    )

    assert retriever.retrieve("q", k=3) == ["1:n1", "1:n2", "1:n3"]


@pytest.mark.unit
def test_passthrough_when_the_linked_exclusion_is_already_in_the_ranking() -> None:
    base = FakeBase(["1:cov", "1:exc", "1:n1"])
    retriever = ExclusionCoRetrievalRetriever(
        base, _graph_with_one_link(), reserved_slots=2
    )

    assert retriever.retrieve("q", k=3) == ["1:cov", "1:exc", "1:n1"]


@pytest.mark.unit
def test_only_the_reserved_number_of_missing_exclusions_is_injected() -> None:
    graph = ClauseGraph(
        [
            _clause("1:root", "other"),
            _clause("1:cov", "coverage", parent_id="1:root"),
            _clause("1:exc1", "exclusion", parent_id="1:root", page_start=2),
            _clause("1:exc2", "exclusion", parent_id="1:root", page_start=9),
            _clause("1:n1", "condition", parent_id="1:root"),
            _clause("1:n2", "condition", parent_id="1:root"),
        ]
    )
    # Two linked exclusions missing, reserved_slots=1 -> only the best-ranked
    # one ("1:exc1", smaller page gap) is injected.
    base = FakeBase(["1:cov", "1:n1", "1:n2"])
    retriever = ExclusionCoRetrievalRetriever(base, graph, reserved_slots=1)

    assert retriever.retrieve("q", k=3) == ["1:cov", "1:n1", "1:exc1"]


@pytest.mark.unit
def test_a_missing_exclusion_is_injected_even_when_another_link_is_present() -> None:
    # The [M3-06] case that set RESERVED_EXCLUSION_SLOTS -> "inject the N best
    # linked exclusions the base missed", not "N minus however many links are
    # already there": a coverage clause can retrieve one section exclusion while
    # still missing the specific one that limits it.
    graph = ClauseGraph(
        [
            _clause("1:root", "other"),
            _clause("1:cov", "coverage", parent_id="1:root"),
            _clause("1:exc_here", "exclusion", parent_id="1:root", page_start=2),
            _clause("1:exc_missing", "exclusion", parent_id="1:root", page_start=9),
            _clause("1:n1", "condition", parent_id="1:root"),
        ]
    )
    base = FakeBase(["1:cov", "1:exc_here", "1:n1"])
    retriever = ExclusionCoRetrievalRetriever(base, graph, reserved_slots=1)

    assert retriever.retrieve("q", k=3) == ["1:cov", "1:exc_here", "1:exc_missing"]


@pytest.mark.unit
def test_injection_stops_when_only_coverage_and_exclusion_survivors_remain() -> None:
    graph = ClauseGraph(
        [
            _clause("1:root", "other"),
            _clause("1:cov", "coverage", parent_id="1:root", page_start=1, page_end=1),
            _clause("1:exc", "exclusion", parent_id="1:root", page_start=2, page_end=2),
            # A second coverage clause and an unrelated exclusion, far away.
            _clause("1:cov2", "coverage", page_start=80, page_end=80),
            _clause("1:exc_other", "exclusion", page_start=90, page_end=90),
        ]
    )
    base = FakeBase(["1:cov", "1:cov2", "1:exc_other"])
    retriever = ExclusionCoRetrievalRetriever(base, graph, reserved_slots=2)

    # "1:exc" links to "1:cov", but every base entry is coverage or exclusion --
    # none may be evicted, so the ranking is returned untouched.
    assert retriever.retrieve("q", k=3) == ["1:cov", "1:cov2", "1:exc_other"]


@pytest.mark.unit
def test_reserved_slots_zero_and_non_positive_k_are_passthrough() -> None:
    base = FakeBase(["1:cov", "1:n1"])
    assert ExclusionCoRetrievalRetriever(
        base, _graph_with_one_link(), reserved_slots=0
    ).retrieve("q", k=2) == ["1:cov", "1:n1"]
    assert (
        ExclusionCoRetrievalRetriever(
            FakeBase(["1:cov"]), _graph_with_one_link()
        ).retrieve("q", k=0)
        == []
    )


@pytest.mark.unit
def test_the_metadata_filter_reaches_the_base() -> None:
    base = FakeBase(["1:cov"])
    retriever = ExclusionCoRetrievalRetriever(base, _graph_with_one_link())
    filt = RetrievalFilter(susep_process="P", cnpj="C")

    retriever.retrieve("q", k=1, metadata_filter=filt)

    assert base.seen_filter is filt
    assert base.seen_k == 1
