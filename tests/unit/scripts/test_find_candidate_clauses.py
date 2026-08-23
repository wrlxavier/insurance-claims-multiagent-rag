"""Tests for the deterministic candidate-clause search script."""

import pytest
from scripts.find_candidate_clauses import (
    ClauseNotFoundError,
    extract_cross_references,
    find_candidates,
    find_document_order_neighbours,
)

from infrastructure.parsing.clause_schema import ParsedClauseRecord


def make_record(**overrides: object) -> ParsedClauseRecord:
    """Build a valid ParsedClauseRecord, overridable per test."""
    fields: dict[str, object] = {
        "schema_version": "v1",
        "clause_id": "1:cond-gerais/7",
        "document_id": "1",
        "parent_id": "1:cond-gerais",
        "path": "cond-gerais/7",
        "title": "7. INFORMAÇÕES PRESTADAS NO MOMENTO DA CONTRATAÇÃO",
        "text": "conforme Cláusula 12 – Perda de Direitos, alínea a",
        "clause_type": "other",
        "type_source": "rule",
        "confidence": None,
        "bundle_section": "CONDIÇÕES CONTRATUAIS GERAIS",
        "page_start": 1,
        "page_end": 1,
        "source": "text",
        "boundary_source": None,
        "susep_process": "123",
        "insurer": "Porto Seguro",
        "cnpj": "00.000.000/0001-00",
        "product_line": "auto",
        "indemnity_regime": "indenização",
        "filing_year": "2024",
    }
    fields.update(overrides)
    return ParsedClauseRecord.model_validate(fields)


@pytest.mark.unit
def test_extract_cross_references_finds_number() -> None:
    assert extract_cross_references("conforme Cláusula 12 – Perda de Direitos") == {
        "12"
    }


@pytest.mark.unit
def test_extract_cross_references_finds_decimal_number() -> None:
    assert extract_cross_references("ver cláusula 4.2 acima") == {"4.2"}


@pytest.mark.unit
def test_extract_cross_references_no_match_returns_empty_set() -> None:
    assert extract_cross_references("nenhuma referência aqui") == set()


@pytest.mark.unit
def test_find_candidates_matches_shared_parent() -> None:
    target = make_record(
        clause_id="1:cond-gerais/7", parent_id="1:cond-gerais", text=""
    )
    sibling = make_record(
        clause_id="1:cond-gerais/8",
        path="cond-gerais/8",
        parent_id="1:cond-gerais",
        bundle_section="OUTRA SEÇÃO",
        title="8. OUTRA",
    )
    candidates = find_candidates([target, sibling], target.clause_id)
    assert [c.clause_id for c in candidates] == ["1:cond-gerais/8"]
    assert candidates[0].reasons == ("shared_parent",)


@pytest.mark.unit
def test_find_candidates_matches_bundle_section() -> None:
    target = make_record(
        clause_id="1:cond-gerais/7",
        parent_id="1:cond-gerais",
        bundle_section="SEÇÃO A",
        text="",
    )
    other = make_record(
        clause_id="1:cond-gerais/9",
        path="cond-gerais/9",
        parent_id="1:cond-gerais/other",
        bundle_section="SEÇÃO A",
        title="9. OUTRA",
    )
    candidates = find_candidates([target, other], target.clause_id)
    assert [c.clause_id for c in candidates] == ["1:cond-gerais/9"]
    assert candidates[0].reasons == ("matching_bundle_section",)


@pytest.mark.unit
def test_find_candidates_matches_textual_cross_reference() -> None:
    target = make_record(
        clause_id="1:cond-gerais/7",
        parent_id="1:cond-gerais",
        bundle_section="SEÇÃO A",
        text="conforme Cláusula 12 – Perda de Direitos",
    )
    referenced = make_record(
        clause_id="1:cond-gerais/12",
        path="cond-gerais/12",
        parent_id="1:cond-gerais/other",
        bundle_section="OUTRA SEÇÃO",
        title="12. PERDA DE DIREITOS",
    )
    candidates = find_candidates([target, referenced], target.clause_id)
    assert [c.clause_id for c in candidates] == ["1:cond-gerais/12"]
    assert candidates[0].reasons == ("cross_reference:12",)


@pytest.mark.unit
def test_find_candidates_merges_reasons_and_sorts_first() -> None:
    target = make_record(
        clause_id="1:cond-gerais/7",
        parent_id="1:cond-gerais",
        bundle_section="SEÇÃO A",
        text="conforme Cláusula 8",
    )
    multi_signal = make_record(
        clause_id="1:cond-gerais/8",
        path="cond-gerais/8",
        parent_id="1:cond-gerais",
        bundle_section="OUTRA SEÇÃO",
        title="8. OUTRA",
    )
    single_signal = make_record(
        clause_id="1:cond-gerais/9",
        path="cond-gerais/9",
        parent_id="1:cond-gerais",
        bundle_section="OUTRA SEÇÃO",
        title="9. OUTRA",
    )
    candidates = find_candidates(
        [target, multi_signal, single_signal], target.clause_id
    )
    assert [c.clause_id for c in candidates] == ["1:cond-gerais/8", "1:cond-gerais/9"]
    assert set(candidates[0].reasons) == {"shared_parent", "cross_reference:8"}


@pytest.mark.unit
def test_find_candidates_excludes_other_documents() -> None:
    target = make_record(
        clause_id="1:cond-gerais/7", parent_id="1:cond-gerais", text=""
    )
    other_doc = make_record(
        clause_id="2:cond-gerais/8",
        document_id="2",
        path="cond-gerais/8",
        parent_id="1:cond-gerais",
        title="8. OUTRA",
    )
    assert find_candidates([target, other_doc], target.clause_id) == []


@pytest.mark.unit
def test_find_candidates_excludes_query_clause_itself() -> None:
    target = make_record(
        clause_id="1:cond-gerais/7", parent_id="1:cond-gerais", text=""
    )
    assert find_candidates([target], target.clause_id) == []


@pytest.mark.unit
def test_find_candidates_respects_max_candidates() -> None:
    target = make_record(
        clause_id="1:cond-gerais/7", parent_id="1:cond-gerais", text=""
    )
    siblings = [
        make_record(
            clause_id=f"1:cond-gerais/{i}",
            path=f"cond-gerais/{i}",
            parent_id="1:cond-gerais",
            title=f"{i}. OUTRA",
        )
        for i in range(8, 20)
    ]
    candidates = find_candidates(
        [target, *siblings], target.clause_id, max_candidates=3
    )
    assert len(candidates) == 3
    assert [c.clause_id for c in candidates] == sorted(c.clause_id for c in candidates)


@pytest.mark.unit
def test_find_candidates_no_signals_returns_empty_list() -> None:
    target = make_record(
        clause_id="1:cond-gerais/7",
        parent_id="1:cond-gerais",
        bundle_section="SEÇÃO A",
        text="",
    )
    unrelated = make_record(
        clause_id="1:cond-gerais/9",
        path="cond-gerais/9",
        parent_id="1:cond-gerais/other",
        bundle_section="OUTRA SEÇÃO",
        title="9. OUTRA",
    )
    assert find_candidates([target, unrelated], target.clause_id) == []


@pytest.mark.unit
def test_find_candidates_raises_for_unknown_clause_id() -> None:
    target = make_record()
    with pytest.raises(ClauseNotFoundError, match="not found"):
        find_candidates([target], "does-not-exist")


def make_root_record(**overrides: object) -> ParsedClauseRecord:
    """Build a root-level record with no bundle_section -- the blind-spot state."""
    fields: dict[str, object] = {
        "clause_id": "9:1-2",
        "path": "1-2",
        "parent_id": None,
        "bundle_section": None,
        "text": "",
    }
    fields.update(overrides)
    return make_record(**fields)


@pytest.mark.unit
def test_document_order_neighbours_empty_when_clause_has_a_parent() -> None:
    target = make_record(parent_id="1:cond-gerais", bundle_section=None)
    assert find_document_order_neighbours([target], target) == set()


@pytest.mark.unit
def test_document_order_neighbours_empty_when_clause_has_a_bundle_section() -> None:
    target = make_root_record(bundle_section="SEÇÃO A")
    assert find_document_order_neighbours([target], target) == set()


@pytest.mark.unit
def test_document_order_neighbours_returns_window_either_side() -> None:
    roots = [
        make_root_record(clause_id=f"9:{i}", path=str(i), page_start=i, page_end=i)
        for i in range(1, 10)
    ]
    target = next(r for r in roots if r.clause_id == "9:5")
    assert find_document_order_neighbours(roots, target, window=2) == {
        "9:3",
        "9:4",
        "9:6",
        "9:7",
    }


@pytest.mark.unit
def test_document_order_neighbours_clamps_at_document_start() -> None:
    roots = [
        make_root_record(clause_id=f"9:{i}", path=str(i), page_start=i, page_end=i)
        for i in range(1, 6)
    ]
    target = next(r for r in roots if r.clause_id == "9:1")
    assert find_document_order_neighbours(roots, target, window=3) == {
        "9:2",
        "9:3",
        "9:4",
    }


@pytest.mark.unit
def test_document_order_neighbours_ignores_other_documents() -> None:
    target = make_root_record(clause_id="9:1", path="1", page_start=1, page_end=1)
    other_doc = make_root_record(
        clause_id="8:1", document_id="8", path="1", page_start=1, page_end=1
    )
    assert find_document_order_neighbours([target, other_doc], target) == set()


@pytest.mark.unit
def test_find_candidates_surfaces_neighbours_for_orphan_clause() -> None:
    """The blind spot the fourth signal exists to close (M2-02 review)."""
    roots = [
        make_root_record(clause_id=f"9:1-{i}", path=f"1-{i}", page_start=i, page_end=i)
        for i in range(1, 5)
    ]
    target = next(r for r in roots if r.clause_id == "9:1-2")
    candidates = find_candidates(roots, target.clause_id)
    assert [c.clause_id for c in candidates] == ["9:1-1", "9:1-3", "9:1-4"]
    assert all("document_order_neighbour" in c.reasons for c in candidates)


@pytest.mark.unit
def test_find_candidates_does_not_add_neighbours_when_bundle_section_is_set() -> None:
    target = make_root_record(
        clause_id="9:1-2", path="1-2", bundle_section="SEÇÃO A", page_start=2
    )
    neighbour = make_root_record(
        clause_id="9:1-3", path="1-3", bundle_section="OUTRA SEÇÃO", page_start=3
    )
    candidates = find_candidates([target, neighbour], target.clause_id)
    assert candidates == []
