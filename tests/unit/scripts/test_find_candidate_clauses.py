"""Tests for the deterministic candidate-clause search script."""

import pytest
from scripts.find_candidate_clauses import (
    ClauseNotFoundError,
    extract_cross_references,
    find_candidates,
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
