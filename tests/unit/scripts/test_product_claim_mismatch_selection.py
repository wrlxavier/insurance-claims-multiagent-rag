"""Tests for deterministic product/claim mismatch scenario selection [M2-05].

Each group covers one selection mechanism the module provides; no LLM calls.
"""

import csv
from pathlib import Path

import pytest
from scripts.product_claim_mismatch_selection import (
    REQUIRED_DOCUMENT_IDS,
    TARGET_COUNTS,
    load_non_casco_documents_by_line,
    select_anchor_clause,
    select_mismatch_slots,
)

from infrastructure.parsing.clause_schema import ParsedClauseRecord


def make_record(**overrides: object) -> ParsedClauseRecord:
    """Build a valid ParsedClauseRecord, overridable per test."""
    fields: dict[str, object] = {
        "schema_version": "v1",
        "clause_id": "16:objetivo",
        "document_id": "16",
        "parent_id": None,
        "path": "objetivo",
        "title": "1. OBJETIVO DO SEGURO",
        "text": "Este seguro tem por objetivo garantir o pagamento de indenização "
        "por danos causados pelo segurado a terceiros.",
        "clause_type": "coverage",
        "type_source": "rule",
        "confidence": None,
        "bundle_section": None,
        "page_start": 1,
        "page_end": 1,
        "source": "text",
        "boundary_source": None,
        "susep_process": "123",
        "insurer": "Test Seguros",
        "cnpj": "00000000000100",
        "product_line": "RCF-A",
        "indemnity_regime": "n/a",
        "filing_year": "2024",
    }
    fields.update(overrides)
    return ParsedClauseRecord.model_validate(fields)


def write_manifest(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "product_line"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


# --- load_non_casco_documents_by_line ---------------------------------------


@pytest.mark.unit
def test_load_non_casco_documents_by_line_excludes_casco_and_sorts(
    tmp_path: Path,
) -> None:
    manifest_path = write_manifest(
        tmp_path,
        [
            {"id": "10", "product_line": "CASCO"},
            {"id": "20", "product_line": "RCF-A"},
            {"id": "16", "product_line": "RCF-A"},
            {"id": "30", "product_line": "CARTA VERDE"},
        ],
    )
    grouped = load_non_casco_documents_by_line(manifest_path)
    assert "CASCO" not in grouped
    assert [row["id"] for row in grouped["RCF-A"]] == ["16", "20"]
    assert [row["id"] for row in grouped["CARTA VERDE"]] == ["30"]


# --- select_anchor_clause ----------------------------------------------------


@pytest.mark.unit
def test_select_anchor_clause_prefers_strong_title_match() -> None:
    weak = make_record(
        clause_id="16:ambito", title="5. Âmbito Geográfico", text="x" * 40
    )
    strong = make_record(
        clause_id="16:objetivo", title="2. OBJETIVO DO SEGURO", text="y" * 40
    )
    chosen = select_anchor_clause([weak, strong])
    assert chosen is not None
    assert chosen.clause_id == "16:objetivo"


@pytest.mark.unit
def test_select_anchor_clause_matches_plural_riscos_cobertos() -> None:
    record = make_record(
        clause_id="17:riscos",
        title="2.  Riscos Cobertos, o seguro garante...",
        text="x" * 40,
    )
    other = make_record(
        clause_id="17:outra", title="6. Atualização Monetária", text="y" * 40
    )
    chosen = select_anchor_clause([other, record])
    assert chosen is not None
    assert chosen.clause_id == "17:riscos"


@pytest.mark.unit
def test_select_anchor_clause_rejects_mid_sentence_false_positive() -> None:
    """A title that merely CONTAINS the phrase mid-sentence must not win.

    Regression case found against the real corpus (document 17): a
    condition clause about notifying risk aggravation mentions "risco
    coberto" mid-sentence and must not outrank a genuine "Âmbito
    Geográfico" heading in the same document.
    """
    false_positive = make_record(
        clause_id="17:condicao",
        title="8.1.1.4. Em caso de agravação do risco coberto, a Seguradora "
        "poderá dar ciência",
        text="x" * 40,
    )
    weak = make_record(
        clause_id="17:ambito", title="5. Âmbito Geográfico", text="y" * 40
    )
    chosen = select_anchor_clause([false_positive, weak])
    assert chosen is not None
    assert chosen.clause_id == "17:ambito"


@pytest.mark.unit
def test_select_anchor_clause_uses_body_opening_when_title_is_silent() -> None:
    """Document 25's real shape: an empty-bodied root heading plus a child
    sub-clause whose title is just the sentence's opening words."""
    body_match = make_record(
        clause_id="25:1.1",
        title="1.1. O presente seguro tem por objetivo garantir ao Segurado",
        text="O presente seguro tem por objetivo garantir ao Segurado o "
        "reembolso de despesas.",
    )
    no_match = make_record(
        clause_id="25:outra", title="Rede Credenciada", text="x" * 40
    )
    chosen = select_anchor_clause([no_match, body_match])
    assert chosen is not None
    assert chosen.clause_id == "25:1.1"


@pytest.mark.unit
def test_select_anchor_clause_falls_back_to_lowest_clause_id() -> None:
    a = make_record(clause_id="16:b", title="Outra Coisa", text="x" * 40)
    b = make_record(clause_id="16:a", title="Mais Uma Coisa", text="y" * 40)
    chosen = select_anchor_clause([a, b])
    assert chosen is not None
    assert chosen.clause_id == "16:a"


@pytest.mark.unit
def test_select_anchor_clause_empty_candidates_returns_none() -> None:
    assert select_anchor_clause([]) is None


# --- select_mismatch_slots ----------------------------------------------------


@pytest.mark.unit
def test_select_mismatch_slots_includes_required_documents(tmp_path: Path) -> None:
    manifest_path = write_manifest(
        tmp_path,
        [
            {"id": "16", "product_line": "RCF-A"},
            {"id": "21", "product_line": "RCF-A"},
            {"id": "22", "product_line": "RCF-A"},
            {"id": "23", "product_line": "RCF-A"},
        ],
    )
    records = [
        make_record(
            clause_id=f"{doc_id}:objetivo",
            document_id=doc_id,
            title="1. OBJETIVO DO SEGURO",
            text="x" * 40,
        )
        for doc_id in ("16", "21", "22", "23")
    ]
    slots = select_mismatch_slots(
        records,
        manifest_path,
        target_counts={"RCF-A": 2},
    )
    document_ids = {slot.document_id for slot in slots}
    assert REQUIRED_DOCUMENT_IDS.issubset(document_ids | {"16", "23"})
    assert {"21", "22"}.issubset(document_ids)
    assert len(slots) == 2


@pytest.mark.unit
def test_select_mismatch_slots_skips_documents_with_no_candidate(
    tmp_path: Path,
) -> None:
    manifest_path = write_manifest(
        tmp_path,
        [
            {"id": "16", "product_line": "RCF-A"},
            {"id": "17", "product_line": "RCF-A"},
        ],
    )
    # Document 17 has no selectable core-type clause at all.
    records = [
        make_record(
            clause_id="16:objetivo",
            document_id="16",
            title="1. OBJETIVO DO SEGURO",
            text="x" * 40,
        ),
        make_record(
            clause_id="17:other",
            document_id="17",
            clause_type="other",
            title="Nota",
            text="y" * 40,
        ),
    ]
    slots = select_mismatch_slots(records, manifest_path, target_counts={"RCF-A": 2})
    assert [slot.document_id for slot in slots] == ["16"]


@pytest.mark.unit
def test_target_counts_meet_dod_floor() -> None:
    """DoD floor is >=8 total product/claim mismatch claims."""
    assert sum(TARGET_COUNTS.values()) >= 8
