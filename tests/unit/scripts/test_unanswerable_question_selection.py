"""Tests for deterministic unanswerable-question candidate selection [M2-05].

Each group covers one selection mechanism the module provides; no LLM
calls, and no live corpus dependency (all fixtures are hand-built records).
"""

import csv
from pathlib import Path

import pytest
from scripts.unanswerable_question_selection import (
    ABSENCE_REASON_PT,
    FACT_TYPE_LABELS_PT,
    FACT_TYPES,
    DecoySpec,
    search_document_for_fact,
    select_clean_absent_slots,
    select_decoy_slots,
    select_unanswerable_slots,
)

from infrastructure.parsing.clause_schema import ParsedClauseRecord


def make_record(**overrides: object) -> ParsedClauseRecord:
    """Build a valid ParsedClauseRecord, overridable per test."""
    fields: dict[str, object] = {
        "schema_version": "v1",
        "clause_id": "1:franquia",
        "document_id": "1",
        "parent_id": None,
        "path": "franquia",
        "title": "Franquia",
        "text": "A participação obrigatória do segurado é chamada de franquia.",
        "clause_type": "condition",
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
        "product_line": "CASCO",
        "indemnity_regime": "VD",
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


# --- search_document_for_fact -------------------------------------------


@pytest.mark.unit
def test_search_document_for_fact_finds_concrete_deductible_value() -> None:
    record = make_record(
        clause_id="1:franquia",
        text="A franquia deste sinistro será de R$ 1.200,00 por evento.",
    )
    hits = search_document_for_fact([record], "1", "deductible")
    assert len(hits) == 1
    assert hits[0][0] == "1:franquia"


@pytest.mark.unit
def test_search_document_for_fact_matches_across_a_line_break() -> None:
    """Clause text wraps at PDF layout boundaries, not sentence structure --
    a search window that excludes newlines misses real hits (regression:
    document 14's real "R$ 400 reais de desconto na\\nfranquia" hit)."""
    record = make_record(
        clause_id="14:desconto",
        document_id="14",
        text="Você recebe R$ 400 reais de desconto na\nfranquia deste produto.",
    )
    hits = search_document_for_fact([record], "14", "deductible")
    assert len(hits) == 1


@pytest.mark.unit
def test_search_document_for_fact_returns_empty_when_absent() -> None:
    record = make_record(
        text="A franquia é a participação obrigatória do segurado no sinistro."
    )
    hits = search_document_for_fact([record], "1", "deductible")
    assert hits == []


@pytest.mark.unit
def test_search_document_for_fact_ignores_other_documents() -> None:
    record = make_record(
        document_id="2",
        text="A franquia deste sinistro será de R$ 1.200,00 por evento.",
    )
    hits = search_document_for_fact([record], "1", "deductible")
    assert hits == []


# --- select_clean_absent_slots -------------------------------------------


@pytest.mark.unit
def test_select_clean_absent_slots_meets_floor_and_spreads_documents(
    tmp_path: Path,
) -> None:
    manifest_path = write_manifest(
        tmp_path,
        [{"id": str(doc_id), "product_line": "CASCO"} for doc_id in range(1, 11)],
    )
    records = [
        make_record(
            clause_id=f"{doc_id}:clausula",
            document_id=str(doc_id),
            text="Texto sem nenhum valor monetário específico.",
        )
        for doc_id in range(1, 11)
    ]
    slots = select_clean_absent_slots(records, manifest_path, target_per_fact_type=2)
    assert len(slots) == 2 * len(FACT_TYPES)
    # Every slot claims a genuinely clean (document, fact_type) pair.
    for slot in slots:
        assert slot.slot_kind == "clean_absent"
        assert search_document_for_fact(records, slot.document_id, slot.fact_type) == []
    # With 10 documents and 2 needed per fact type (10 total across 5 fact
    # types), documents should not need to repeat.
    document_ids = [slot.document_id for slot in slots]
    assert len(set(document_ids)) == len(document_ids)


@pytest.mark.unit
def test_select_clean_absent_slots_excludes_documents_with_a_hit(
    tmp_path: Path,
) -> None:
    manifest_path = write_manifest(
        tmp_path,
        [
            {"id": "1", "product_line": "CASCO"},
            {"id": "2", "product_line": "CASCO"},
        ],
    )
    records = [
        make_record(
            clause_id="1:franquia",
            document_id="1",
            text="A franquia deste sinistro será de R$ 1.200,00.",
        ),
        make_record(
            clause_id="2:franquia",
            document_id="2",
            text="A franquia é a participação obrigatória do segurado.",
        ),
    ]
    slots = select_clean_absent_slots(records, manifest_path, target_per_fact_type=1)
    deductible_slots = [s for s in slots if s.fact_type == "deductible"]
    assert [s.document_id for s in deductible_slots] == ["2"]


@pytest.mark.unit
def test_select_clean_absent_slots_falls_back_to_reuse_when_corpus_is_small(
    tmp_path: Path,
) -> None:
    manifest_path = write_manifest(tmp_path, [{"id": "1", "product_line": "CASCO"}])
    records = [make_record(text="Texto sem nenhum valor monetário específico.")]
    slots = select_clean_absent_slots(records, manifest_path, target_per_fact_type=1)
    # Only one document exists, so every fact type must reuse it rather
    # than come up short.
    assert len(slots) == len(FACT_TYPES)
    assert all(slot.document_id == "1" for slot in slots)


# --- select_decoy_slots ---------------------------------------------------


@pytest.mark.unit
def test_select_decoy_slots_builds_a_slot_when_the_hit_still_fires() -> None:
    record = make_record(
        clause_id="1:franquia",
        text="A franquia deste sinistro será de R$ 1.200,00 por evento.",
    )
    spec = DecoySpec(
        document_id="1",
        fact_type="deductible",
        decoy_clause_id="1:franquia",
        why_it_does_not_answer="Não é o valor geral aplicável.",
    )
    slots = select_decoy_slots([record], decoy_specs=(spec,))
    assert len(slots) == 1
    assert slots[0].slot_kind == "decoy"
    assert slots[0].decoy_clause_id == "1:franquia"
    assert "Não é o valor geral aplicável." in slots[0].search_evidence


@pytest.mark.unit
def test_select_decoy_slots_drops_spec_when_hit_no_longer_fires() -> None:
    """A corpus re-parse that changes the clause's text must not silently
    ship a stale decoy -- the spec is dropped, not fabricated."""
    record = make_record(
        clause_id="1:franquia", text="Texto reescrito sem nenhum valor."
    )
    spec = DecoySpec(
        document_id="1",
        fact_type="deductible",
        decoy_clause_id="1:franquia",
        why_it_does_not_answer="Não é o valor geral aplicável.",
    )
    slots = select_decoy_slots([record], decoy_specs=(spec,))
    assert slots == []


@pytest.mark.unit
def test_select_decoy_slots_drops_spec_when_hit_is_a_different_clause() -> None:
    record = make_record(
        clause_id="1:outra-clausula",
        text="A franquia deste sinistro será de R$ 1.200,00 por evento.",
    )
    spec = DecoySpec(
        document_id="1",
        fact_type="deductible",
        decoy_clause_id="1:franquia",  # not the clause_id that actually hit
        why_it_does_not_answer="Não é o valor geral aplicável.",
    )
    slots = select_decoy_slots([record], decoy_specs=(spec,))
    assert slots == []


# --- select_unanswerable_slots --------------------------------------------


@pytest.mark.unit
def test_select_unanswerable_slots_is_clean_absent_then_decoy(tmp_path: Path) -> None:
    """select_unanswerable_slots is exactly the concatenation of the two
    selectors -- clean_absent slots first, decoy slots appended after."""
    manifest_path = write_manifest(tmp_path, [{"id": "1", "product_line": "CASCO"}])
    records = [make_record()]

    combined = select_unanswerable_slots(records, manifest_path, target_per_fact_type=1)
    clean = select_clean_absent_slots(records, manifest_path, target_per_fact_type=1)
    decoy = select_decoy_slots(records)

    assert combined == clean + decoy


# --- static data completeness --------------------------------------------


@pytest.mark.unit
def test_every_fact_type_has_a_label_and_absence_reason() -> None:
    for fact_type in FACT_TYPES:
        assert fact_type in FACT_TYPE_LABELS_PT
        assert fact_type in ABSENCE_REASON_PT
        assert ABSENCE_REASON_PT[fact_type]
