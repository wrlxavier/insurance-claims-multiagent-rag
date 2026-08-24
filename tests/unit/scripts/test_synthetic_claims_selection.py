"""Tests for deterministic synthetic-claim scenario selection [M2-04].

Each group covers one selection mechanism the module provides; no LLM calls.
"""

import csv
from pathlib import Path

import pytest
from scripts.synthetic_claims_selection import (
    MISSING_FACT_PATTERNS,
    _round_robin_take,
    load_documents_by_product_line,
    select_compatible_slots,
    select_incompatible_slots,
    select_insufficient_information_slots,
)

from infrastructure.parsing.clause_schema import ParsedClauseRecord


def make_record(**overrides: object) -> ParsedClauseRecord:
    """Build a valid ParsedClauseRecord, overridable per test."""
    fields: dict[str, object] = {
        "schema_version": "v1",
        "clause_id": "1:riscos-cobertos",
        "document_id": "1",
        "parent_id": None,
        "path": "riscos-cobertos",
        "title": "RISCOS COBERTOS",
        "text": "Garante os prejuízos decorrentes de colisão do veículo segurado.",
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
        "product_line": "CASCO",
        "indemnity_regime": "VD",
        "filing_year": "2024",
    }
    fields.update(overrides)
    return ParsedClauseRecord.model_validate(fields)


LONG_TEXT = "Garante os prejuízos decorrentes de colisão do veículo segurado. " * 3


def write_manifest(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "product_line"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


# --- _round_robin_take -------------------------------------------------------


@pytest.mark.unit
def test_round_robin_take_spreads_across_groups() -> None:
    items = [("a", 1), ("a", 2), ("b", 3), ("c", 4)]
    taken = _round_robin_take(items, 3)
    assert [key for key, _ in taken] == ["a", "b", "c"]


@pytest.mark.unit
def test_round_robin_take_falls_back_to_second_item_in_group() -> None:
    items = [("a", 1), ("a", 2)]
    taken = _round_robin_take(items, 2)
    assert [value for _, value in taken] == [1, 2]


@pytest.mark.unit
def test_round_robin_take_stops_when_exhausted() -> None:
    items = [("a", 1)]
    taken = _round_robin_take(items, 5)
    assert taken == [("a", 1)]


# --- load_documents_by_product_line -----------------------------------------


@pytest.mark.unit
def test_load_documents_by_product_line_groups_and_sorts(tmp_path: Path) -> None:
    manifest_path = write_manifest(
        tmp_path,
        [
            {"id": "10", "product_line": "CASCO"},
            {"id": "2", "product_line": "CASCO"},
            {"id": "16", "product_line": "RCF-A"},
        ],
    )
    grouped = load_documents_by_product_line(manifest_path)
    assert [row["id"] for row in grouped["CASCO"]] == ["2", "10"]
    assert [row["id"] for row in grouped["RCF-A"]] == ["16"]


# --- select_compatible_slots -------------------------------------------------


@pytest.mark.unit
def test_select_compatible_slots_picks_coverage_clauses() -> None:
    records = [
        make_record(
            clause_id="1:coverage",
            document_id="1",
            clause_type="coverage",
            text=LONG_TEXT,
        ),
        make_record(
            clause_id="1:condition",
            document_id="1",
            clause_type="condition",
            text=LONG_TEXT,
        ),
    ]
    slots = select_compatible_slots(
        records,
        [{"id": "1", "product_line": "CASCO"}],
        product_line="CASCO",
        ancestor_titles={},
        twins={},
        target_count=5,
    )
    assert len(slots) == 1
    assert slots[0].scenario_type == "compatible"
    assert slots[0].primary_clause_id == "1:coverage"
    assert slots[0].row_id == "sc-compat-1-00"


@pytest.mark.unit
def test_select_compatible_slots_respects_already_used_ids() -> None:
    records = [
        make_record(clause_id="1:coverage", document_id="1", text=LONG_TEXT),
    ]
    slots = select_compatible_slots(
        records,
        [{"id": "1", "product_line": "CASCO"}],
        product_line="CASCO",
        ancestor_titles={},
        twins={},
        target_count=5,
        already_used_ids=frozenset({"1:coverage"}),
    )
    assert slots == []


@pytest.mark.unit
def test_select_compatible_slots_spreads_across_documents() -> None:
    records = [
        make_record(clause_id="1:coverage", document_id="1", text=LONG_TEXT),
        make_record(clause_id="2:coverage", document_id="2", text=LONG_TEXT),
    ]
    slots = select_compatible_slots(
        records,
        [{"id": "1", "product_line": "CASCO"}, {"id": "2", "product_line": "CASCO"}],
        product_line="CASCO",
        ancestor_titles={},
        twins={},
        target_count=2,
    )
    assert {slot.document_id for slot in slots} == {"1", "2"}


# --- select_incompatible_slots -----------------------------------------------


@pytest.mark.unit
def test_select_incompatible_slots_pairs_structural_coverage_and_exclusion() -> None:
    coverage = make_record(
        clause_id="1:parent/coverage",
        document_id="1",
        parent_id="1:parent",
        path="parent/coverage",
        clause_type="coverage",
        text=LONG_TEXT,
    )
    exclusion = make_record(
        clause_id="1:parent/exclusion",
        document_id="1",
        parent_id="1:parent",
        path="parent/exclusion",
        title="RISCOS EXCLUÍDOS",
        clause_type="exclusion",
        text=LONG_TEXT,
    )
    records = [coverage, exclusion]
    slots = select_incompatible_slots(
        records,
        [{"id": "1", "product_line": "CASCO"}],
        product_line="CASCO",
        ancestor_titles={},
        twins={},
        target_count=5,
    )
    assert len(slots) == 1
    assert slots[0].scenario_type == "incompatible"
    assert slots[0].primary_clause_id == "1:parent/coverage"
    assert slots[0].secondary_clause_id == "1:parent/exclusion"
    assert slots[0].selection_notes == ""


@pytest.mark.unit
def test_select_incompatible_slots_falls_back_to_standalone_exclusion() -> None:
    # No shared parent/bundle_section/cross-reference -- find_candidates finds
    # nothing structural, so the standalone-exclusion fallback should fire.
    exclusion = make_record(
        clause_id="1:exclusion",
        document_id="1",
        clause_type="exclusion",
        title="RISCOS EXCLUÍDOS",
        text=LONG_TEXT,
    )
    records = [exclusion]
    slots = select_incompatible_slots(
        records,
        [{"id": "1", "product_line": "CASCO"}],
        product_line="CASCO",
        ancestor_titles={},
        twins={},
        target_count=5,
    )
    assert len(slots) == 1
    assert slots[0].primary_clause_id == "1:exclusion"
    assert slots[0].secondary_clause_id is None
    assert slots[0].selection_notes == "exclusion_only_fallback"


# --- select_insufficient_information_slots -----------------------------------


@pytest.mark.unit
def test_select_insufficient_information_slots_matches_fact_pattern() -> None:
    clause = make_record(
        clause_id="1:ambito",
        document_id="1",
        clause_type="condition",
        title="ÂMBITO GEOGRÁFICO",
        text="A cobertura é válida em todo o território nacional. " * 3,
    )
    records = [clause]
    slots = select_insufficient_information_slots(
        records,
        [{"id": "1", "product_line": "CASCO"}],
        product_line="CASCO",
        ancestor_titles={},
        twins={},
        target_count=5,
    )
    assert len(slots) == 1
    assert slots[0].scenario_type == "insufficient_information"
    assert slots[0].primary_clause_id == "1:ambito"
    assert slots[0].missing_fact_type == "ambito_geografico"


@pytest.mark.unit
def test_select_insufficient_information_slots_skips_clauses_with_no_match() -> None:
    clause = make_record(
        clause_id="1:generic",
        document_id="1",
        clause_type="condition",
        title="DISPOSIÇÕES GERAIS",
        text=LONG_TEXT,
    )
    slots = select_insufficient_information_slots(
        [clause],
        [{"id": "1", "product_line": "CASCO"}],
        product_line="CASCO",
        ancestor_titles={},
        twins={},
        target_count=5,
    )
    assert slots == []


@pytest.mark.unit
def test_missing_fact_patterns_cover_every_declared_fact_type() -> None:
    # Every fact type must have a non-empty pattern -- a silently empty
    # pattern would match everything (re.search("", text) is always true).
    for fact_type, pattern in MISSING_FACT_PATTERNS.items():
        assert pattern.strip(), fact_type
