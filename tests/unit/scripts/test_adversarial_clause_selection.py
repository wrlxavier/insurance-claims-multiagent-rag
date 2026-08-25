"""Tests for deterministic adversarial-pair selection [M2-03].

No LLM calls. Each group covers one selection mechanism the module provides;
the four category slot-builders are tested against small monkeypatched
constants rather than the real corpus, so the mechanism is verified without
depending on production clause ids drifting.
"""

import csv
from pathlib import Path

import pytest
import scripts.adversarial_clause_selection as acs_module
from scripts.adversarial_clause_selection import (
    find_near_duplicate_clause_pairs,
    find_shared_title_clause_pairs,
    is_clause_selectable,
    select_bundle_section_slots,
    select_coverage_with_exclusion_slots,
    select_cross_document_slots,
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


LONG_TEXT_A = "Garante os prejuízos decorrentes de colisão do veículo segurado. " * 3
LONG_TEXT_B = LONG_TEXT_A[:-20] + "com o para-choque amassado."


# --- find_near_duplicate_clause_pairs --------------------------------------


@pytest.mark.unit
def test_find_near_duplicate_clause_pairs_ranks_by_ratio() -> None:
    a = [make_record(clause_id="a:1", document_id="a", text=LONG_TEXT_A)]
    b = [
        make_record(clause_id="b:1", document_id="b", text=LONG_TEXT_A),
        make_record(clause_id="b:2", document_id="b", text=LONG_TEXT_B),
    ]
    pairs = find_near_duplicate_clause_pairs(a, b, min_ratio=0.5)
    assert [p.clause_id_b for p in pairs] == ["b:1", "b:2"]
    assert pairs[0].ratio > pairs[1].ratio


@pytest.mark.unit
def test_find_near_duplicate_clause_pairs_respects_min_ratio_floor() -> None:
    a = [make_record(clause_id="a:1", document_id="a", text=LONG_TEXT_A)]
    b = [
        make_record(
            clause_id="b:1", document_id="b", text="Texto completamente diferente. " * 4
        )
    ]
    pairs = find_near_duplicate_clause_pairs(a, b, min_ratio=0.9)
    assert pairs == []


@pytest.mark.unit
def test_find_near_duplicate_clause_pairs_respects_max_ratio_ceiling() -> None:
    a = [make_record(clause_id="a:1", document_id="a", text=LONG_TEXT_A)]
    b = [make_record(clause_id="b:1", document_id="b", text=LONG_TEXT_A)]
    pairs = find_near_duplicate_clause_pairs(a, b, min_ratio=0.5, max_ratio=0.95)
    assert pairs == []


@pytest.mark.unit
def test_find_near_duplicate_clause_pairs_skips_different_clause_type() -> None:
    a = [
        make_record(
            clause_id="a:1", document_id="a", text=LONG_TEXT_A, clause_type="coverage"
        )
    ]
    b = [
        make_record(
            clause_id="b:1", document_id="b", text=LONG_TEXT_A, clause_type="exclusion"
        )
    ]
    pairs = find_near_duplicate_clause_pairs(a, b, min_ratio=0.5)
    assert pairs == []


@pytest.mark.unit
def test_find_near_duplicate_clause_pairs_skips_short_text() -> None:
    a = [make_record(clause_id="a:1", document_id="a", text="curto")]
    b = [make_record(clause_id="b:1", document_id="b", text="curto")]
    pairs = find_near_duplicate_clause_pairs(a, b, min_ratio=0.5)
    assert pairs == []


@pytest.mark.unit
def test_find_near_duplicate_clause_pairs_skips_length_mismatch() -> None:
    a = [make_record(clause_id="a:1", document_id="a", text=LONG_TEXT_A)]
    b = [
        make_record(
            clause_id="b:1",
            document_id="b",
            text=LONG_TEXT_A + LONG_TEXT_A + LONG_TEXT_A,
        )
    ]
    pairs = find_near_duplicate_clause_pairs(a, b, min_ratio=0.1)
    assert pairs == []


# --- find_shared_title_clause_pairs -----------------------------------------


@pytest.mark.unit
def test_find_shared_title_clause_pairs_matches_after_stripping_numbering() -> None:
    a = [make_record(clause_id="a:1", document_id="a", title="3. ÂMBITO GEOGRÁFICO")]
    b = [make_record(clause_id="b:1", document_id="b", title="23. Âmbito Geográfico")]
    pairs = find_shared_title_clause_pairs(a, b)
    assert [(x.clause_id, y.clause_id) for x, y in pairs] == [("a:1", "b:1")]


@pytest.mark.unit
def test_find_shared_title_clause_pairs_no_match_on_different_titles() -> None:
    a = [make_record(clause_id="a:1", document_id="a", title="1. FRANQUIA")]
    b = [make_record(clause_id="b:1", document_id="b", title="2. SALVADOS")]
    assert find_shared_title_clause_pairs(a, b) == []


# --- is_clause_selectable ---------------------------------------------------


@pytest.mark.unit
def test_is_clause_selectable_rejects_twin() -> None:
    record = make_record(clause_id="a:1", text=LONG_TEXT_A)
    assert (
        is_clause_selectable(record, twins={"a:1": frozenset({"a:2"})}, child_counts={})
        is False
    )


@pytest.mark.unit
def test_is_clause_selectable_rejects_short_text() -> None:
    record = make_record(clause_id="a:1", text="curto")
    assert is_clause_selectable(record, twins={}, child_counts={}) is False


@pytest.mark.unit
def test_is_clause_selectable_accepts_normal_clause() -> None:
    record = make_record(clause_id="a:1", text=LONG_TEXT_A)
    assert is_clause_selectable(record, twins={}, child_counts={}) is True


# --- select_cross_document_slots --------------------------------------------


@pytest.mark.unit
def test_select_cross_document_slots_uses_ranked_near_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(acs_module, "SAME_INSURER_PAIRS", (("a", "b"),))
    monkeypatch.setattr(acs_module, "MANUAL_NEAR_DUPLICATE_PAIRS", {})
    records = [
        make_record(clause_id="a:1", document_id="a", text=LONG_TEXT_A),
        make_record(clause_id="b:1", document_id="b", text=LONG_TEXT_A),
    ]
    slots = select_cross_document_slots(records)
    assert len(slots) == 1
    assert slots[0].row_id == "xdoc-axb-00"
    assert slots[0].adversarial_category == "cross_document"
    assert slots[0].document_id == "a"
    assert slots[0].primary_clause_id == "a:1"
    assert slots[0].distractor_clause_id == "b:1"


@pytest.mark.unit
def test_select_cross_document_slots_falls_back_to_manual_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(acs_module, "SAME_INSURER_PAIRS", (("a", "b"),))
    monkeypatch.setattr(
        acs_module, "MANUAL_NEAR_DUPLICATE_PAIRS", {("a", "b"): [("a:1", "b:1")]}
    )
    records = [
        make_record(clause_id="a:1", document_id="a", text=LONG_TEXT_A),
        make_record(
            clause_id="b:1", document_id="b", text="Texto totalmente distinto. " * 4
        ),
    ]
    slots = select_cross_document_slots(records)
    assert len(slots) == 1
    assert (slots[0].primary_clause_id, slots[0].distractor_clause_id) == ("a:1", "b:1")


@pytest.mark.unit
def test_select_cross_document_slots_skips_unselectable_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(acs_module, "SAME_INSURER_PAIRS", (("a", "b"),))
    monkeypatch.setattr(acs_module, "MANUAL_NEAR_DUPLICATE_PAIRS", {})
    records = [
        # Byte-identical twins within document "a" -- ambiguous ground truth.
        make_record(clause_id="a:1", document_id="a", text=LONG_TEXT_A),
        make_record(clause_id="a:2", document_id="a", text=LONG_TEXT_A),
        make_record(clause_id="b:1", document_id="b", text=LONG_TEXT_A),
    ]
    slots = select_cross_document_slots(records)
    assert slots == []


# --- select_bundle_section_slots --------------------------------------------


@pytest.mark.unit
def test_select_bundle_section_slots_uses_hardcoded_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(acs_module, "BUNDLE_DOCUMENT_ID", "10")
    monkeypatch.setattr(acs_module, "BUNDLE_SECTION_PAIRS", (("10:x", "10:y"),))
    records = [
        make_record(
            clause_id="10:x", document_id="10", text=LONG_TEXT_A, bundle_section="X"
        ),
        make_record(
            clause_id="10:y", document_id="10", text=LONG_TEXT_B, bundle_section="Y"
        ),
    ]
    slots = select_bundle_section_slots(records)
    assert len(slots) == 1
    assert slots[0].row_id == "bundle-10-00"
    assert slots[0].adversarial_category == "bundle_section"
    assert slots[0].document_id == "10"
    assert (slots[0].primary_clause_id, slots[0].distractor_clause_id) == (
        "10:x",
        "10:y",
    )


@pytest.mark.unit
def test_select_bundle_section_slots_skips_missing_clause_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(acs_module, "BUNDLE_DOCUMENT_ID", "10")
    monkeypatch.setattr(
        acs_module, "BUNDLE_SECTION_PAIRS", (("10:missing", "10:also-missing"),)
    )
    assert select_bundle_section_slots([]) == []


# --- select_coverage_with_exclusion_slots -----------------------------------


@pytest.mark.unit
def test_select_coverage_with_exclusion_slots_pairs_coverage_with_exclusion(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "product_line"])
        writer.writeheader()
        writer.writerow({"id": "1", "product_line": "CASCO"})

    coverage = make_record(
        clause_id="1:parent/coverage",
        document_id="1",
        parent_id="1:parent",
        path="parent/coverage",
        title="RISCOS COBERTOS",
        text=LONG_TEXT_A,
        clause_type="coverage",
    )
    exclusion = make_record(
        clause_id="1:parent/exclusion",
        document_id="1",
        parent_id="1:parent",
        path="parent/exclusion",
        title="RISCOS EXCLUÍDOS",
        text=LONG_TEXT_B,
        clause_type="exclusion",
    )
    records = [coverage, exclusion]

    slots = select_coverage_with_exclusion_slots(records, manifest_path, max_slots=5)

    assert len(slots) == 1
    assert slots[0].adversarial_category == "coverage_with_exclusion"
    assert slots[0].primary_clause_id == "1:parent/coverage"
    assert slots[0].secondary_clause_id == "1:parent/exclusion"


@pytest.mark.unit
def test_select_coverage_with_exclusion_slots_excludes_already_used_ids(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "product_line"])
        writer.writeheader()
        writer.writerow({"id": "1", "product_line": "CASCO"})

    coverage = make_record(
        clause_id="1:parent/coverage",
        document_id="1",
        parent_id="1:parent",
        path="parent/coverage",
        clause_type="coverage",
        text=LONG_TEXT_A,
    )
    exclusion = make_record(
        clause_id="1:parent/exclusion",
        document_id="1",
        parent_id="1:parent",
        path="parent/exclusion",
        clause_type="exclusion",
        text=LONG_TEXT_B,
    )
    records = [coverage, exclusion]

    slots = select_coverage_with_exclusion_slots(
        records,
        manifest_path,
        already_used_ids=frozenset({"1:parent/coverage"}),
        max_slots=5,
    )
    assert slots == []
