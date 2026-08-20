"""Tests for the LLM validation script."""

import pytest
from scripts.validate_parsing_quality_sample import (
    apply_llm_validation,
    derive_failure_mode_tag,
    resolve_page_range,
)


@pytest.mark.unit
def test_resolve_page_range_adds_one_page_of_margin_each_side() -> None:
    first, last = resolve_page_range(page_start=4, page_end=7, page_count=20)

    assert (first, last) == (3, 8)


@pytest.mark.unit
def test_resolve_page_range_clamps_to_document_start() -> None:
    first, last = resolve_page_range(page_start=1, page_end=2, page_count=20)

    assert (first, last) == (1, 3)


@pytest.mark.unit
def test_resolve_page_range_clamps_to_document_end() -> None:
    first, last = resolve_page_range(page_start=18, page_end=20, page_count=20)

    assert (first, last) == (17, 20)


@pytest.mark.unit
def test_resolve_page_range_single_page_document() -> None:
    first, last = resolve_page_range(page_start=1, page_end=1, page_count=1)

    assert (first, last) == (1, 1)


@pytest.mark.unit
def test_derive_failure_mode_tag_all_correct_is_empty() -> None:
    tag = derive_failure_mode_tag(
        boundary_correct=True, type_correct=True, provenance_correct=True
    )

    assert tag == ""


@pytest.mark.unit
def test_derive_failure_mode_tag_boundary_takes_priority() -> None:
    tag = derive_failure_mode_tag(
        boundary_correct=False, type_correct=False, provenance_correct=False
    )

    assert tag == "boundary_mismatch_llm"


@pytest.mark.unit
def test_derive_failure_mode_tag_type_mismatch_only() -> None:
    tag = derive_failure_mode_tag(
        boundary_correct=True, type_correct=False, provenance_correct=True
    )

    assert tag == "type_mismatch_llm"


@pytest.mark.unit
def test_derive_failure_mode_tag_provenance_mismatch_only() -> None:
    tag = derive_failure_mode_tag(
        boundary_correct=True, type_correct=True, provenance_correct=False
    )

    assert tag == "provenance_mismatch_llm"


def _sample_row() -> dict[str, str]:
    return {
        "sample_id": "1",
        "clause_id": "1:glossario",
        "predicted_clause_type": "definition",
        "boundary_correct": "",
        "boundary_notes": "",
        "reference_clause_type": "",
        "provenance_correct": "",
        "provenance_notes": "",
        "failure_mode_tag": "",
        "reviewer_notes": "",
    }


@pytest.mark.unit
def test_apply_llm_validation_fills_judgment_columns_from_llm_output() -> None:
    row = _sample_row()
    validation: dict[str, object] = {
        "llm_boundary_correct": True,
        "llm_boundary_notes": "Starts and ends cleanly on the recorded pages.",
        "llm_reference_clause_type": "definition",
        "llm_provenance_correct": True,
        "llm_provenance_notes": "Insurer and product line match the footer.",
        "llm_reasoning": "Boundary, type and provenance all check out.",
    }

    updated = apply_llm_validation(row, validation)

    assert updated["boundary_correct"] == "TRUE"
    assert updated["boundary_notes"] == "Starts and ends cleanly on the recorded pages."
    assert updated["reference_clause_type"] == "definition"
    assert updated["provenance_correct"] == "TRUE"
    assert updated["failure_mode_tag"] == ""
    assert updated["reviewer_notes"] == "Boundary, type and provenance all check out."
    # Original row is not mutated.
    assert row["boundary_correct"] == ""


@pytest.mark.unit
def test_apply_llm_validation_detects_type_mismatch_against_predicted_type() -> None:
    row = _sample_row()
    validation: dict[str, object] = {
        "llm_boundary_correct": True,
        "llm_boundary_notes": "ok",
        "llm_reference_clause_type": "exclusion",
        "llm_provenance_correct": True,
        "llm_provenance_notes": "ok",
        "llm_reasoning": "The parser mislabeled this as a definition.",
    }

    updated = apply_llm_validation(row, validation)

    assert updated["reference_clause_type"] == "exclusion"
    assert updated["failure_mode_tag"] == "type_mismatch_llm"
