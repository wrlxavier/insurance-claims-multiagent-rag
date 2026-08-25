"""Tests for the golden-set validation script."""

from pathlib import Path

import pytest
from pydantic import ValidationError
from scripts.validate_golden_set import (
    GoldenSetValidationError,
    check_document_ids_exist,
    check_question_ids_unique,
    check_question_type_matches_filename,
    check_reference_clause_ids_exist,
)

from infrastructure.evaluation.golden_set_schema import GoldenQuestion


def make_question(**overrides: object) -> GoldenQuestion:
    """Build a valid direct_lookup GoldenQuestion, overridable per test."""
    fields: dict[str, object] = {
        "schema_version": "v1",
        "question_id": "direct_lookup-001",
        "document_id": "1",
        "question": "O que caracteriza perda total do veículo?",
        "reference_clause_ids": ["1:4/4.2"],
        "question_type": "direct_lookup",
        "difficulty": "easy",
        "expected_verdict": None,
        "notes": "",
        "authored_at": "2026-08-21",
    }
    fields.update(overrides)
    return GoldenQuestion.model_validate(fields)


@pytest.mark.unit
def test_unanswerable_question_rejects_nonempty_reference_clause_ids() -> None:
    with pytest.raises(ValidationError, match="empty reference_clause_ids"):
        make_question(
            question_type="unanswerable",
            reference_clause_ids=["1:4/4.2"],
            expected_verdict="insufficient_information",
        )


@pytest.mark.unit
def test_unanswerable_question_requires_insufficient_information_verdict() -> None:
    with pytest.raises(ValidationError, match="insufficient_information"):
        make_question(
            question_type="unanswerable",
            reference_clause_ids=[],
            expected_verdict="compatible",
        )


@pytest.mark.unit
def test_unanswerable_question_accepts_empty_reference_clause_ids() -> None:
    question = make_question(
        question_type="unanswerable",
        reference_clause_ids=[],
        expected_verdict="insufficient_information",
    )
    assert question.reference_clause_ids == []


@pytest.mark.unit
def test_non_unanswerable_question_requires_at_least_one_reference_clause() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        make_question(reference_clause_ids=[])


@pytest.mark.unit
def test_check_question_type_matches_filename_rejects_mismatch() -> None:
    question = make_question(question_type="definition")
    with pytest.raises(GoldenSetValidationError, match="expected 'direct_lookup'"):
        check_question_type_matches_filename(
            Path("data/golden_set/direct_lookup.jsonl"), [question]
        )


@pytest.mark.unit
def test_check_question_type_matches_filename_rejects_unknown_filename() -> None:
    question = make_question()
    with pytest.raises(GoldenSetValidationError, match="not a valid question_type"):
        check_question_type_matches_filename(
            Path("data/golden_set/not_a_type.jsonl"), [question]
        )


@pytest.mark.unit
def test_check_question_type_matches_filename_accepts_match() -> None:
    question = make_question(question_type="direct_lookup")
    check_question_type_matches_filename(
        Path("data/golden_set/direct_lookup.jsonl"), [question]
    )


@pytest.mark.unit
def test_check_document_ids_exist_rejects_unknown_document() -> None:
    question = make_question(document_id="999")
    with pytest.raises(GoldenSetValidationError, match="unknown document_id"):
        check_document_ids_exist(
            Path("data/golden_set/direct_lookup.jsonl"), [question], {"1", "2"}
        )


@pytest.mark.unit
def test_check_reference_clause_ids_exist_rejects_unknown_clause() -> None:
    question = make_question(reference_clause_ids=["1:missing"])
    with pytest.raises(GoldenSetValidationError, match="unknown clause_id"):
        check_reference_clause_ids_exist(
            Path("data/golden_set/direct_lookup.jsonl"), [question], {"1:4/4.2"}
        )


@pytest.mark.unit
def test_check_question_ids_unique_rejects_duplicate_across_files() -> None:
    question = make_question(question_id="direct_lookup-001")
    other = make_question(question_id="direct_lookup-001")
    with pytest.raises(GoldenSetValidationError, match="duplicate question_id"):
        check_question_ids_unique(
            [
                (Path("data/golden_set/direct_lookup.jsonl"), question),
                (Path("data/golden_set/direct_lookup.jsonl"), other),
            ]
        )


@pytest.mark.unit
def test_check_question_ids_unique_accepts_distinct_ids() -> None:
    first = make_question(question_id="direct_lookup-001")
    second = make_question(question_id="direct_lookup-002")
    check_question_ids_unique(
        [
            (Path("data/golden_set/direct_lookup.jsonl"), first),
            (Path("data/golden_set/direct_lookup.jsonl"), second),
        ]
    )
