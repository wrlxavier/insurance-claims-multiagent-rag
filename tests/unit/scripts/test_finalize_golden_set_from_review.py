"""Tests for promoting approved draft rows into the golden set [M2-02].

Uses `monkeypatch` to redirect [scripts.finalize_golden_set_from_review.
GOLDEN_SET_DIR] at a tmp_path for every test that touches it, so these
tests never read or write the repo's real `data/golden_set/`.
"""

from datetime import date
from pathlib import Path

import pytest
import scripts.finalize_golden_set_from_review as finalize_module
from scripts.finalize_golden_set_from_review import (
    FinalizeError,
    finalize_rows,
    is_approved,
    next_sequence_numbers,
    row_to_golden_question,
)

from infrastructure.evaluation.golden_set_schema import GoldenQuestion

AUTHORED_AT = date(2026, 8, 22)


def make_row(**overrides: str) -> dict[str, str]:
    """Build a valid, approved, unfinalized draft CSV row, overridable per test."""
    fields = {
        "row_id": "1-00",
        "document_id": "1",
        "question": "O que caracteriza perda total do veículo?",
        "reference_clause_ids": "1:4/4.2",
        "question_type": "direct_lookup",
        "difficulty": "easy",
        "expected_verdict": "",
        "notes": "",
        "approved": "Y",
        "finalized_question_id": "",
    }
    fields.update(overrides)
    return fields


@pytest.mark.unit
@pytest.mark.parametrize("value", ["Y", "y", "yes", "TRUE", "1"])
def test_is_approved_accepts_truthy_values(value: str) -> None:
    assert is_approved(make_row(approved=value)) is True


@pytest.mark.unit
@pytest.mark.parametrize("value", ["", "N", "no", "maybe"])
def test_is_approved_rejects_non_truthy_values(value: str) -> None:
    assert is_approved(make_row(approved=value)) is False


@pytest.mark.unit
def test_row_to_golden_question_splits_reference_clause_ids() -> None:
    row = make_row(reference_clause_ids="1:a;1:b")
    question = row_to_golden_question(
        row, question_id="direct_lookup-001", authored_at=AUTHORED_AT
    )
    assert question.reference_clause_ids == ["1:a", "1:b"]


@pytest.mark.unit
def test_row_to_golden_question_maps_empty_expected_verdict_to_none() -> None:
    row = make_row(expected_verdict="")
    question = row_to_golden_question(
        row, question_id="direct_lookup-001", authored_at=AUTHORED_AT
    )
    assert question.expected_verdict is None


@pytest.mark.unit
def test_row_to_golden_question_sets_authored_at() -> None:
    row = make_row()
    question = row_to_golden_question(
        row, question_id="direct_lookup-001", authored_at=AUTHORED_AT
    )
    assert question.authored_at == "2026-08-22"


@pytest.mark.unit
def test_row_to_golden_question_raises_finalize_error_on_invalid_row() -> None:
    row = make_row(question_type="unanswerable", reference_clause_ids="1:a")
    with pytest.raises(FinalizeError):
        row_to_golden_question(
            row, question_id="unanswerable-001", authored_at=AUTHORED_AT
        )


@pytest.mark.unit
def test_next_sequence_numbers_continues_from_existing_max(tmp_path: Path) -> None:
    path = tmp_path / "direct_lookup.jsonl"
    path.write_text(
        '{"question_id": "direct_lookup-001"}\n{"question_id": "direct_lookup-003"}\n',
        encoding="utf-8",
    )
    assert next_sequence_numbers(tmp_path)["direct_lookup"] == 4


@pytest.mark.unit
def test_next_sequence_numbers_returns_empty_for_no_files(tmp_path: Path) -> None:
    assert next_sequence_numbers(tmp_path) == {}


@pytest.mark.unit
def test_finalize_rows_skips_unapproved_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(finalize_module, "GOLDEN_SET_DIR", tmp_path)
    rows = [make_row(row_id="1-00", approved="")]
    _, new_by_type = finalize_rows(rows, authored_at=AUTHORED_AT)
    assert new_by_type == {}


@pytest.mark.unit
def test_finalize_rows_skips_already_finalized_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(finalize_module, "GOLDEN_SET_DIR", tmp_path)
    rows = [make_row(row_id="1-00", finalized_question_id="direct_lookup-001")]
    _, new_by_type = finalize_rows(rows, authored_at=AUTHORED_AT)
    assert new_by_type == {}


@pytest.mark.unit
def test_finalize_rows_assigns_sequential_ids_within_one_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(finalize_module, "GOLDEN_SET_DIR", tmp_path)
    rows = [make_row(row_id="1-00"), make_row(row_id="1-01")]
    updated_rows, new_by_type = finalize_rows(rows, authored_at=AUTHORED_AT)
    ids = [q.question_id for q in new_by_type["direct_lookup"]]
    assert ids == ["direct_lookup-001", "direct_lookup-002"]
    assert [row["finalized_question_id"] for row in updated_rows] == ids


@pytest.mark.unit
def test_finalize_rows_continues_numbering_from_existing_golden_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(finalize_module, "GOLDEN_SET_DIR", tmp_path)
    (tmp_path / "direct_lookup.jsonl").write_text(
        '{"question_id": "direct_lookup-005"}\n', encoding="utf-8"
    )
    rows = [make_row(row_id="1-00")]
    _, new_by_type = finalize_rows(rows, authored_at=AUTHORED_AT)
    assert new_by_type["direct_lookup"][0].question_id == "direct_lookup-006"


@pytest.mark.unit
def test_finalize_rows_returns_valid_golden_questions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(finalize_module, "GOLDEN_SET_DIR", tmp_path)
    rows = [make_row(row_id="1-00")]
    _, new_by_type = finalize_rows(rows, authored_at=AUTHORED_AT)
    assert isinstance(new_by_type["direct_lookup"][0], GoldenQuestion)
