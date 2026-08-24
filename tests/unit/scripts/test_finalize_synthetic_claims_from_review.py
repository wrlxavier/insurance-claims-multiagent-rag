"""Tests for promoting approved draft rows into the synthetic claim set [M2-04].

Uses `monkeypatch` to redirect [scripts.finalize_synthetic_claims_from_review.
CLAIMS_PATH] at a tmp_path file for every test that touches it, so these tests
never read or write the repo's real `data/synthetic_claims/claims.jsonl`.
"""

import sys
from datetime import date
from pathlib import Path

import pytest
import scripts.finalize_synthetic_claims_from_review as finalize_module
from scripts.finalize_synthetic_claims_from_review import (
    DEFAULT_CSV_PATH,
    FinalizeError,
    _parse_args,
    finalize_rows,
    is_approved,
    next_sequence_numbers,
    row_to_synthetic_claim,
)

from infrastructure.evaluation.synthetic_claims_schema import SyntheticClaim

AUTHORED_AT = date(2026, 8, 24)


def make_row(**overrides: str) -> dict[str, str]:
    """Build a valid, approved, unfinalized draft CSV row, overridable per test."""
    fields = {
        "row_id": "sc-compat-1-00",
        "document_id": "1",
        "narrative": "Meu carro bateu na traseira de outro veículo ontem.",
        "reference_clause_ids": "1:coverage",
        "expected_verdict": "compatible",
        "missing_fact_type": "",
        "review_correction": "",
        "authored_at": "",
        "approved": "Y",
        "finalized_claim_id": "",
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
def test_row_to_synthetic_claim_splits_reference_clause_ids() -> None:
    row = make_row(reference_clause_ids="1:a;1:b")
    claim = row_to_synthetic_claim(
        row, claim_id="compatible-001", authored_at=AUTHORED_AT
    )
    assert claim.reference_clause_ids == ["1:a", "1:b"]


@pytest.mark.unit
def test_row_to_synthetic_claim_sets_authored_at() -> None:
    row = make_row()
    claim = row_to_synthetic_claim(
        row, claim_id="compatible-001", authored_at=AUTHORED_AT
    )
    assert claim.authored_at == "2026-08-24"


@pytest.mark.unit
def test_row_to_synthetic_claim_prefers_row_authored_at() -> None:
    row = make_row(authored_at="2026-01-01")
    claim = row_to_synthetic_claim(
        row, claim_id="compatible-001", authored_at=AUTHORED_AT
    )
    assert claim.authored_at == "2026-01-01"


@pytest.mark.unit
def test_row_to_synthetic_claim_maps_review_correction_to_notes() -> None:
    row = make_row(review_correction="ajustei a data")
    claim = row_to_synthetic_claim(
        row, claim_id="compatible-001", authored_at=AUTHORED_AT
    )
    assert claim.notes == "ajustei a data"


@pytest.mark.unit
def test_row_to_synthetic_claim_raises_finalize_error_on_invalid_row() -> None:
    # insufficient_information requires missing_fact_type -- omitted here.
    row = make_row(expected_verdict="insufficient_information", missing_fact_type="")
    with pytest.raises(FinalizeError):
        row_to_synthetic_claim(
            row, claim_id="insufficient_information-001", authored_at=AUTHORED_AT
        )


@pytest.mark.unit
def test_next_sequence_numbers_continues_from_existing_max(tmp_path: Path) -> None:
    path = tmp_path / "claims.jsonl"
    path.write_text(
        '{"claim_id": "compatible-001"}\n{"claim_id": "compatible-003"}\n',
        encoding="utf-8",
    )
    assert next_sequence_numbers(path)["compatible"] == 4


@pytest.mark.unit
def test_next_sequence_numbers_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert next_sequence_numbers(tmp_path / "does-not-exist.jsonl") == {}


@pytest.mark.unit
def test_finalize_rows_skips_unapproved_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(finalize_module, "CLAIMS_PATH", tmp_path / "claims.jsonl")
    rows = [make_row(row_id="sc-compat-1-00", approved="")]
    _, new_claims, pii_blocked = finalize_rows(rows, authored_at=AUTHORED_AT)
    assert new_claims == []
    assert pii_blocked == []


@pytest.mark.unit
def test_finalize_rows_skips_already_finalized_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(finalize_module, "CLAIMS_PATH", tmp_path / "claims.jsonl")
    rows = [make_row(row_id="sc-compat-1-00", finalized_claim_id="compatible-001")]
    _, new_claims, _ = finalize_rows(rows, authored_at=AUTHORED_AT)
    assert new_claims == []


@pytest.mark.unit
def test_finalize_rows_blocks_rows_flagged_by_pii_safety_net(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(finalize_module, "CLAIMS_PATH", tmp_path / "claims.jsonl")
    rows = [make_row(row_id="sc-compat-1-00", narrative="a placa era ABC1234")]
    _, new_claims, pii_blocked = finalize_rows(rows, authored_at=AUTHORED_AT)
    assert new_claims == []
    assert pii_blocked == ["sc-compat-1-00"]


@pytest.mark.unit
def test_finalize_rows_assigns_sequential_ids_within_one_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(finalize_module, "CLAIMS_PATH", tmp_path / "claims.jsonl")
    rows = [make_row(row_id="sc-compat-1-00"), make_row(row_id="sc-compat-2-00")]
    updated_rows, new_claims, _ = finalize_rows(rows, authored_at=AUTHORED_AT)
    ids = [claim.claim_id for claim in new_claims]
    assert ids == ["compatible-001", "compatible-002"]
    assert [row["finalized_claim_id"] for row in updated_rows] == ids


@pytest.mark.unit
def test_finalize_rows_continues_numbering_from_existing_claims_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    claims_path = tmp_path / "claims.jsonl"
    claims_path.write_text('{"claim_id": "compatible-005"}\n', encoding="utf-8")
    monkeypatch.setattr(finalize_module, "CLAIMS_PATH", claims_path)
    rows = [make_row(row_id="sc-compat-1-00")]
    _, new_claims, _ = finalize_rows(rows, authored_at=AUTHORED_AT)
    assert new_claims[0].claim_id == "compatible-006"


@pytest.mark.unit
def test_finalize_rows_returns_valid_synthetic_claims(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(finalize_module, "CLAIMS_PATH", tmp_path / "claims.jsonl")
    rows = [make_row(row_id="sc-compat-1-00")]
    _, new_claims, _ = finalize_rows(rows, authored_at=AUTHORED_AT)
    assert isinstance(new_claims[0], SyntheticClaim)


@pytest.mark.unit
def test_parse_args_defaults_to_default_csv_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["finalize_synthetic_claims_from_review.py"])
    assert _parse_args().csv == DEFAULT_CSV_PATH


@pytest.mark.unit
def test_parse_args_accepts_explicit_csv_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finalize_synthetic_claims_from_review.py",
            "--csv",
            "eval/other.csv",
        ],
    )
    assert _parse_args().csv == Path("eval/other.csv")
