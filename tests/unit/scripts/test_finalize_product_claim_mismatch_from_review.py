"""Tests for promoting approved draft rows into the product/claim mismatch set [M2-05].

Uses `monkeypatch` to redirect [scripts.finalize_product_claim_mismatch_from_review.
MISMATCH_CLAIMS_PATH] at a tmp_path file for every test that touches it, so these
tests never read or write the repo's real
`data/synthetic_claims/product_claim_mismatch.jsonl`.
"""

import sys
from datetime import date
from pathlib import Path

import pytest
import scripts.finalize_product_claim_mismatch_from_review as finalize_module
from scripts.finalize_product_claim_mismatch_from_review import (
    DEFAULT_CSV_PATH,
    FinalizeError,
    _parse_args,
    finalize_rows,
    is_approved,
    next_sequence_number,
    row_to_mismatch_claim,
)

from infrastructure.evaluation.synthetic_claims_schema import SyntheticClaim

AUTHORED_AT = date(2026, 8, 24)


def make_row(**overrides: str) -> dict[str, str]:
    """Build a valid, approved, unfinalized draft CSV row, overridable per test."""
    fields = {
        "row_id": "mismatch-21-00",
        "document_id": "21",
        "narrative": "Bateram no meu carro e amassaram a lateral toda.",
        "reference_clause_ids": "21:objetivo",
        "expected_verdict": "incompatible",
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
def test_row_to_mismatch_claim_splits_reference_clause_ids() -> None:
    row = make_row(reference_clause_ids="21:a;21:b")
    claim = row_to_mismatch_claim(row, claim_id="mismatch-001", authored_at=AUTHORED_AT)
    assert claim.reference_clause_ids == ["21:a", "21:b"]


@pytest.mark.unit
def test_row_to_mismatch_claim_sets_authored_at() -> None:
    row = make_row()
    claim = row_to_mismatch_claim(row, claim_id="mismatch-001", authored_at=AUTHORED_AT)
    assert claim.authored_at == "2026-08-24"


@pytest.mark.unit
def test_row_to_mismatch_claim_prefers_row_authored_at() -> None:
    row = make_row(authored_at="2026-01-01")
    claim = row_to_mismatch_claim(row, claim_id="mismatch-001", authored_at=AUTHORED_AT)
    assert claim.authored_at == "2026-01-01"


@pytest.mark.unit
def test_row_to_mismatch_claim_maps_review_correction_to_notes() -> None:
    row = make_row(review_correction="ajustei o relato")
    claim = row_to_mismatch_claim(row, claim_id="mismatch-001", authored_at=AUTHORED_AT)
    assert claim.notes == "ajustei o relato"

    assert claim.missing_fact_type is None


@pytest.mark.unit
def test_row_to_mismatch_claim_raises_finalize_error_on_invalid_row() -> None:
    # SyntheticClaim requires at least one reference_clause_ids entry.
    row = make_row(reference_clause_ids="")
    with pytest.raises(FinalizeError):
        row_to_mismatch_claim(row, claim_id="mismatch-001", authored_at=AUTHORED_AT)


@pytest.mark.unit
def test_next_sequence_number_continues_from_existing_max(tmp_path: Path) -> None:
    path = tmp_path / "product_claim_mismatch.jsonl"
    path.write_text(
        '{"claim_id": "mismatch-001"}\n{"claim_id": "mismatch-003"}\n',
        encoding="utf-8",
    )
    assert next_sequence_number(path) == 4


@pytest.mark.unit
def test_next_sequence_number_starts_at_one_for_missing_file(tmp_path: Path) -> None:
    assert next_sequence_number(tmp_path / "does-not-exist.jsonl") == 1


@pytest.mark.unit
def test_finalize_rows_skips_unapproved_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        finalize_module, "MISMATCH_CLAIMS_PATH", tmp_path / "mismatch.jsonl"
    )
    rows = [make_row(approved="")]
    _, new_claims, pii_blocked = finalize_rows(rows, authored_at=AUTHORED_AT)
    assert new_claims == []
    assert pii_blocked == []


@pytest.mark.unit
def test_finalize_rows_skips_already_finalized_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        finalize_module, "MISMATCH_CLAIMS_PATH", tmp_path / "mismatch.jsonl"
    )
    rows = [make_row(finalized_claim_id="mismatch-001")]
    _, new_claims, _ = finalize_rows(rows, authored_at=AUTHORED_AT)
    assert new_claims == []


@pytest.mark.unit
def test_finalize_rows_blocks_rows_flagged_by_pii_safety_net(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        finalize_module, "MISMATCH_CLAIMS_PATH", tmp_path / "mismatch.jsonl"
    )
    rows = [make_row(row_id="mismatch-21-00", narrative="a placa era ABC1234")]
    _, new_claims, pii_blocked = finalize_rows(rows, authored_at=AUTHORED_AT)
    assert new_claims == []
    assert pii_blocked == ["mismatch-21-00"]


@pytest.mark.unit
def test_finalize_rows_assigns_sequential_mismatch_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        finalize_module, "MISMATCH_CLAIMS_PATH", tmp_path / "mismatch.jsonl"
    )
    rows = [make_row(row_id="mismatch-21-00"), make_row(row_id="mismatch-22-00")]
    updated_rows, new_claims, _ = finalize_rows(rows, authored_at=AUTHORED_AT)
    ids = [claim.claim_id for claim in new_claims]
    assert ids == ["mismatch-001", "mismatch-002"]
    assert [row["finalized_claim_id"] for row in updated_rows] == ids


@pytest.mark.unit
def test_finalize_rows_continues_numbering_from_existing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    claims_path = tmp_path / "mismatch.jsonl"
    claims_path.write_text('{"claim_id": "mismatch-005"}\n', encoding="utf-8")
    monkeypatch.setattr(finalize_module, "MISMATCH_CLAIMS_PATH", claims_path)
    rows = [make_row()]
    _, new_claims, _ = finalize_rows(rows, authored_at=AUTHORED_AT)
    assert new_claims[0].claim_id == "mismatch-006"


@pytest.mark.unit
def test_finalize_rows_returns_valid_synthetic_claims(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        finalize_module, "MISMATCH_CLAIMS_PATH", tmp_path / "mismatch.jsonl"
    )
    rows = [make_row()]
    _, new_claims, _ = finalize_rows(rows, authored_at=AUTHORED_AT)
    assert isinstance(new_claims[0], SyntheticClaim)


@pytest.mark.unit
def test_finalize_rows_never_sets_missing_fact_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mismatch claims are always `incompatible`, never `insufficient_information`,
    so missing_fact_type must never be set even if a row's CSV column carries one."""
    monkeypatch.setattr(
        finalize_module, "MISMATCH_CLAIMS_PATH", tmp_path / "mismatch.jsonl"
    )
    rows = [make_row()]
    _, new_claims, _ = finalize_rows(rows, authored_at=AUTHORED_AT)
    assert new_claims[0].missing_fact_type is None


@pytest.mark.unit
def test_parse_args_defaults_to_default_csv_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["finalize_product_claim_mismatch_from_review.py"])
    assert _parse_args().csv == DEFAULT_CSV_PATH


@pytest.mark.unit
def test_parse_args_accepts_explicit_csv_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finalize_product_claim_mismatch_from_review.py",
            "--csv",
            "eval/other.csv",
        ],
    )
    assert _parse_args().csv == Path("eval/other.csv")
