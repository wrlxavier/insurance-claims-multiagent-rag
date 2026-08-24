"""Tests for the product/claim mismatch set validation script [M2-05]."""

import pytest
from scripts.validate_product_claim_mismatch import (
    ProductClaimMismatchValidationError,
    check_claim_ids_unique,
    check_document_ids_exist,
    check_reference_clause_ids_exist,
    check_targets_non_casco_documents,
    check_verdict_is_always_incompatible,
)

from infrastructure.evaluation.synthetic_claims_schema import SyntheticClaim


def make_claim(**overrides: object) -> SyntheticClaim:
    """Build a valid mismatch SyntheticClaim, overridable per test."""
    fields: dict[str, object] = {
        "schema_version": "v1",
        "claim_id": "mismatch-001",
        "document_id": "21",
        "narrative": "Bateram no meu carro e amassaram a lateral toda.",
        "reference_clause_ids": ["21:objetivo"],
        "expected_verdict": "incompatible",
        "missing_fact_type": None,
        "notes": "",
        "authored_at": "2026-08-24",
    }
    fields.update(overrides)
    return SyntheticClaim.model_validate(fields)


@pytest.mark.unit
def test_check_document_ids_exist_rejects_unknown_document() -> None:
    claim = make_claim(document_id="999")
    with pytest.raises(
        ProductClaimMismatchValidationError, match="unknown document_id"
    ):
        check_document_ids_exist([claim], {"21", "22"})


@pytest.mark.unit
def test_check_document_ids_exist_accepts_known_document() -> None:
    claim = make_claim(document_id="21")
    check_document_ids_exist([claim], {"21", "22"})


@pytest.mark.unit
def test_check_reference_clause_ids_exist_rejects_unknown_clause() -> None:
    claim = make_claim(reference_clause_ids=["21:missing"])
    with pytest.raises(ProductClaimMismatchValidationError, match="unknown clause_id"):
        check_reference_clause_ids_exist([claim], {"21:objetivo"})


@pytest.mark.unit
def test_check_reference_clause_ids_exist_accepts_known_clauses() -> None:
    claim = make_claim(reference_clause_ids=["21:objetivo"])
    check_reference_clause_ids_exist([claim], {"21:objetivo"})


@pytest.mark.unit
def test_check_claim_ids_unique_rejects_duplicate() -> None:
    first = make_claim(claim_id="mismatch-001")
    second = make_claim(claim_id="mismatch-001")
    with pytest.raises(ProductClaimMismatchValidationError, match="duplicate claim_id"):
        check_claim_ids_unique([first, second])


@pytest.mark.unit
def test_check_claim_ids_unique_accepts_distinct_ids() -> None:
    first = make_claim(claim_id="mismatch-001")
    second = make_claim(claim_id="mismatch-002")
    check_claim_ids_unique([first, second])


# --- check_targets_non_casco_documents ------------------------------------


@pytest.mark.unit
def test_check_targets_non_casco_documents_rejects_casco_target() -> None:
    claim = make_claim(document_id="1")
    with pytest.raises(ProductClaimMismatchValidationError, match="non-CASCO document"):
        check_targets_non_casco_documents([claim], {"1": "CASCO", "21": "RCF-A"})


@pytest.mark.unit
def test_check_targets_non_casco_documents_accepts_non_casco_targets() -> None:
    claim = make_claim(document_id="21")
    check_targets_non_casco_documents([claim], {"1": "CASCO", "21": "RCF-A"})


@pytest.mark.unit
def test_check_targets_non_casco_documents_rejects_unknown_document() -> None:
    claim = make_claim(document_id="999")
    with pytest.raises(ProductClaimMismatchValidationError, match="non-CASCO document"):
        check_targets_non_casco_documents([claim], {"1": "CASCO"})


# --- check_verdict_is_always_incompatible ---------------------------------


@pytest.mark.unit
def test_check_verdict_is_always_incompatible_rejects_other_verdicts() -> None:
    claim = make_claim(expected_verdict="compatible", reference_clause_ids=["21:a"])
    with pytest.raises(
        ProductClaimMismatchValidationError, match="must be 'incompatible'"
    ):
        check_verdict_is_always_incompatible([claim])


@pytest.mark.unit
def test_check_verdict_is_always_incompatible_accepts_incompatible() -> None:
    claim = make_claim(expected_verdict="incompatible")
    check_verdict_is_always_incompatible([claim])
