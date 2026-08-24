"""Tests for the synthetic-claim-set validation script [M2-04]."""

import pytest
from pydantic import ValidationError
from scripts.validate_synthetic_claims import (
    SyntheticClaimsValidationError,
    check_claim_ids_unique,
    check_document_ids_exist,
    check_reference_clause_ids_exist,
)

from infrastructure.evaluation.synthetic_claims_schema import SyntheticClaim


def make_claim(**overrides: object) -> SyntheticClaim:
    """Build a valid compatible SyntheticClaim, overridable per test."""
    fields: dict[str, object] = {
        "schema_version": "v1",
        "claim_id": "compatible-001",
        "document_id": "1",
        "narrative": "Meu carro bateu na traseira de outro veículo ontem.",
        "reference_clause_ids": ["1:coverage"],
        "expected_verdict": "compatible",
        "missing_fact_type": None,
        "notes": "",
        "authored_at": "2026-08-24",
    }
    fields.update(overrides)
    return SyntheticClaim.model_validate(fields)


@pytest.mark.unit
def test_insufficient_information_claim_requires_missing_fact_type() -> None:
    with pytest.raises(ValidationError, match="must set missing_fact_type"):
        make_claim(expected_verdict="insufficient_information", missing_fact_type=None)


@pytest.mark.unit
def test_insufficient_information_claim_accepts_missing_fact_type() -> None:
    claim = make_claim(
        expected_verdict="insufficient_information",
        missing_fact_type="ambito_geografico",
    )
    assert claim.missing_fact_type is not None
    assert claim.missing_fact_type.value == "ambito_geografico"


@pytest.mark.unit
def test_non_insufficient_information_claim_rejects_missing_fact_type() -> None:
    with pytest.raises(ValidationError, match="must not set missing_fact_type"):
        make_claim(expected_verdict="compatible", missing_fact_type="ambito_geografico")


@pytest.mark.unit
def test_claim_requires_at_least_one_reference_clause() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        make_claim(reference_clause_ids=[])


@pytest.mark.unit
def test_check_document_ids_exist_rejects_unknown_document() -> None:
    claim = make_claim(document_id="999")
    with pytest.raises(SyntheticClaimsValidationError, match="unknown document_id"):
        check_document_ids_exist([claim], {"1", "2"})


@pytest.mark.unit
def test_check_document_ids_exist_accepts_known_document() -> None:
    claim = make_claim(document_id="1")
    check_document_ids_exist([claim], {"1", "2"})


@pytest.mark.unit
def test_check_reference_clause_ids_exist_rejects_unknown_clause() -> None:
    claim = make_claim(reference_clause_ids=["1:missing"])
    with pytest.raises(SyntheticClaimsValidationError, match="unknown clause_id"):
        check_reference_clause_ids_exist([claim], {"1:coverage"})


@pytest.mark.unit
def test_check_reference_clause_ids_exist_accepts_known_clauses() -> None:
    claim = make_claim(reference_clause_ids=["1:coverage"])
    check_reference_clause_ids_exist([claim], {"1:coverage"})


@pytest.mark.unit
def test_check_claim_ids_unique_rejects_duplicate() -> None:
    first = make_claim(claim_id="compatible-001")
    second = make_claim(claim_id="compatible-001")
    with pytest.raises(SyntheticClaimsValidationError, match="duplicate claim_id"):
        check_claim_ids_unique([first, second])


@pytest.mark.unit
def test_check_claim_ids_unique_accepts_distinct_ids() -> None:
    first = make_claim(claim_id="compatible-001")
    second = make_claim(claim_id="compatible-002")
    check_claim_ids_unique([first, second])
