"""The domain exception hierarchy [M5-01]."""

import pytest

from domain.errors import (
    CitationRequiredError,
    DomainError,
    InvalidCnpjError,
    InvalidValueObjectError,
    InvariantViolationError,
)


@pytest.mark.unit
def test_value_object_errors_are_also_value_errors() -> None:
    # A caller can `except ValueError` a bad constructor argument the
    # conventional way, or catch the precise domain type.
    assert issubclass(InvalidCnpjError, ValueError)
    assert issubclass(InvalidCnpjError, InvalidValueObjectError)
    assert issubclass(InvalidValueObjectError, DomainError)


@pytest.mark.unit
def test_invariant_errors_are_domain_errors_but_not_value_errors() -> None:
    assert issubclass(InvariantViolationError, DomainError)
    assert not issubclass(InvariantViolationError, ValueError)


@pytest.mark.unit
def test_citation_required_error_carries_the_assessment_id() -> None:
    error = CitationRequiredError("assessment-42")

    assert isinstance(error, InvariantViolationError)
    assert error.assessment_id == "assessment-42"
    assert "assessment-42" in str(error)
