"""The PolicyClause entity [M5-01]."""

import dataclasses

import pytest

from domain.clause_classification import ClauseType
from domain.policy_clause import PolicyClause
from domain.susep_process import SusepProcess


def _clause(**overrides: object) -> PolicyClause:
    fields: dict[str, object] = {
        "clause_id": "15414610650202459:2.1",
        "susep_process": SusepProcess("15414.610650/2024-59"),
        "document_id": "1",
        "clause_type": ClauseType.COVERAGE,
        "text": "A cobertura compreende colisao, incendio e roubo.",
        "heading": "2.1 Coberturas",
    }
    fields.update(overrides)
    return PolicyClause(**fields)  # type: ignore[arg-type]


@pytest.mark.unit
def test_accepts_a_full_clause() -> None:
    clause = _clause()

    assert clause.clause_type is ClauseType.COVERAGE
    assert clause.heading == "2.1 Coberturas"


@pytest.mark.unit
def test_heading_is_optional() -> None:
    assert _clause(heading="").heading == ""


@pytest.mark.unit
@pytest.mark.parametrize("field", ["clause_id", "document_id", "text"])
def test_rejects_an_empty_string_field(field: str) -> None:
    with pytest.raises(ValueError):
        _clause(**{field: ""})


@pytest.mark.unit
def test_rejects_a_bare_string_clause_type() -> None:
    with pytest.raises(ValueError):
        _clause(clause_type="coverage")


@pytest.mark.unit
def test_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _clause().text = "x"  # type: ignore[misc]
