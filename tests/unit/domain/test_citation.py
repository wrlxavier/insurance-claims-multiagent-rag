"""The Citation value object [M5-01]."""

import dataclasses

import pytest

from domain.citation import Citation
from domain.clause_classification import ClauseType
from domain.susep_process import SusepProcess


def _citation(**overrides: object) -> Citation:
    fields: dict[str, object] = {
        "clause_id": "15414610650202459:2.1",
        "document_id": "1",
        "susep_process": SusepProcess("15414.610650/2024-59"),
        "clause_type": ClauseType.COVERAGE,
        "excerpt": "A cobertura compreende colisao, incendio e roubo.",
        "relevance_score": 0.83,
    }
    fields.update(overrides)
    return Citation(**fields)  # type: ignore[arg-type]


@pytest.mark.unit
def test_accepts_a_full_citation() -> None:
    citation = _citation()

    assert citation.clause_type is ClauseType.COVERAGE
    assert citation.relevance_score == 0.83


@pytest.mark.unit
def test_relevance_score_defaults_to_zero() -> None:
    # A structurally co-retrieved exclusion the ranker missed carries 0.0.
    citation = Citation(
        clause_id="c1",
        document_id="1",
        susep_process=SusepProcess("15414.610650/2024-59"),
        clause_type=ClauseType.EXCLUSION,
        excerpt="Estao excluidos os eventos decorrentes de...",
    )

    assert citation.relevance_score == 0.0


@pytest.mark.unit
@pytest.mark.parametrize("field", ["clause_id", "document_id", "excerpt"])
def test_rejects_an_empty_string_field(field: str) -> None:
    with pytest.raises(ValueError):
        _citation(**{field: ""})


@pytest.mark.unit
def test_rejects_a_negative_relevance_score() -> None:
    with pytest.raises(ValueError):
        _citation(relevance_score=-0.1)


@pytest.mark.unit
def test_rejects_a_bare_string_clause_type() -> None:
    with pytest.raises(ValueError):
        _citation(clause_type="coverage")


@pytest.mark.unit
def test_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _citation().relevance_score = 0.9  # type: ignore[misc]
