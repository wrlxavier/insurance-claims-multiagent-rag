"""The Assessment entity and its invariants [M5-01]."""

import dataclasses

import pytest

from domain.assessment import Assessment
from domain.citation import Citation
from domain.clause_classification import ClauseType
from domain.errors import CitationRequiredError, VerdictNotPermittedError
from domain.susep_process import SusepProcess
from domain.verdict import Verdict


def _citation() -> Citation:
    return Citation(
        clause_id="15414610650202459:2.1",
        document_id="1",
        susep_process=SusepProcess("15414.610650/2024-59"),
        clause_type=ClauseType.COVERAGE,
        excerpt="A cobertura compreende colisao, incendio e roubo.",
        relevance_score=0.83,
    )


def _assessment(**overrides: object) -> Assessment:
    fields: dict[str, object] = {
        "assessment_id": "assessment-1",
        "claim_id": "claim-1",
        "verdict": Verdict.COMPATIBLE,
        "reasoning": "O evento descrito e uma colisao, coberta pela clausula 2.1.",
        "citations": (_citation(),),
        "confidence": 0.72,
        "recommended_action": "Encaminhar para analise humana.",
    }
    fields.update(overrides)
    return Assessment(**fields)  # type: ignore[arg-type]


@pytest.mark.unit
def test_accepts_a_full_assessment() -> None:
    assessment = _assessment()

    assert assessment.verdict is Verdict.COMPATIBLE
    assert len(assessment.citations) == 1


@pytest.mark.unit
def test_rejects_an_assessment_with_no_citations() -> None:
    with pytest.raises(CitationRequiredError) as excinfo:
        _assessment(citations=())

    assert excinfo.value.assessment_id == "assessment-1"


@pytest.mark.unit
def test_the_at_least_one_citation_rule_is_unconditional() -> None:
    # Even an insufficient_information verdict must carry a citation; the
    # graph's abstain-on-empty-retrieval output is not a persistable
    # Assessment.
    assessment = _assessment(verdict=Verdict.INSUFFICIENT_INFORMATION)

    assert assessment.verdict is Verdict.INSUFFICIENT_INFORMATION
    with pytest.raises(CitationRequiredError):
        _assessment(verdict=Verdict.INSUFFICIENT_INFORMATION, citations=())


@pytest.mark.unit
def test_rejects_a_verdict_that_is_not_a_verdict_member() -> None:
    with pytest.raises(VerdictNotPermittedError):
        _assessment(verdict="compatible")


@pytest.mark.unit
@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_rejects_a_confidence_outside_the_unit_interval(confidence: float) -> None:
    with pytest.raises(ValueError):
        _assessment(confidence=confidence)


@pytest.mark.unit
@pytest.mark.parametrize(
    "field", ["assessment_id", "claim_id", "reasoning", "recommended_action"]
)
def test_rejects_an_empty_string_field(field: str) -> None:
    with pytest.raises(ValueError):
        _assessment(**{field: ""})


@pytest.mark.unit
def test_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _assessment().confidence = 0.9  # type: ignore[misc]
