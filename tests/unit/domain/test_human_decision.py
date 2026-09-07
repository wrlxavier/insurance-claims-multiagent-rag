"""The HumanDecision entity and its invariants [M5-01]."""

import dataclasses
from datetime import UTC, datetime

import pytest

from domain.assessment import Assessment
from domain.citation import Citation
from domain.clause_classification import ClauseType
from domain.errors import DecisionMustReferenceAssessmentError
from domain.human_decision import DecisionOutcome, HumanDecision
from domain.susep_process import SusepProcess
from domain.verdict import Verdict

_ASSESSMENT_ID = "assessment-1"


def _assessment(assessment_id: str = _ASSESSMENT_ID, **overrides: object) -> Assessment:
    fields: dict[str, object] = {
        "assessment_id": assessment_id,
        "claim_id": "claim-1",
        "verdict": Verdict.INCOMPATIBLE,
        "reasoning": "O evento e uma colisao; a apolice cobre apenas roubo.",
        "citations": (
            Citation(
                clause_id="15414610650202459:3.2",
                document_id="1",
                susep_process=SusepProcess("15414.610650/2024-59"),
                clause_type=ClauseType.EXCLUSION,
                excerpt="Estao excluidos os danos por colisao.",
            ),
        ),
        "confidence": 0.6,
        "recommended_action": "Recusar o sinistro.",
    }
    fields.update(overrides)
    return Assessment(**fields)  # type: ignore[arg-type]


def _decision(**overrides: object) -> HumanDecision:
    fields: dict[str, object] = {
        "assessment_id": _ASSESSMENT_ID,
        "decision": DecisionOutcome.APPROVE,
        "decided_at": datetime(2026, 9, 3, 12, 0, tzinfo=UTC),
        "notes": "",
        "edited_assessment": None,
    }
    fields.update(overrides)
    return HumanDecision(**fields)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize("outcome", [DecisionOutcome.APPROVE, DecisionOutcome.REJECT])
def test_accepts_approve_and_reject_without_an_edit(outcome: DecisionOutcome) -> None:
    decision = _decision(decision=outcome)

    assert decision.decision is outcome
    assert decision.edited_assessment is None


@pytest.mark.unit
def test_rejects_a_decision_with_no_assessment_reference() -> None:
    with pytest.raises(DecisionMustReferenceAssessmentError):
        _decision(assessment_id="")


@pytest.mark.unit
def test_edit_requires_an_edited_assessment() -> None:
    with pytest.raises(ValueError):
        _decision(decision=DecisionOutcome.EDIT)


@pytest.mark.unit
def test_non_edit_must_not_carry_an_edited_assessment() -> None:
    with pytest.raises(ValueError):
        _decision(decision=DecisionOutcome.APPROVE, edited_assessment=_assessment())


@pytest.mark.unit
def test_rejects_a_bare_string_decision() -> None:
    with pytest.raises(ValueError):
        _decision(decision="approve")


@pytest.mark.unit
def test_rejects_a_naive_decided_at() -> None:
    with pytest.raises(ValueError):
        _decision(decided_at=datetime(2026, 1, 1))  # noqa: DTZ001


@pytest.mark.unit
def test_edited_assessment_must_revise_the_referenced_assessment() -> None:
    with pytest.raises(ValueError):
        _decision(
            decision=DecisionOutcome.EDIT,
            edited_assessment=_assessment(assessment_id="a-different-one"),
        )


@pytest.mark.unit
def test_edit_round_trips_the_revised_verdict() -> None:
    revised = _assessment(verdict=Verdict.COMPATIBLE, recommended_action="Aprovar.")
    decision = _decision(decision=DecisionOutcome.EDIT, edited_assessment=revised)

    assert decision.edited_assessment is not None
    assert decision.edited_assessment.verdict is Verdict.COMPATIBLE


@pytest.mark.unit
def test_decision_outcome_values_match_the_graph_literal() -> None:
    assert {o.value for o in DecisionOutcome} == {"approve", "edit", "reject"}


@pytest.mark.unit
def test_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _decision().notes = "x"  # type: ignore[misc]
