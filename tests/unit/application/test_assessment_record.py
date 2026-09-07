"""The AssessmentRecord aggregate and its invariants [M5-02]."""

from datetime import datetime

import pytest

from application.assessment_record import AssessmentRecord, AssessmentStatus
from domain.assessment import Assessment
from domain.errors import CitationRequiredError
from domain.human_decision import DecisionOutcome, HumanDecision
from domain.verdict import Verdict
from tests.unit.application.fakes import (
    FIXED_NOW,
    abstain_result,
    make_orchestrator_result,
    make_record,
)


def _decision(
    assessment_id: str = "assessment-1", **overrides: object
) -> HumanDecision:
    fields: dict[str, object] = {
        "assessment_id": assessment_id,
        "decision": DecisionOutcome.APPROVE,
        "decided_at": FIXED_NOW,
        "notes": "",
        "edited_assessment": None,
    }
    fields.update(overrides)
    return HumanDecision(**fields)  # type: ignore[arg-type]


@pytest.mark.unit
def test_grounded_record_projects_to_a_domain_assessment() -> None:
    record = make_record()

    assessment = record.as_domain_assessment()

    assert isinstance(assessment, Assessment)
    assert assessment.assessment_id == record.assessment_id
    assert assessment.claim_id == record.claim_id
    assert assessment.verdict is record.verdict
    assert assessment.citations == record.citations
    assert record.is_grounded


@pytest.mark.unit
def test_abstain_record_constructs_but_does_not_project() -> None:
    record = make_record(
        verdict=Verdict.INSUFFICIENT_INFORMATION,
        citations=(),
        confidence=0.2,
        context_sufficient=False,
    )

    assert not record.is_grounded
    with pytest.raises(CitationRequiredError):
        record.as_domain_assessment()


@pytest.mark.unit
def test_decided_record_requires_a_decision() -> None:
    with pytest.raises(ValueError, match="DECIDED record must carry a decision"):
        make_record(status=AssessmentStatus.DECIDED, decision=None)


@pytest.mark.unit
def test_awaiting_review_record_rejects_a_decision() -> None:
    with pytest.raises(ValueError, match="must not carry a decision"):
        make_record(status=AssessmentStatus.AWAITING_REVIEW, decision=_decision())


@pytest.mark.unit
def test_decision_must_reference_this_assessment() -> None:
    with pytest.raises(ValueError, match="reference the assessment it settled"):
        make_record(
            assessment_id="assessment-1",
            status=AssessmentStatus.DECIDED,
            decision=_decision(assessment_id="assessment-2"),
        )


@pytest.mark.unit
def test_rejects_a_naive_created_at() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        make_record(created_at=datetime(2026, 9, 3, 12, 0))


@pytest.mark.unit
def test_rejects_confidence_out_of_range() -> None:
    with pytest.raises(ValueError, match="confidence must be in"):
        make_record(confidence=1.5)


@pytest.mark.unit
def test_rejects_a_non_verdict_verdict() -> None:
    with pytest.raises(ValueError, match="must be a domain.verdict.Verdict"):
        make_record(verdict="compatible")


@pytest.mark.unit
def test_from_orchestrator_result_copies_the_system_opinion() -> None:
    flag_result = make_orchestrator_result(
        consistency_flags=(),
        missing_information=("vehicle_info",),
        context_sufficient=True,
        awaiting_review=True,
    )

    record = AssessmentRecord.from_orchestrator_result(
        flag_result,
        assessment_id="assessment-9",
        claim_id="claim-9",
        created_at=FIXED_NOW,
        status=AssessmentStatus.AWAITING_REVIEW,
    )

    assert record.assessment_id == "assessment-9"
    assert record.claim_id == "claim-9"
    assert record.verdict is flag_result.verdict
    assert record.reasoning == flag_result.reasoning
    assert record.recommended_action == flag_result.recommended_action
    assert record.citations == flag_result.citations
    assert record.confidence == flag_result.confidence
    assert record.missing_information == ("vehicle_info",)
    assert record.status is AssessmentStatus.AWAITING_REVIEW
    assert record.decision is None


@pytest.mark.unit
def test_from_orchestrator_result_carries_a_decision_when_decided() -> None:
    record = AssessmentRecord.from_orchestrator_result(
        abstain_result(awaiting_review=False),
        assessment_id="assessment-1",
        claim_id="claim-1",
        created_at=FIXED_NOW,
        status=AssessmentStatus.DECIDED,
        decision=_decision(decision=DecisionOutcome.REJECT, notes="fora de vigencia"),
    )

    assert record.status is AssessmentStatus.DECIDED
    assert record.decision is not None
    assert record.decision.decision is DecisionOutcome.REJECT
    assert record.citations == ()
