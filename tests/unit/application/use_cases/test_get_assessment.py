"""The GetAssessment use case [M5-02]."""

import pytest

from application.assessment_record import AssessmentRecord, AssessmentStatus
from application.errors import AssessmentNotFoundError
from application.use_cases.get_assessment import GetAssessment
from domain.human_decision import DecisionOutcome, HumanDecision
from domain.verdict import Verdict
from tests.unit.application.fakes import (
    FIXED_NOW,
    InMemoryAssessmentRepository,
    make_record,
)


@pytest.mark.unit
def test_returns_the_stored_record() -> None:
    store = {"assessment-1": make_record(assessment_id="assessment-1")}
    get_assessment = GetAssessment(assessments=InMemoryAssessmentRepository(store))

    record = get_assessment("assessment-1")

    assert isinstance(record, AssessmentRecord)
    assert record.assessment_id == "assessment-1"


@pytest.mark.unit
def test_unknown_id_raises_not_found_carrying_the_id() -> None:
    get_assessment = GetAssessment(assessments=InMemoryAssessmentRepository({}))

    with pytest.raises(AssessmentNotFoundError) as excinfo:
        get_assessment("missing")

    assert excinfo.value.assessment_id == "missing"


@pytest.mark.unit
def test_returns_a_decided_record_with_its_decision() -> None:
    decision = HumanDecision(
        assessment_id="assessment-1",
        decision=DecisionOutcome.APPROVE,
        decided_at=FIXED_NOW,
    )
    store = {
        "assessment-1": make_record(status=AssessmentStatus.DECIDED, decision=decision)
    }
    get_assessment = GetAssessment(assessments=InMemoryAssessmentRepository(store))

    record = get_assessment("assessment-1")

    assert record.status is AssessmentStatus.DECIDED
    assert record.decision is decision


@pytest.mark.unit
def test_serves_an_abstain_record_without_raising() -> None:
    store = {
        "assessment-1": make_record(
            verdict=Verdict.INSUFFICIENT_INFORMATION,
            citations=(),
            confidence=0.2,
            context_sufficient=False,
        )
    }
    get_assessment = GetAssessment(assessments=InMemoryAssessmentRepository(store))

    record = get_assessment("assessment-1")

    assert record.citations == ()
    assert not record.is_grounded
