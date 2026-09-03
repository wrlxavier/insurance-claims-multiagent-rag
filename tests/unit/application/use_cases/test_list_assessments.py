"""The ListAssessments use case [M5-02]."""

from datetime import timedelta

import pytest

from application.assessment_record import AssessmentRecord, AssessmentStatus
from application.use_cases.list_assessments import ListAssessments
from domain.human_decision import DecisionOutcome, HumanDecision
from tests.unit.application.fakes import (
    FIXED_NOW,
    InMemoryAssessmentRepository,
    make_record,
)


def _decided(
    assessment_id: str, claim_id: str, offset_minutes: int
) -> AssessmentRecord:
    decision = HumanDecision(
        assessment_id=assessment_id,
        decision=DecisionOutcome.APPROVE,
        decided_at=FIXED_NOW,
    )
    return make_record(
        assessment_id=assessment_id,
        claim_id=claim_id,
        created_at=FIXED_NOW + timedelta(minutes=offset_minutes),
        status=AssessmentStatus.DECIDED,
        decision=decision,
    )


def _awaiting(
    assessment_id: str, claim_id: str, offset_minutes: int
) -> AssessmentRecord:
    return make_record(
        assessment_id=assessment_id,
        claim_id=claim_id,
        created_at=FIXED_NOW + timedelta(minutes=offset_minutes),
    )


@pytest.mark.unit
def test_empty_store_returns_empty_tuple() -> None:
    list_assessments = ListAssessments(assessments=InMemoryAssessmentRepository({}))

    assert list_assessments() == ()


@pytest.mark.unit
def test_returns_records_newest_first() -> None:
    store = {
        "a-1": _awaiting("a-1", "claim-1", 0),
        "a-2": _awaiting("a-2", "claim-1", 30),
        "a-3": _awaiting("a-3", "claim-2", 10),
    }
    list_assessments = ListAssessments(assessments=InMemoryAssessmentRepository(store))

    ids = [record.assessment_id for record in list_assessments()]

    assert ids == ["a-2", "a-3", "a-1"]


@pytest.mark.unit
def test_ties_broken_by_assessment_id() -> None:
    store = {
        "a-2": _awaiting("a-2", "claim-1", 0),
        "a-1": _awaiting("a-1", "claim-1", 0),
    }
    list_assessments = ListAssessments(assessments=InMemoryAssessmentRepository(store))

    ids = [record.assessment_id for record in list_assessments()]

    assert ids == ["a-1", "a-2"]


@pytest.mark.unit
def test_filter_by_claim_id() -> None:
    store = {
        "a-1": _awaiting("a-1", "claim-1", 0),
        "a-2": _awaiting("a-2", "claim-2", 10),
    }
    list_assessments = ListAssessments(assessments=InMemoryAssessmentRepository(store))

    result = list_assessments(claim_id="claim-2")

    assert [record.assessment_id for record in result] == ["a-2"]


@pytest.mark.unit
def test_filter_by_status() -> None:
    store = {
        "a-1": _awaiting("a-1", "claim-1", 0),
        "a-2": _decided("a-2", "claim-1", 10),
    }
    list_assessments = ListAssessments(assessments=InMemoryAssessmentRepository(store))

    result = list_assessments(status=AssessmentStatus.DECIDED)

    assert [record.assessment_id for record in result] == ["a-2"]


@pytest.mark.unit
def test_limit_and_offset_page_the_result() -> None:
    store = {f"a-{i}": _awaiting(f"a-{i}", "claim-1", i) for i in range(5)}
    list_assessments = ListAssessments(assessments=InMemoryAssessmentRepository(store))

    page = list_assessments(limit=2, offset=1)

    # newest-first is a-4, a-3, a-2, a-1, a-0 -> offset 1, limit 2
    assert [record.assessment_id for record in page] == ["a-3", "a-2"]


@pytest.mark.unit
def test_combined_claim_and_status_filter() -> None:
    store = {
        "a-1": _decided("a-1", "claim-1", 0),
        "a-2": _decided("a-2", "claim-2", 10),
        "a-3": _awaiting("a-3", "claim-1", 20),
    }
    list_assessments = ListAssessments(assessments=InMemoryAssessmentRepository(store))

    result = list_assessments(claim_id="claim-1", status=AssessmentStatus.DECIDED)

    assert [record.assessment_id for record in result] == ["a-1"]


@pytest.mark.unit
def test_rejects_a_non_positive_limit() -> None:
    list_assessments = ListAssessments(assessments=InMemoryAssessmentRepository({}))

    with pytest.raises(ValueError, match="limit must be positive"):
        list_assessments(limit=0)
