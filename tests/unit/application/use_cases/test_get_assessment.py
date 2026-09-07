"""The GetAssessment use case [M5-02, M5-05].

Since M5-05 it returns an ``AssessmentReadModel`` spanning the whole lifecycle:
the record once it exists, otherwise the job's ``pending`` / ``running`` /
``failed`` state.
"""

from datetime import UTC, datetime

import pytest

from application.assessment_job import JobFailure, JobStatus
from application.assessment_record import AssessmentStatus
from application.errors import AssessmentNotFoundError
from application.use_cases.get_assessment import GetAssessment
from domain.human_decision import DecisionOutcome, HumanDecision
from domain.verdict import Verdict
from tests.unit.application.fakes import (
    FIXED_NOW,
    InMemoryAssessmentJobRepository,
    InMemoryAssessmentRepository,
    make_job,
    make_record,
)


def _use_case(
    *,
    records: dict[str, object] | None = None,
    jobs: dict[str, object] | None = None,
) -> GetAssessment:
    return GetAssessment(
        assessments=InMemoryAssessmentRepository(dict(records or {})),  # type: ignore[arg-type]
        jobs=InMemoryAssessmentJobRepository(dict(jobs or {})),  # type: ignore[arg-type]
    )


@pytest.mark.unit
def test_returns_the_stored_record_as_a_completed_view() -> None:
    get_assessment = _use_case(
        records={"assessment-1": make_record(assessment_id="assessment-1")}
    )

    model = get_assessment("assessment-1")

    assert model.status == "awaiting_review"
    assert model.record is not None
    assert model.record.assessment_id == "assessment-1"
    assert model.error is None


@pytest.mark.unit
def test_returns_a_decided_record_with_its_decision() -> None:
    decision = HumanDecision(
        assessment_id="assessment-1",
        decision=DecisionOutcome.APPROVE,
        decided_at=FIXED_NOW,
    )
    get_assessment = _use_case(
        records={
            "assessment-1": make_record(
                status=AssessmentStatus.DECIDED, decision=decision
            )
        }
    )

    model = get_assessment("assessment-1")

    assert model.status == "decided"
    assert model.record is not None and model.record.decision is decision


@pytest.mark.unit
def test_serves_an_abstain_record_without_raising() -> None:
    get_assessment = _use_case(
        records={
            "assessment-1": make_record(
                verdict=Verdict.INSUFFICIENT_INFORMATION,
                citations=(),
                confidence=0.2,
                context_sufficient=False,
            )
        }
    )

    model = get_assessment("assessment-1")

    assert model.record is not None and not model.record.is_grounded


@pytest.mark.unit
def test_a_pending_job_reads_as_pending_with_no_record() -> None:
    get_assessment = _use_case(jobs={"a1": make_job(assessment_id="a1")})

    model = get_assessment("a1")

    assert model.status == "pending"
    assert model.record is None
    assert model.error is None


@pytest.mark.unit
def test_a_failed_job_carries_the_preserved_cause() -> None:
    job = make_job(
        assessment_id="a1",
        status=JobStatus.FAILED,
        attempts=3,
        failure=JobFailure(
            kind="permanent",
            error_type="ValueError",
            message="claim narrative was rejected",
            failed_at=datetime(2026, 9, 4, tzinfo=UTC),
        ),
    )
    get_assessment = _use_case(jobs={"a1": job})

    model = get_assessment("a1")

    assert model.status == "failed"
    assert model.error == "claim narrative was rejected"


@pytest.mark.unit
def test_the_record_wins_once_it_exists_even_if_a_job_row_lingers() -> None:
    get_assessment = _use_case(
        records={"a1": make_record(assessment_id="a1")},
        jobs={"a1": make_job(assessment_id="a1", status=JobStatus.SUCCEEDED)},
    )

    assert get_assessment("a1").status == "awaiting_review"


@pytest.mark.unit
def test_unknown_id_raises_not_found_carrying_the_id() -> None:
    with pytest.raises(AssessmentNotFoundError) as excinfo:
        _use_case()("missing")

    assert excinfo.value.assessment_id == "missing"
