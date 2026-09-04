"""The RunAssessment use case -- the worker's half of the queue [M5-05]."""

from __future__ import annotations

import pytest

from application.assessment_job import AssessmentJob, JobStatus
from application.assessment_record import AssessmentRecord, AssessmentStatus
from application.errors import PermanentAssessmentError, TransientAssessmentError
from application.use_cases.run_assessment import RunAssessment
from tests.unit.application.fakes import (
    FakeClaimAssessmentOrchestrator,
    FixedClock,
    abstain_result,
    make_job,
    make_orchestrator_result,
    make_uow_factory,
)


class _RateLimitError(Exception):
    """Stand-in for a provider 429 -- the classifier keys off this type."""


def _build(
    *,
    job: AssessmentJob,
    orchestrator: FakeClaimAssessmentOrchestrator | None = None,
    max_attempts: int = 3,
) -> tuple[RunAssessment, dict[str, AssessmentJob], dict[str, AssessmentRecord]]:
    job_store = {job.assessment_id: job}
    record_store: dict[str, AssessmentRecord] = {}
    use_case = RunAssessment(
        clock=FixedClock(),
        orchestrator=orchestrator or FakeClaimAssessmentOrchestrator(),
        uow_factory=make_uow_factory(record_store, None, job_store),
        is_transient=lambda exc: isinstance(exc, _RateLimitError),
        max_attempts=max_attempts,
    )
    return use_case, job_store, record_store


@pytest.mark.unit
def test_happy_path_writes_the_record_and_succeeds_the_job() -> None:
    run, jobs, records = _build(job=make_job(assessment_id="a1"))

    run("a1")

    assert jobs["a1"].status is JobStatus.SUCCEEDED
    assert jobs["a1"].attempts == 1
    assert jobs["a1"].failure is None
    record = records["a1"]
    assert isinstance(record, AssessmentRecord)
    assert record.status is AssessmentStatus.AWAITING_REVIEW


@pytest.mark.unit
def test_abstain_run_still_succeeds_with_a_zero_citation_record() -> None:
    orch = FakeClaimAssessmentOrchestrator(
        start_result=abstain_result(awaiting_review=True)
    )
    run, jobs, records = _build(job=make_job(assessment_id="a1"), orchestrator=orch)

    run("a1")

    assert jobs["a1"].status is JobStatus.SUCCEEDED
    assert records["a1"].citations == ()


@pytest.mark.unit
def test_transient_failure_with_attempts_left_returns_to_pending_and_raises() -> None:
    orch = FakeClaimAssessmentOrchestrator(raise_on_start=_RateLimitError("429"))
    run, jobs, records = _build(job=make_job(assessment_id="a1"), orchestrator=orch)

    with pytest.raises(TransientAssessmentError):
        run("a1")

    job = jobs["a1"]
    assert job.status is JobStatus.PENDING
    assert job.attempts == 1
    assert job.failure is not None
    assert job.failure.kind == "transient"
    assert job.failure.error_type == "_RateLimitError"
    assert records == {}


@pytest.mark.unit
def test_transient_failure_on_the_last_attempt_dead_letters() -> None:
    orch = FakeClaimAssessmentOrchestrator(raise_on_start=_RateLimitError("429"))
    run, jobs, _ = _build(
        job=make_job(assessment_id="a1", attempts=2),
        orchestrator=orch,
        max_attempts=3,
    )

    with pytest.raises(TransientAssessmentError):
        run("a1")

    assert jobs["a1"].status is JobStatus.FAILED
    assert jobs["a1"].attempts == 3


@pytest.mark.unit
def test_real_failure_dead_letters_immediately_as_permanent() -> None:
    orch = FakeClaimAssessmentOrchestrator(raise_on_start=ValueError("bad claim"))
    run, jobs, _ = _build(job=make_job(assessment_id="a1"), orchestrator=orch)

    with pytest.raises(PermanentAssessmentError):
        run("a1")

    job = jobs["a1"]
    assert job.status is JobStatus.FAILED
    assert job.failure is not None and job.failure.kind == "permanent"
    assert "bad claim" in job.failure.message


@pytest.mark.unit
def test_a_run_that_does_not_pause_is_a_permanent_failure() -> None:
    orch = FakeClaimAssessmentOrchestrator(
        start_result=make_orchestrator_result(awaiting_review=False)
    )
    run, jobs, records = _build(job=make_job(assessment_id="a1"), orchestrator=orch)

    with pytest.raises(PermanentAssessmentError, match="did not pause"):
        run("a1")

    assert jobs["a1"].status is JobStatus.FAILED
    assert records == {}


@pytest.mark.unit
@pytest.mark.parametrize("status", [JobStatus.SUCCEEDED, JobStatus.FAILED])
def test_a_redelivered_terminal_job_is_a_no_op(status: JobStatus) -> None:
    orch = FakeClaimAssessmentOrchestrator(raise_on_start=RuntimeError("must not run"))
    run, jobs, _ = _build(
        job=make_job(assessment_id="a1", status=status, attempts=1),
        orchestrator=orch,
    )

    run("a1")  # does not raise -- the orchestrator is never called

    assert jobs["a1"].status is status
    assert orch.started == []


@pytest.mark.unit
def test_an_unknown_job_is_a_permanent_failure() -> None:
    run, _, _ = _build(job=make_job(assessment_id="a1"))

    with pytest.raises(PermanentAssessmentError, match="no assessment job"):
        run("ghost")
