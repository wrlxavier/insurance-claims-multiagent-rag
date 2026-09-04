"""The SubmitClaim use case [M5-02, M5-05].

Since M5-05, ``SubmitClaim`` does not run the graph -- it persists an
``AssessmentJob`` in ``PENDING`` and hands the id to the queue. The run itself is
``RunAssessment``'s (see ``test_run_assessment.py``).
"""

import pytest

from application.assessment_job import AssessmentJob, JobStatus
from application.assessment_record import AssessmentRecord
from application.use_cases.submit_claim import SubmitClaim
from tests.unit.application.fakes import (
    FIXED_NOW,
    SUSEP,
    FakeAssessmentQueue,
    FixedClock,
    NaiveClock,
    SequentialIds,
    make_uow_factory,
)

_NARRATIVE = "Bati o carro em uma colisao na avenida no dia 05/01/2026."


def _build(
    *,
    clock: object | None = None,
    ids: SequentialIds | None = None,
) -> tuple[SubmitClaim, dict[str, AssessmentJob], FakeAssessmentQueue]:
    job_store: dict[str, AssessmentJob] = {}
    queue = FakeAssessmentQueue()
    record_store: dict[str, AssessmentRecord] = {}
    use_case = SubmitClaim(
        clock=clock or FixedClock(),  # type: ignore[arg-type]
        queue=queue,
        uow_factory=make_uow_factory(record_store, None, job_store),
        new_id=ids or SequentialIds("id"),
    )
    return use_case, job_store, queue


@pytest.mark.unit
def test_happy_path_persists_a_pending_job_and_enqueues_it() -> None:
    submit_claim, job_store, queue = _build()

    job = submit_claim(raw_text=_NARRATIVE)

    assert isinstance(job, AssessmentJob)
    assert job.status is JobStatus.PENDING
    assert job.attempts == 0
    assert job.raw_text == _NARRATIVE
    assert job.submitted_at == FIXED_NOW == job.created_at == job.updated_at
    assert job_store[job.assessment_id] == job
    assert queue.enqueued == [job.assessment_id]


@pytest.mark.unit
def test_mints_distinct_claim_and_assessment_ids_from_new_id() -> None:
    ids = SequentialIds("id")
    submit_claim, _, _ = _build(ids=ids)

    job = submit_claim(raw_text=_NARRATIVE)

    assert ids.issued == ["id-1", "id-2"]
    assert job.claim_id == "id-1"
    assert job.assessment_id == "id-2"


@pytest.mark.unit
def test_honours_an_explicit_claim_id() -> None:
    ids = SequentialIds("id")
    submit_claim, _, _ = _build(ids=ids)

    job = submit_claim(raw_text=_NARRATIVE, claim_id="claim-external-7")

    assert job.claim_id == "claim-external-7"
    assert job.assessment_id == "id-1"
    assert ids.issued == ["id-1"]


@pytest.mark.unit
def test_policy_ref_is_stored_on_the_job_in_canonical_form() -> None:
    submit_claim, job_store, _ = _build()

    job = submit_claim(raw_text=_NARRATIVE, policy_ref=SUSEP)

    assert job.policy_ref == SUSEP.value
    assert job_store[job.assessment_id].policy_ref == SUSEP.value


@pytest.mark.unit
def test_empty_narrative_is_rejected_and_nothing_is_persisted_or_enqueued() -> None:
    submit_claim, job_store, queue = _build()

    with pytest.raises(ValueError, match="raw_text must not be empty"):
        submit_claim(raw_text="")

    assert job_store == {}
    assert queue.enqueued == []


@pytest.mark.unit
def test_a_naive_clock_is_rejected_before_persisting() -> None:
    submit_claim, job_store, queue = _build(clock=NaiveClock())

    with pytest.raises(ValueError, match="timezone-aware"):
        submit_claim(raw_text=_NARRATIVE)

    assert job_store == {}
    assert queue.enqueued == []


@pytest.mark.unit
def test_the_job_is_committed_before_it_is_enqueued() -> None:
    """A caller polling GET must always find the row the 202 promised."""
    job_store: dict[str, AssessmentJob] = {}
    record_store: dict[str, AssessmentRecord] = {}
    seen: list[bool] = []

    class _RecordingQueue:
        def enqueue(self, assessment_id: str) -> None:
            seen.append(assessment_id in job_store)

    submit_claim = SubmitClaim(
        clock=FixedClock(),
        queue=_RecordingQueue(),
        uow_factory=make_uow_factory(record_store, None, job_store),
        new_id=SequentialIds("id"),
    )
    submit_claim(raw_text=_NARRATIVE)

    assert seen == [True]
