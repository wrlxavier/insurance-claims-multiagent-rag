"""What ``GET /v1/assessments/{id}`` returns across the whole lifecycle [M5-05].

Before M5-05 the read endpoint could only answer once an ``AssessmentRecord``
existed. With the queue, the id in the 202 must resolve immediately -- while the
run is still ``pending`` or ``running``, and after it has ``failed``. This is the
unified projection the presentation layer renders: a lifecycle ``status``, the
failure cause when there is one, and the full ``AssessmentRecord`` once the graph
has produced it.

``status`` is the five-value union a caller polls:

- ``pending`` / ``running`` -- from the [application.assessment_job.AssessmentJob]
  (queued, or a worker is on it);
- ``failed`` -- the job dead-lettered; ``error`` carries the preserved cause;
- ``awaiting_review`` / ``decided`` -- from the ``AssessmentRecord`` once it
  exists (the job, now ``SUCCEEDED``, stops being the source of truth).

Standard library and domain/application types only (enforced by
tests/architecture/test_layer_boundaries.py).
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from application.assessment_job import AssessmentJob, JobStatus
from application.assessment_record import AssessmentRecord, AssessmentStatus

ReadStatus = Literal["pending", "running", "awaiting_review", "decided", "failed"]

_RECORD_STATUS_TO_READ: dict[AssessmentStatus, ReadStatus] = {
    AssessmentStatus.AWAITING_REVIEW: "awaiting_review",
    AssessmentStatus.DECIDED: "decided",
}

# A job that is SUCCEEDED but whose record we somehow could not read is shown as
# still running rather than inventing a terminal state -- an atomic write makes
# this unreachable in practice.
_JOB_STATUS_TO_READ: dict[JobStatus, ReadStatus] = {
    JobStatus.PENDING: "pending",
    JobStatus.RUNNING: "running",
    JobStatus.SUCCEEDED: "running",
    JobStatus.FAILED: "failed",
}


@dataclass(frozen=True)
class AssessmentReadModel:
    """One assessment as a caller sees it: lifecycle status + the record if ready."""

    assessment_id: str
    claim_id: str
    status: ReadStatus
    created_at: datetime
    error: str | None = None
    record: AssessmentRecord | None = None

    @classmethod
    def from_record(cls, record: AssessmentRecord) -> "AssessmentReadModel":
        """The completed view -- ``awaiting_review`` or ``decided``."""
        return cls(
            assessment_id=record.assessment_id,
            claim_id=record.claim_id,
            status=_RECORD_STATUS_TO_READ[record.status],
            created_at=record.created_at,
            record=record,
        )

    @classmethod
    def from_job(cls, job: AssessmentJob) -> "AssessmentReadModel":
        """The in-flight or failed view -- no record yet."""
        return cls(
            assessment_id=job.assessment_id,
            claim_id=job.claim_id,
            status=_JOB_STATUS_TO_READ[job.status],
            created_at=job.created_at,
            error=job.failure.message if job.failure is not None else None,
        )
