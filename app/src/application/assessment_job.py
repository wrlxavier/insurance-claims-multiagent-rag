"""The queued assessment run -- its lifecycle before a recommendation exists [M5-05].

A 207-page policy is not an HTTP request. ``POST /v1/assessments`` no longer runs
the graph in the handler: it persists an ``AssessmentJob`` in ``PENDING`` and
hands the id back behind a 202. A Redis worker picks the job up, runs the graph
to the human checkpoint, and -- on success -- writes the
[application.assessment_record.AssessmentRecord] the read endpoints already serve.

``AssessmentJob`` is a deliberately separate aggregate from ``AssessmentRecord``:
the record is invariant-laden (a non-empty verdict, prose, a >=1-citation
projection) and cannot represent a claim that has not been assessed yet. The job
carries only what the worker needs to (re)build the domain ``Claim`` plus the run
state a caller polls: ``status``, ``attempts`` and, on failure, the preserved
``JobFailure`` cause.

``submitted_at`` is stamped once, at submission, and is stable across retries --
it feeds ``Claim.submitted_at``, which the consistency checks compare the loss
date against, so a retry must not move it.

Standard library and domain/application types only (enforced by
tests/architecture/test_layer_boundaries.py).
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Literal


class JobStatus(Enum):
    """Where a queued assessment run sits.

    ``SUCCEEDED`` means the graph paused at the human checkpoint and an
    ``AssessmentRecord`` now exists -- that record (``AWAITING_REVIEW`` ->
    ``DECIDED``) is the source of truth from then on; the job row is kept for its
    attempt history. ``FAILED`` is the dead-letter state: the cause is in
    ``failure`` and, operationally, in RQ's ``FailedJobRegistry``.
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class JobFailure:
    """Why a run did not complete -- preserved on the job, readable through the API.

    ``kind`` is the transient/real split the DoD asks for: ``transient`` failures
    (a provider rate limit, a 5xx, a dropped connection) are retried with
    backoff; ``permanent`` failures (a malformed claim, a contract breach, a bug)
    go straight to the dead-letter with no wasted retries.
    """

    kind: Literal["transient", "permanent"]
    error_type: str
    message: str
    failed_at: datetime

    def __post_init__(self) -> None:
        """Reject an out-of-vocabulary kind, an empty type, or a naive timestamp."""
        if self.kind not in ("transient", "permanent"):
            raise ValueError(
                f"JobFailure.kind must be 'transient' or 'permanent', got {self.kind!r}"
            )
        if not self.error_type:
            raise ValueError("JobFailure.error_type must not be empty")
        if self.failed_at.tzinfo is None or self.failed_at.utcoffset() is None:
            raise ValueError("JobFailure.failed_at must be timezone-aware")


@dataclass(frozen=True)
class AssessmentJob:
    """One queued assessment run across its lifecycle -- the persisted unit."""

    assessment_id: str
    claim_id: str
    raw_text: str
    policy_ref: str | None
    submitted_at: datetime
    status: JobStatus
    attempts: int
    created_at: datetime
    updated_at: datetime
    failure: JobFailure | None = None

    def __post_init__(self) -> None:
        """Enforce the non-empty fields, the types, and the status/failure pairing."""
        for name in ("assessment_id", "claim_id", "raw_text"):
            if not getattr(self, name):
                raise ValueError(f"AssessmentJob.{name} must not be empty")
        if not isinstance(self.status, JobStatus):
            raise ValueError(
                f"AssessmentJob.status must be a JobStatus, got {self.status!r}"
            )
        if self.attempts < 0:
            raise ValueError(
                f"AssessmentJob.attempts must not be negative, got {self.attempts}"
            )
        for name in ("submitted_at", "created_at", "updated_at"):
            value: datetime = getattr(self, name)
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"AssessmentJob.{name} must be timezone-aware")
        if self.failure is not None and self.status in (
            JobStatus.RUNNING,
            JobStatus.SUCCEEDED,
        ):
            raise ValueError(
                f"a {self.status.value!r} job must not carry a failure cause"
            )

    @property
    def is_terminal(self) -> bool:
        """Whether the worker should treat this job as done (idempotent redelivery)."""
        return self.status in (JobStatus.SUCCEEDED, JobStatus.FAILED)
