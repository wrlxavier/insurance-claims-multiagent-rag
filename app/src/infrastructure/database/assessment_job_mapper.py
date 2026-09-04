"""Map the queued-run aggregate to and from its database row -- [M5-05].

Pure functions, no session, between [application.assessment_job.AssessmentJob]
(plus its [application.assessment_job.JobFailure]) and
[infrastructure.database.models.AssessmentJobRow]. Mirrors
``assessment_mapper.py``: the ``JobStatus`` enum and the ``failure`` value object
are written as their canonical JSON shape and rebuilt on read (the domain
constructors re-validate, so a corrupted row fails loudly here rather than
several layers up).
"""

from collections.abc import Mapping
from datetime import datetime
from typing import Literal, cast

from application.assessment_job import AssessmentJob, JobFailure, JobStatus
from infrastructure.database.models import AssessmentJobRow


def _as_aware(value: datetime) -> datetime:
    """Guard: a ``timestamptz`` column must never hand back a naive datetime."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"expected a timezone-aware datetime, got {value!r}")
    return value


def _require_str(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise TypeError(f"expected a string for {key!r}, got {value!r}")
    return value


def _failure_to_json(failure: JobFailure) -> dict[str, object]:
    return {
        "kind": failure.kind,
        "error_type": failure.error_type,
        "message": failure.message,
        "failed_at": failure.failed_at.isoformat(),
    }


def _failure_from_json(data: Mapping[str, object]) -> JobFailure:
    return JobFailure(
        kind=cast(Literal["transient", "permanent"], _require_str(data, "kind")),
        error_type=_require_str(data, "error_type"),
        message=_require_str(data, "message"),
        failed_at=_as_aware(datetime.fromisoformat(_require_str(data, "failed_at"))),
    )


def job_to_row(job: AssessmentJob) -> AssessmentJobRow:
    """Build the ``assessment_job`` row for a job aggregate."""
    return AssessmentJobRow(
        assessment_id=job.assessment_id,
        claim_id=job.claim_id,
        raw_text=job.raw_text,
        policy_ref=job.policy_ref,
        submitted_at=job.submitted_at,
        status=job.status.value,
        attempts=job.attempts,
        failure=_failure_to_json(job.failure) if job.failure is not None else None,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def row_to_job(row: AssessmentJobRow) -> AssessmentJob:
    """Rebuild a job aggregate from its row (the aggregate re-validates)."""
    return AssessmentJob(
        assessment_id=row.assessment_id,
        claim_id=row.claim_id,
        raw_text=row.raw_text,
        policy_ref=row.policy_ref,
        submitted_at=_as_aware(row.submitted_at),
        status=JobStatus(row.status),
        attempts=row.attempts,
        created_at=_as_aware(row.created_at),
        updated_at=_as_aware(row.updated_at),
        failure=_failure_from_json(row.failure) if row.failure is not None else None,
    )
