"""The assessment_job row <-> aggregate mapper [M5-05]."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from application.assessment_job import AssessmentJob, JobFailure, JobStatus
from infrastructure.database.assessment_job_mapper import job_to_row, row_to_job

_NOW = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)


def _job(**overrides: object) -> AssessmentJob:
    fields: dict[str, object] = {
        "assessment_id": "a1",
        "claim_id": "c1",
        "raw_text": "Bati o carro.",
        "policy_ref": "15414.610650/2024-59",
        "submitted_at": _NOW,
        "status": JobStatus.PENDING,
        "attempts": 0,
        "created_at": _NOW,
        "updated_at": _NOW,
        "failure": None,
    }
    fields.update(overrides)
    return AssessmentJob(**fields)  # type: ignore[arg-type]


@pytest.mark.unit
def test_round_trips_a_pending_job() -> None:
    job = _job()
    assert row_to_job(job_to_row(job)) == job


@pytest.mark.unit
def test_round_trips_a_failed_job_with_its_cause() -> None:
    job = _job(
        status=JobStatus.FAILED,
        attempts=3,
        policy_ref=None,
        failure=JobFailure(
            kind="transient",
            error_type="RateLimitError",
            message="429 Too Many Requests",
            failed_at=_NOW,
        ),
    )
    restored = row_to_job(job_to_row(job))
    assert restored == job
    assert restored.failure is not None
    assert restored.failure.failed_at == _NOW


@pytest.mark.unit
def test_a_naive_timestamp_from_the_row_fails_loudly() -> None:
    row = job_to_row(_job())
    row.submitted_at = datetime(2026, 9, 4, 12, 0)  # naive
    with pytest.raises(ValueError, match="timezone-aware"):
        row_to_job(row)
