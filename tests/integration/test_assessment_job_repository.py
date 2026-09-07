"""The SQLAlchemy ``AssessmentJobRepository`` against a real Postgres -- [M5-05].

Covers what only the database proves: the job aggregate round-trips through the
``assessment_job`` table (pending and failed-with-cause), timestamps come back
tz-aware, ``update`` of a missing row raises, a duplicate ``add`` hits the primary
key, and the ``status`` CHECK rejects an out-of-vocabulary value.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session

from application.assessment_job import AssessmentJob, JobFailure, JobStatus
from infrastructure.database.assessment_job_repository import (
    SqlAlchemyAssessmentJobRepository,
)

pytestmark = pytest.mark.integration

_NOW = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)


def _job(**overrides: object) -> AssessmentJob:
    fields: dict[str, object] = {
        "assessment_id": "a1",
        "claim_id": "c1",
        "raw_text": "Bati o carro na traseira de outro veiculo.",
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


def test_round_trips_a_pending_job(db_session: Session) -> None:
    repo = SqlAlchemyAssessmentJobRepository(db_session)
    job = _job()
    repo.add(job)
    db_session.commit()

    db_session.expunge_all()
    assert repo.get("a1") == job


def test_round_trips_a_failed_job_with_its_cause(db_session: Session) -> None:
    repo = SqlAlchemyAssessmentJobRepository(db_session)
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
    repo.add(job)
    db_session.commit()
    db_session.expunge_all()

    restored = repo.get("a1")
    assert restored == job
    assert restored is not None and restored.failure is not None
    assert restored.failure.failed_at == _NOW


def test_update_of_a_missing_job_raises_keyerror(db_session: Session) -> None:
    repo = SqlAlchemyAssessmentJobRepository(db_session)
    with pytest.raises(KeyError):
        repo.update(_job(assessment_id="ghost"))


def test_a_duplicate_add_hits_the_primary_key(db_session: Session) -> None:
    repo = SqlAlchemyAssessmentJobRepository(db_session)
    repo.add(_job())
    with pytest.raises(IntegrityError):
        repo.add(_job())
    db_session.rollback()


def test_the_status_check_rejects_an_unknown_value(db_session: Session) -> None:
    SqlAlchemyAssessmentJobRepository(db_session).add(_job())
    db_session.flush()
    with pytest.raises(DBAPIError):
        db_session.execute(
            text(
                "UPDATE assessment_job SET status = 'weird' WHERE assessment_id = 'a1'"
            )
        )
    db_session.rollback()


def test_update_advances_the_lifecycle(db_session: Session) -> None:
    repo = SqlAlchemyAssessmentJobRepository(db_session)
    repo.add(_job())
    db_session.commit()

    running = _job(status=JobStatus.RUNNING, attempts=1)
    repo.update(running)
    db_session.commit()
    db_session.expunge_all()

    assert repo.get("a1") == running
