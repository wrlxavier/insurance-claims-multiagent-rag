"""The SQLAlchemy ``AssessmentJobRepository`` -- [M5-05].

Implements [application.ports.assessment_job_repository.AssessmentJobRepository]
over the ``assessment_job`` table. Constructed with a ``Session``:

* ``add`` / ``update`` are used through a
  [infrastructure.database.unit_of_work.SqlAlchemyUnitOfWork] -- this class
  flushes, never commits;
* ``get`` works on any ``Session`` (the read endpoints build a short-lived
  instance on a fresh session, like ``SqlAlchemyAssessmentRepository``).

Row <-> aggregate translation is entirely in
[infrastructure.database.assessment_job_mapper].
"""

from sqlalchemy.orm import Session

from application.assessment_job import AssessmentJob
from infrastructure.database.assessment_job_mapper import job_to_row, row_to_job
from infrastructure.database.models import AssessmentJobRow


class SqlAlchemyAssessmentJobRepository:
    """Persist and read ``AssessmentJob`` aggregates in Postgres."""

    def __init__(self, session: Session) -> None:
        """Bind the repository to a session (the unit of work owns its lifecycle)."""
        self._session = session

    def add(self, job: AssessmentJob) -> None:
        """Persist a new job. A duplicate id raises ``IntegrityError`` on flush."""
        self._session.add(job_to_row(job))
        self._session.flush()

    def update(self, job: AssessmentJob) -> None:
        """Replace the stored job with the same ``assessment_id``.

        Raises:
            KeyError: no job exists for ``job.assessment_id`` -- matching the
                in-memory fake's contract.
        """
        if self._session.get(AssessmentJobRow, job.assessment_id) is None:
            raise KeyError(f"assessment job {job.assessment_id!r} does not exist")
        self._session.merge(job_to_row(job))
        self._session.flush()

    def get(self, assessment_id: str) -> AssessmentJob | None:
        """Return the job for ``assessment_id``, or ``None`` if there is none."""
        row = self._session.get(AssessmentJobRow, assessment_id)
        return row_to_job(row) if row is not None else None
