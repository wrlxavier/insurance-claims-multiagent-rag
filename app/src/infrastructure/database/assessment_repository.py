"""The SQLAlchemy ``AssessmentRepository`` -- [M5-03].

Implements [application.ports.assessment_repository.AssessmentRepository] over
the ``assessment`` / ``human_decision`` tables. Constructed with a ``Session``:

* the write methods (``add`` / ``update``) are used through a
  [infrastructure.database.unit_of_work.SqlAlchemyUnitOfWork], which owns the
  transaction -- this class flushes, never commits;
* the read methods (``get`` / ``list``) work on any ``Session``, so the [M5-04]
  composition root can build a short-lived read instance on a fresh session
  outside the unit of work ("two bindings", ``docs/ARCHITECTURE.md`` M5-02).

The row <-> aggregate translation is entirely in
[infrastructure.database.assessment_mapper].
"""

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from application.assessment_record import AssessmentRecord, AssessmentStatus
from infrastructure.database.assessment_mapper import record_to_rows, rows_to_record
from infrastructure.database.models import AssessmentRow, HumanDecisionRow

_DEFAULT_LIMIT = 50


class SqlAlchemyAssessmentRepository:
    """Persist and query ``AssessmentRecord`` aggregates in Postgres."""

    def __init__(self, session: Session) -> None:
        """Bind the repository to a session (the unit of work owns its lifecycle)."""
        self._session = session

    def add(self, record: AssessmentRecord) -> None:
        """Persist a new record. A duplicate id raises ``IntegrityError`` on flush."""
        assessment_row, decision_row = record_to_rows(record)
        self._session.add(assessment_row)
        if decision_row is not None:
            self._session.add(decision_row)
        self._session.flush()

    def update(self, record: AssessmentRecord) -> None:
        """Replace the stored record with the same ``assessment_id``.

        Raises:
            KeyError: no record exists for ``record.assessment_id`` -- matching
                the in-memory fake's contract.
        """
        existing = self._session.get(AssessmentRow, record.assessment_id)
        if existing is None:
            raise KeyError(f"assessment {record.assessment_id!r} does not exist")

        assessment_row, decision_row = record_to_rows(record)
        self._session.merge(assessment_row)

        stored_decision = self._session.get(HumanDecisionRow, record.assessment_id)
        if decision_row is not None:
            self._session.merge(decision_row)
        elif stored_decision is not None:
            self._session.delete(stored_decision)
        self._session.flush()

    def get(self, assessment_id: str) -> AssessmentRecord | None:
        """Return the record for ``assessment_id``, or ``None`` if there is none."""
        assessment_row = self._session.get(AssessmentRow, assessment_id)
        if assessment_row is None:
            return None
        decision_row = self._session.get(HumanDecisionRow, assessment_id)
        return rows_to_record(assessment_row, decision_row)

    def list(
        self,
        *,
        claim_id: str | None = None,
        status: AssessmentStatus | None = None,
        limit: int = _DEFAULT_LIMIT,
        offset: int = 0,
    ) -> tuple[AssessmentRecord, ...]:
        """Return records newest first (``created_at`` desc, then ``assessment_id``)."""
        statement = select(AssessmentRow)
        if claim_id is not None:
            statement = statement.where(AssessmentRow.claim_id == claim_id)
        if status is not None:
            statement = statement.where(AssessmentRow.status == status.value)
        statement = (
            statement.order_by(
                AssessmentRow.created_at.desc(), AssessmentRow.assessment_id
            )
            .limit(limit)
            .offset(offset)
        )

        assessment_rows = list(self._session.execute(statement).scalars())
        decisions = self._decisions_for([row.assessment_id for row in assessment_rows])
        return tuple(
            rows_to_record(row, decisions.get(row.assessment_id))
            for row in assessment_rows
        )

    def _decisions_for(
        self, assessment_ids: Sequence[str]
    ) -> dict[str, HumanDecisionRow]:
        """One query for every decision row in the current page."""
        if not assessment_ids:
            return {}
        rows = self._session.execute(
            select(HumanDecisionRow).where(
                HumanDecisionRow.assessment_id.in_(assessment_ids)
            )
        ).scalars()
        return {row.assessment_id: row for row in rows}
