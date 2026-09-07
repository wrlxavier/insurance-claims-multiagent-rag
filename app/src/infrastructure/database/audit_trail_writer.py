"""The transactional audit-trail writer -- [M5-04].

``SqlAlchemyAuditTrailWriter`` satisfies
``application.ports.audit_trail_writer.AuditTrailWriter``. Unlike
``SqlAlchemyAuditTrailSink`` (the graph-facing adapter, which owns its own
session and commits), this one is bound to the ``SqlAlchemyUnitOfWork``'s
session and only flushes -- so the trail the resume path captured lands in the
same transaction as the settled ``AssessmentRecord``.
"""

from collections.abc import Sequence

from sqlalchemy.orm import Session

from application.audit_trail_entry import AuditTrailEntry
from infrastructure.database.audit_repository import append_audit_entries


class SqlAlchemyAuditTrailWriter:
    """Append captured audit-trail entries within the current unit of work."""

    def __init__(self, session: Session) -> None:
        """Bind the writer to the unit of work's session."""
        self._session = session

    def append(
        self,
        *,
        claim_id: str,
        thread_id: str,
        entries: Sequence[AuditTrailEntry],
    ) -> None:
        """Persist ``entries``, skipping any already written. Flushes, never commits."""
        append_audit_entries(
            self._session, claim_id=claim_id, thread_id=thread_id, entries=entries
        )
