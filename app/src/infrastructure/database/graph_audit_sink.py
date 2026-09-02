"""The one database-to-graph bridge for the audit trail -- [M4-09].

``SqlAlchemyAuditTrailSink`` satisfies
[infrastructure.graph.context.AuditTrailSink] over the project's SQLAlchemy
session factory. It is the counterpart to
[infrastructure.rag.graph_retrieval_adapter.GraphRetrievalAdapter]: the graph
declares the capability it needs, exactly one adapter supplies it, and no node
imports a session.

It owns the transaction -- one commit per call -- because the graph node calling
it has no transaction of its own to join and must not be left holding an open
session across a checkpoint.
"""

from collections.abc import Sequence

from sqlalchemy.orm import Session, sessionmaker

from infrastructure.database.audit_repository import append_audit_events
from infrastructure.graph.state import AuditRecord


class SqlAlchemyAuditTrailSink:
    """Write a graph run's audit trail to the ``audit_event`` table."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """Bind the sink to a session factory (one session per ``record`` call)."""
        self._session_factory = session_factory

    def record(
        self,
        *,
        claim_id: str,
        thread_id: str,
        records: Sequence[AuditRecord],
    ) -> int:
        """Persist ``records`` and commit. Return the number of new rows."""
        with self._session_factory() as session:
            written = append_audit_events(
                session, claim_id=claim_id, thread_id=thread_id, records=records
            )
            session.commit()
        return written
