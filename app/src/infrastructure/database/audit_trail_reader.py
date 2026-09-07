"""The durable audit-trail read model -- [M5-04].

``SqlAlchemyAuditTrailReader`` satisfies
``application.ports.audit_trail_reader.AuditTrailReader``, the read behind
``GET /v1/assessments/{id}/audit``. Read-only over the append-only ``audit_event``
table, keyed by ``thread_id`` -- which is the ``assessment_id``.

The trail is written once, in the ``human_review`` node, after the analyst
decides; an assessment still ``AWAITING_REVIEW`` returns an empty tuple.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from application.audit_trail_entry import AuditTrailEntry
from infrastructure.database.models import AuditEventRow


def _row_to_entry(row: AuditEventRow) -> AuditTrailEntry:
    return AuditTrailEntry(
        sequence=row.sequence,
        timestamp=row.timestamp,
        node=row.node,
        action=row.action,
        model=row.model,
        model_version=row.model_version,
        input_tokens=row.input_tokens,
        output_tokens=row.output_tokens,
        total_tokens=row.total_tokens,
        confidence=row.confidence,
        node_input=row.node_input,
        payload=row.payload,
    )


class SqlAlchemyAuditTrailReader:
    """Read one assessment run's durable audit trail from ``audit_event``."""

    def __init__(self, session: Session) -> None:
        """Bind the reader to a session (any session -- it never writes)."""
        self._session = session

    def get_trail(self, assessment_id: str) -> tuple[AuditTrailEntry, ...]:
        """Return the trail for ``assessment_id`` in ``sequence`` order."""
        rows = (
            self._session.execute(
                select(AuditEventRow)
                .where(AuditEventRow.thread_id == assessment_id)
                .order_by(AuditEventRow.sequence)
            )
            .scalars()
            .all()
        )
        return tuple(_row_to_entry(row) for row in rows)
