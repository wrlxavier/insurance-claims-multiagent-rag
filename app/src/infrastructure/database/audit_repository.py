"""The write path for the durable audit trail -- [M4-09].

``append_audit_events`` is the only way a row reaches ``audit_event``, and it is
insert-only: there is deliberately no update or delete helper here, so the
append-only property [M5-03] will enforce in the database holds by construction
in the meantime.

The insert is idempotent on ``(thread_id, sequence)``. That is not defensive
padding -- it is required. The checkpoint node that calls this re-runs from the
top every time its LangGraph thread is resumed, and a run that crashes between
the write and the checkpoint commit resumes into exactly the same write. An
event's position in its thread's trail is deterministic, so repeating the write
is a no-op rather than a duplicate.
"""

from collections.abc import Sequence

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from infrastructure.database.models import AuditEventRow
from infrastructure.graph.state import AuditRecord


def _row_values(
    record: AuditRecord, *, claim_id: str, thread_id: str, sequence: int
) -> dict[str, object]:
    """Map one [AuditRecord] to an ``audit_event`` row.

    ``AuditEvent.token_usage`` is flattened into the three integer columns; every
    other field maps by name.
    """
    event = record.event
    usage = event.token_usage
    return {
        "thread_id": thread_id,
        "sequence": sequence,
        "claim_id": claim_id,
        "timestamp": event.timestamp,
        "node": event.node,
        "action": event.action,
        "model": event.model,
        "model_version": event.model_version,
        "input_tokens": usage.input_tokens if usage is not None else None,
        "output_tokens": usage.output_tokens if usage is not None else None,
        "total_tokens": usage.total_tokens if usage is not None else None,
        "confidence": event.confidence,
        "node_input": event.node_input,
        "payload": record.payload,
    }


def append_audit_events(
    session: Session,
    *,
    claim_id: str,
    thread_id: str,
    records: Sequence[AuditRecord],
) -> int:
    """Insert one row per record, skipping any already written. Return the count.

    ``records`` is the run's whole trail in order; the index of each record is
    its ``sequence``. The caller owns the transaction -- this flushes but never
    commits.
    """
    if not records:
        return 0

    # RETURNING, not `rowcount`: a multi-row insert goes through SQLAlchemy's
    # executemany path, where the driver may report `rowcount` as -1. The rows
    # that come back are exactly the ones the conflict clause did *not* skip.
    statement = (
        pg_insert(AuditEventRow)
        .on_conflict_do_nothing(index_elements=["thread_id", "sequence"])
        .returning(AuditEventRow.sequence)
    )
    written = session.execute(
        statement,
        [
            _row_values(record, claim_id=claim_id, thread_id=thread_id, sequence=index)
            for index, record in enumerate(records)
        ],
    ).all()
    session.flush()
    return len(written)
