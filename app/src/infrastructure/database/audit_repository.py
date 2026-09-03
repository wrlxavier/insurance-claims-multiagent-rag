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

from application.audit_trail_entry import AuditTrailEntry
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


def _entry_values(
    entry: AuditTrailEntry, *, claim_id: str, thread_id: str
) -> dict[str, object]:
    """Map one ``AuditTrailEntry`` DTO to an ``audit_event`` row.

    The [M5-04] resume path captures the graph's trail as ``AuditTrailEntry``
    DTOs (``sequence`` already assigned) so the use case can persist it in its
    own transaction; this is the same row shape ``_row_values`` produces.
    """
    return {
        "thread_id": thread_id,
        "sequence": entry.sequence,
        "claim_id": claim_id,
        "timestamp": entry.timestamp,
        "node": entry.node,
        "action": entry.action,
        "model": entry.model,
        "model_version": entry.model_version,
        "input_tokens": entry.input_tokens,
        "output_tokens": entry.output_tokens,
        "total_tokens": entry.total_tokens,
        "confidence": entry.confidence,
        "node_input": entry.node_input,
        "payload": dict(entry.payload) if entry.payload is not None else None,
    }


def _insert_rows(session: Session, rows: list[dict[str, object]]) -> int:
    """Insert ``rows`` into ``audit_event``, skipping any already written.

    RETURNING, not ``rowcount``: a multi-row insert goes through SQLAlchemy's
    executemany path, where the driver may report ``rowcount`` as -1. The rows
    that come back are exactly the ones the conflict clause did *not* skip. The
    caller owns the transaction -- this flushes but never commits.
    """
    if not rows:
        return 0
    statement = (
        pg_insert(AuditEventRow)
        .on_conflict_do_nothing(index_elements=["thread_id", "sequence"])
        .returning(AuditEventRow.sequence)
    )
    written = session.execute(statement, rows).all()
    session.flush()
    return len(written)


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
    return _insert_rows(
        session,
        [
            _row_values(record, claim_id=claim_id, thread_id=thread_id, sequence=index)
            for index, record in enumerate(records)
        ],
    )


def append_audit_entries(
    session: Session,
    *,
    claim_id: str,
    thread_id: str,
    entries: Sequence[AuditTrailEntry],
) -> int:
    """Insert one row per entry, skipping any already written. Return the count.

    The [M5-04] counterpart of ``append_audit_events``: the entries already carry
    their ``sequence`` (assigned when the resume path captured the trail), and
    the idempotent ``ON CONFLICT`` keeps a decision retry a no-op.
    """
    return _insert_rows(
        session,
        [
            _entry_values(entry, claim_id=claim_id, thread_id=thread_id)
            for entry in entries
        ],
    )
