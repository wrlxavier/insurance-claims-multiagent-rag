"""``audit_event`` is append-only at the database -- [M5-03].

The DoD: "the audit trail is append-only; add a test asserting it cannot be
updated." Migration ``20260903_02`` installs a ``BEFORE UPDATE OR DELETE``
trigger that raises; this proves both halves fail loudly and that the insert
path (``append_audit_events``, ``INSERT ... ON CONFLICT DO NOTHING``) is
untouched.
"""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from infrastructure.database.audit_repository import append_audit_events
from infrastructure.graph.state import AuditEvent, AuditRecord

pytestmark = pytest.mark.integration

_THREAD = "thread-append-only"
_CLAIM = "claim-append-only"


def _seed(session: Session) -> None:
    append_audit_events(
        session,
        claim_id=_CLAIM,
        thread_id=_THREAD,
        records=[
            AuditRecord(AuditEvent(node="intake", action="extract_entities")),
            AuditRecord(
                AuditEvent(node="human_review", action="human_decision:approve"),
                {"decision": "approve"},
            ),
        ],
    )
    session.commit()


def test_update_is_rejected(db_session: Session) -> None:
    _seed(db_session)

    with pytest.raises(DBAPIError, match="append-only"):
        db_session.execute(
            text("UPDATE audit_event SET action = 'tampered' WHERE thread_id = :t"),
            {"t": _THREAD},
        )
    db_session.rollback()

    still = (
        db_session.execute(
            text(
                "SELECT action FROM audit_event WHERE thread_id = :t ORDER BY sequence"
            ),
            {"t": _THREAD},
        )
        .scalars()
        .all()
    )
    assert still == ["extract_entities", "human_decision:approve"]


def test_delete_is_rejected(db_session: Session) -> None:
    _seed(db_session)

    with pytest.raises(DBAPIError, match="append-only"):
        db_session.execute(
            text("DELETE FROM audit_event WHERE thread_id = :t"), {"t": _THREAD}
        )
    db_session.rollback()

    count = db_session.execute(
        text("SELECT count(*) FROM audit_event WHERE thread_id = :t"), {"t": _THREAD}
    ).scalar_one()
    assert count == 2


def test_the_idempotent_insert_path_still_works(db_session: Session) -> None:
    _seed(db_session)

    # The same trail again: `ON CONFLICT DO NOTHING` is an INSERT, so the trigger
    # never fires -- the checkpoint node's re-write on resume is unaffected.
    written = append_audit_events(
        db_session,
        claim_id=_CLAIM,
        thread_id=_THREAD,
        records=[
            AuditRecord(AuditEvent(node="intake", action="extract_entities")),
            AuditRecord(
                AuditEvent(node="human_review", action="human_decision:approve"),
                {"decision": "approve"},
            ),
        ],
    )
    db_session.commit()

    assert written == 0
