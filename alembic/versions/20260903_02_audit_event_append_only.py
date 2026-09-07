"""Make audit_event append-only at the database.

[M5-03]'s DoD: "the audit trail is append-only ... add a test asserting it
cannot be updated." Until now the property held only by construction --
[infrastructure.database.audit_repository] offers an insert and nothing else.
This adds the database-level guarantee: a `BEFORE UPDATE OR DELETE` trigger on
`audit_event` that raises.

A trigger, not an `ON UPDATE ... DO INSTEAD NOTHING` rule: a rule would swallow
the write silently, so a bug (or a compromised connection) that tries to rewrite
history would look like it succeeded. The trigger fails loudly, which is also
what makes the DoD's "cannot be updated" test possible.

The insert path is untouched: `append_audit_events` uses
`INSERT ... ON CONFLICT (thread_id, sequence) DO NOTHING`, which never fires an
UPDATE, so the checkpoint node's idempotent re-write on resume still works.

Revision ID: 20260903_02
Revises: 20260903_01
Create Date: 2026-09-03 00:00:01
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260903_02"
down_revision: str | None = "20260903_01"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

_CREATE_FUNCTION = """
CREATE FUNCTION audit_event_reject_mutation() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'audit_event is append-only: % is not permitted', TG_OP
        USING ERRCODE = 'restrict_violation';
END;
$$ LANGUAGE plpgsql;
"""

_CREATE_TRIGGER = """
CREATE TRIGGER audit_event_append_only
    BEFORE UPDATE OR DELETE ON audit_event
    FOR EACH ROW EXECUTE FUNCTION audit_event_reject_mutation();
"""


def upgrade() -> None:
    """Install the reject-mutation function and the audit_event trigger."""
    op.execute(_CREATE_FUNCTION)
    op.execute(_CREATE_TRIGGER)


def downgrade() -> None:
    """Remove the trigger and its function."""
    op.execute("DROP TRIGGER audit_event_append_only ON audit_event")
    op.execute("DROP FUNCTION audit_event_reject_mutation()")
