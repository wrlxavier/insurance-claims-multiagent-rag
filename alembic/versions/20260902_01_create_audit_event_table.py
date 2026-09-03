"""Create the audit_event table.

[M4-09]'s durable audit trail: one row per
[infrastructure.graph.state.AuditEvent] a graph run produced, written by the
human-review checkpoint once the analyst has decided.

The trail already survives a restart inside the LangGraph checkpoint, but only
LangGraph can read that back. [M4-09]'s DoD asks for a record that is durable
*and* separate from graph state, so it also lands here, where plain SQL reaches
it.

Scope note: docs/DATABASE.md originally assigned the audit table to [M5-03]
along with the domain tables. Only the audit table moved forward, to the issue
whose DoD requires it; [M5-03] still owns `Assessment`/`HumanDecision` tables and
the database-level append-only enforcement (a rule or trigger rejecting
UPDATE/DELETE), which is not attempted here -- until then the property holds
only by construction, because `infrastructure.database.audit_repository` offers
no update path.

`(thread_id, sequence)` is the primary key rather than an invented event id:
the checkpoint node re-runs from the top every time its thread is resumed, so
the write has to be repeatable, and an event's position within its thread's
trail is already deterministic. That makes `ON CONFLICT DO NOTHING` sufficient.

`AuditEvent.token_usage` is flattened into three nullable integer columns; see
`ChunkRow`'s sibling note in app/src/infrastructure/database/models.py. No CHECK
constraints: unlike `chunk`, none of these columns is a closed enum -- `node` and
`action` are free-form strings each node chooses, and constraining them here
would turn adding a node into a migration.

Revision ID: 20260902_01
Revises: 20260827_03
Create Date: 2026-09-02 00:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260902_01"
down_revision: str | None = "20260827_03"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    """Create the `audit_event` table and the `claim_id` lookup index."""
    op.create_table(
        "audit_event",
        sa.Column("thread_id", sa.Text(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("claim_id", sa.Text(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("node", sa.Text(), nullable=False),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=True),
        sa.Column("model_version", sa.Text(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("node_input", sa.Text(), nullable=True),
        sa.Column("payload", postgresql.JSONB(), nullable=True),
        # Bare name: `Base.metadata`'s naming_convention -- which Alembic's op
        # context also applies -- expands this to `pk_audit_event`, matching the
        # model exactly.
        sa.PrimaryKeyConstraint("thread_id", "sequence"),
    )
    # `thread_id` is already covered by the primary key; `claim_id` is what a
    # human searches the trail by.
    op.create_index("ix_audit_event_claim_id", "audit_event", ["claim_id"])


def downgrade() -> None:
    """Drop the `audit_event` table (its index goes with it)."""
    op.drop_table("audit_event")
