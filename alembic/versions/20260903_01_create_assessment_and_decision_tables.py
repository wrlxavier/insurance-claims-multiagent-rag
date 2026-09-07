"""Create the assessment and human_decision tables.

[M5-03]'s domain tables: the persistence side of
[application.assessment_record.AssessmentRecord] (the servable aggregate, which
unlike a grounded [domain.assessment.Assessment] can carry zero citations) and
the analyst's [domain.human_decision.HumanDecision] recorded beside it.

`citations` and `consistency_flags` are `JSONB` arrays of objects, not child
tables: both are frozen value-object tuples with no identity, always read whole
with the record and never queried by field -- the same call as
`audit_event.payload`. `missing_information` is a plain `TEXT[]`.

`human_decision.assessment_id` is both the primary key and a foreign key to
`assessment` -- "a decision always references the assessment it acted on" (M5-01)
made structural. The `edited_assessment` JSONB is present exactly when
`decision = 'edit'`, enforced by a CHECK that mirrors `HumanDecision`'s own
validator.

The enum value lists are plain literals here (like `20260827_02`) so the
migration imports no app code that could move under it;
`tests/unit/infrastructure/database/test_models.py` ties the model's CHECKs back
to the domain enums, and any divergence surfaces as an `alembic check` diff.

Revision ID: 20260903_01
Revises: 20260902_01
Create Date: 2026-09-03 00:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260903_01"
down_revision: str | None = "20260902_01"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

_VERDICT_VALUES = ("compatible", "incompatible", "insufficient_information")
_STATUS_VALUES = ("awaiting_review", "decided")
_DECISION_VALUES = ("approve", "edit", "reject")


def _in_check(column: str, values: Sequence[str]) -> str:
    """Render ``column IN ('a', 'b', ...)`` for a CHECK constraint."""
    rendered = ", ".join(f"'{value}'" for value in values)
    return f"{column} IN ({rendered})"


def upgrade() -> None:
    """Create `assessment`, `human_decision`, and the query indexes."""
    op.create_table(
        "assessment",
        sa.Column("assessment_id", sa.Text(), nullable=False),
        sa.Column("claim_id", sa.Text(), nullable=False),
        sa.Column("verdict", sa.Text(), nullable=False),
        sa.Column("reasoning", sa.Text(), nullable=False),
        sa.Column("recommended_action", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("context_sufficient", sa.Boolean(), nullable=True),
        sa.Column("clarification_exhausted", sa.Boolean(), nullable=False),
        sa.Column("missing_information", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("citations", postgresql.JSONB(), nullable=False),
        sa.Column("consistency_flags", postgresql.JSONB(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        # Bare constraint names: `Base.metadata`'s naming_convention -- which
        # Alembic's op context also applies -- expands these to
        # `ck_assessment_<name>` / `pk_assessment`, matching the model.
        sa.CheckConstraint(_in_check("verdict", _VERDICT_VALUES), name="verdict_valid"),
        sa.CheckConstraint(_in_check("status", _STATUS_VALUES), name="status_valid"),
        sa.PrimaryKeyConstraint("assessment_id"),
    )
    op.create_index("ix_assessment_claim_id", "assessment", ["claim_id"])
    op.create_index("ix_assessment_status", "assessment", ["status"])
    op.create_index("ix_assessment_created_at", "assessment", ["created_at"])

    op.create_table(
        "human_decision",
        sa.Column("assessment_id", sa.Text(), nullable=False),
        sa.Column("decision", sa.Text(), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("notes", sa.Text(), nullable=False),
        sa.Column("edited_assessment", postgresql.JSONB(), nullable=True),
        sa.CheckConstraint(
            _in_check("decision", _DECISION_VALUES), name="decision_valid"
        ),
        sa.CheckConstraint(
            "(decision = 'edit') = (edited_assessment IS NOT NULL)",
            name="edited_assessment_iff_edit",
        ),
        # Unnamed FK: the naming_convention builds
        # `fk_human_decision_assessment_id_assessment` on both sides.
        sa.ForeignKeyConstraint(["assessment_id"], ["assessment.assessment_id"]),
        sa.PrimaryKeyConstraint("assessment_id"),
    )


def downgrade() -> None:
    """Drop `human_decision` first (it references `assessment`), then `assessment`."""
    op.drop_table("human_decision")
    op.drop_table("assessment")
