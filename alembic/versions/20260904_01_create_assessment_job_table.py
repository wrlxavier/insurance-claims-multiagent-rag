"""Create the assessment_job table.

[M5-05]'s queued-run lifecycle: the persistence side of
[application.assessment_job.AssessmentJob]. ``POST /v1/assessments`` no longer
runs the graph in the request -- it writes one of these rows in ``pending`` and a
Redis worker picks it up. The row carries the run state a caller polls
(``pending`` -> ``running`` -> ``succeeded`` / ``failed``) and everything a
worker needs to rebuild the domain ``Claim`` after a retry or a redelivery.

``failure`` is ``JSONB`` (``kind`` / ``error_type`` / ``message`` /
``failed_at``): a frozen value object read whole with the row, never queried by
field. ``status`` is ``TEXT`` + a named CHECK (chunk / assessment precedent). No
FK to ``assessment``: the job row exists before any assessment does, and a failed
run keeps its job row without ever producing one.

The status value list is a plain literal here (like ``20260903_01``) so the
migration imports no app code; ``tests/unit/infrastructure/database/test_models.py``
ties the model's CHECK back to ``JobStatus``.

Revision ID: 20260904_01
Revises: 20260903_02
Create Date: 2026-09-04 00:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260904_01"
down_revision: str | None = "20260903_02"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

_STATUS_VALUES = ("pending", "running", "succeeded", "failed")


def _in_check(column: str, values: Sequence[str]) -> str:
    """Render ``column IN ('a', 'b', ...)`` for a CHECK constraint."""
    rendered = ", ".join(f"'{value}'" for value in values)
    return f"{column} IN ({rendered})"


def upgrade() -> None:
    """Create `assessment_job` and its two query indexes."""
    op.create_table(
        "assessment_job",
        sa.Column("assessment_id", sa.Text(), nullable=False),
        sa.Column("claim_id", sa.Text(), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("policy_ref", sa.Text(), nullable=True),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("failure", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        # Bare name: the metadata naming_convention expands it to
        # `ck_assessment_job_status_valid`, matching the model.
        sa.CheckConstraint(_in_check("status", _STATUS_VALUES), name="status_valid"),
        sa.PrimaryKeyConstraint("assessment_id"),
    )
    op.create_index("ix_assessment_job_status", "assessment_job", ["status"])
    op.create_index("ix_assessment_job_claim_id", "assessment_job", ["claim_id"])


def downgrade() -> None:
    """Drop `assessment_job`."""
    op.drop_index("ix_assessment_job_claim_id", table_name="assessment_job")
    op.drop_index("ix_assessment_job_status", table_name="assessment_job")
    op.drop_table("assessment_job")
