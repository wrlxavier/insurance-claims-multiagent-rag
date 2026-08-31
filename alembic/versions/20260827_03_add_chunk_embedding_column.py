"""Add the chunk embedding vector column.

[M3-02]'s embedding-pipeline half, first slice: the `halfvec(768)` column the
batched embedding pipeline fills. The dimension and the half-precision storage
follow the pinned model contract in
`app/src/infrastructure/rag/embedding_config.py` (`EMBEDDING_DIMENSIONS = 768`)
and [M0-08]'s `halfvec` decision in docs/DATABASE.md.

The column is nullable with no default: an un-embedded chunk carries `NULL`, and
`WHERE embedding IS NULL` is the pipeline's resumable cursor. The ANN index over
this column (`halfvec_cosine_ops`) and its build-time/size record are a later
[M3-02] slice, not here -- exact `<=>` cosine ordering works on the bare column
in the meantime.

`768` is kept as a literal, not an import: the migration imports no app code that
could move under it, matching `20260827_02`. `tests/unit/infrastructure/database/
test_models.py` ties `ChunkRow.embedding` back to `EMBEDDING_DIMENSIONS`, and any
divergence surfaces as an `alembic revision --autogenerate` diff.

Revision ID: 20260827_03
Revises: 20260827_02
Create Date: 2026-08-27 00:00:02
"""

from __future__ import annotations

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op

revision: str = "20260827_03"
down_revision: str | None = "20260827_02"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    """Add the nullable `embedding halfvec(768)` column to `chunk`."""
    op.add_column(
        "chunk",
        sa.Column("embedding", pgvector.sqlalchemy.HALFVEC(768), nullable=True),
    )


def downgrade() -> None:
    """Drop the `embedding` column."""
    op.drop_column("chunk", "embedding")
