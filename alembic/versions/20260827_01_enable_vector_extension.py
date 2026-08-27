"""Enable the pgvector extension.

[M0-08] creates the database capability, not the schema: no chunk table
([M3-02]), no checkpointer tables ([M4-09]), no domain or audit tables
([M5-03]). This migration therefore creates the extension and nothing else.

`vector` is not a trusted extension, so the role running this migration must
be a superuser -- or the extension must already exist, in which case
`IF NOT EXISTS` makes this a no-op. See docs/DATABASE.md.

Revision ID: 20260827_01
Revises:
Create Date: 2026-08-27 00:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260827_01"
down_revision: str | None = None
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    """Create the `vector` extension."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    """Drop the `vector` extension."""
    op.execute("DROP EXTENSION IF EXISTS vector")
