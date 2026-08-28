"""Create the chunk table.

[M3-02]'s schema half: the chunk corpus indexed in Postgres for retrieval,
carrying the metadata columns and indexes [M3-04] filters by (SUSEP process,
insurer CNPJ, product line, `bundle_section`, clause type), the full [M1-05]
provenance, plus `source` (`text` | `ocr`) from [M1-02] and `type_source` from
[M1-05], so downstream errors stay attributable.

`bundle_section` is genuinely nullable -- no server default, no sentinel -- so a
strict M3-04 filter `WHERE bundle_section = :x` silently and testably excludes
unknown-bundle chunks.

The `embedding` vector column and its ANN index are intentionally not here: the
vector dimension depends on the embedding-model choice, a separate [M3-02] DoD
item, so they land in the embedding-pipeline PR via `ALTER TABLE chunk ADD
COLUMN`. See docs/DATABASE.md.

Revision ID: 20260827_02
Revises: 20260827_01
Create Date: 2026-08-27 00:00:01
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260827_02"
down_revision: str | None = "20260827_01"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

# These must match the CHECK constraints `ChunkRow` builds from the domain
# enums (ChunkRule, ClauseType, TypeSource) and the `source` literal. Kept as
# plain literals here so the migration imports no app code that can move under
# it; `tests/unit/infrastructure/database/test_models.py` ties the model side
# to the enums, and any divergence surfaces as an `alembic revision
# --autogenerate` diff.
_RULE_VALUES = ("single", "merged", "item_boundary_split", "sliding_window_split")
_CLAUSE_TYPE_VALUES = (
    "coverage",
    "exclusion",
    "condition",
    "definition",
    "procedure",
    "other",
)
_TYPE_SOURCE_VALUES = ("rule", "llm")
_SOURCE_VALUES = ("text", "ocr")


def _in_check(column: str, values: Sequence[str]) -> str:
    """Render ``column IN ('a', 'b', ...)`` for a CHECK constraint."""
    rendered = ", ".join(f"'{value}'" for value in values)
    return f"{column} IN ({rendered})"


def upgrade() -> None:
    """Create the `chunk` table and the indexes M3-04 filters by."""
    op.create_table(
        "chunk",
        sa.Column("chunk_id", sa.Text(), nullable=False),
        sa.Column("schema_version", sa.Text(), nullable=False),
        sa.Column("document_id", sa.Text(), nullable=False),
        sa.Column("clause_id", sa.Text(), nullable=False),
        sa.Column("source_clause_ids", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("parent_path", sa.Text(), nullable=False),
        sa.Column("embedded_text", sa.Text(), nullable=False),
        sa.Column("display_text", sa.Text(), nullable=False),
        sa.Column("char_count", sa.Integer(), nullable=False),
        sa.Column("rule", sa.Text(), nullable=False),
        sa.Column("clause_type", sa.Text(), nullable=False),
        sa.Column("type_source", sa.Text(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("bundle_section", sa.Text(), nullable=True),
        sa.Column("susep_process", sa.Text(), nullable=False),
        sa.Column("insurer", sa.Text(), nullable=False),
        sa.Column("cnpj", sa.Text(), nullable=False),
        sa.Column("product_line", sa.Text(), nullable=False),
        sa.Column("indemnity_regime", sa.Text(), nullable=False),
        sa.Column("filing_year", sa.Text(), nullable=False),
        # Bare constraint names: `Base.metadata`'s naming_convention -- which
        # Alembic's op context also applies -- expands these to
        # `ck_chunk_<name>` / `pk_chunk`, matching the model exactly.
        sa.CheckConstraint(_in_check("rule", _RULE_VALUES), name="rule_valid"),
        sa.CheckConstraint(
            _in_check("clause_type", _CLAUSE_TYPE_VALUES),
            name="clause_type_valid",
        ),
        sa.CheckConstraint(
            _in_check("type_source", _TYPE_SOURCE_VALUES),
            name="type_source_valid",
        ),
        sa.CheckConstraint(_in_check("source", _SOURCE_VALUES), name="source_valid"),
        sa.PrimaryKeyConstraint("chunk_id"),
    )
    op.create_index("ix_chunk_clause_type", "chunk", ["clause_type"])
    op.create_index("ix_chunk_bundle_section", "chunk", ["bundle_section"])
    op.create_index("ix_chunk_susep_process", "chunk", ["susep_process"])
    op.create_index("ix_chunk_cnpj", "chunk", ["cnpj"])
    op.create_index("ix_chunk_product_line", "chunk", ["product_line"])
    # M3-04's default retrieval path: SUSEP process + insurer CNPJ together.
    op.create_index("ix_chunk_susep_process_cnpj", "chunk", ["susep_process", "cnpj"])


def downgrade() -> None:
    """Drop the `chunk` table (its indexes and constraints go with it)."""
    op.drop_table("chunk")
