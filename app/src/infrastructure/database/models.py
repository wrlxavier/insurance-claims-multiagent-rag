"""SQLAlchemy ORM models -- [M3-02].

``ChunkRow`` is the persistence representation of [domain.chunk.Chunk] /
[infrastructure.rag.chunk_schema.ChunkRecord]: the chunk corpus indexed in
Postgres for retrieval. It is a third representation on purpose -- the domain
dataclass stays stdlib-only (pydantic and sqlalchemy are forbidden imports in
domain/application, see tests/architecture/test_layer_boundaries.py), the
pydantic ``ChunkRecord`` is the ``build/`` serialization row, and this is the
table.

Scope note: the ``embedding`` column is the ``halfvec(768)`` vector the [M3-02]
embedding pipeline fills; its ANN index and the ANN-vs-exact measurement are a
later slice of the same issue. See docs/DATABASE.md.
"""

from collections.abc import Iterable

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import CheckConstraint, Float, Index, Integer, Text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from domain.chunk import ChunkRule
from domain.clause_classification import ClauseType, TypeSource
from infrastructure.database.base import Base
from infrastructure.rag.embedding_config import EMBEDDING_DIMENSIONS

_SOURCE_VALUES = ("text", "ocr")


def _in_clause(column: str, values: Iterable[str]) -> str:
    """Render ``column IN ('a', 'b', ...)`` for a CHECK constraint."""
    rendered = ", ".join(f"'{value}'" for value in values)
    return f"{column} IN ({rendered})"


class ChunkRow(Base):
    """One chunk row: clause text plus the metadata [M3-04] filters by.

    Column names mirror [infrastructure.rag.chunk_schema.ChunkRecord] field for
    field (except ``text`` -> ``embedded_text``), so the write path in
    [infrastructure.database.chunk_repository] is a direct mapping and "carry
    the full [M1-05] provenance" is auditable column by column.
    """

    __tablename__ = "chunk"

    # `chunk_id` is deterministic upstream ([M3-01]/[M1-07]): `clause_id` for a
    # one-chunk clause, `f"{clause_id}#{index}"` for a split, where
    # `clause_id = f"{document_id}:{path}"`. Same input, same id -- so it is the
    # natural key, and the write path upserts on it (a re-run neither
    # duplicates nor needs a wipe).
    chunk_id: Mapped[str] = mapped_column(Text, primary_key=True)

    schema_version: Mapped[str] = mapped_column(Text, nullable=False)
    document_id: Mapped[str] = mapped_column(Text, nullable=False)
    clause_id: Mapped[str] = mapped_column(Text, nullable=False)
    source_clause_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # "" when the anchor clause has no ancestors -- a real value, never NULL.
    parent_path: Mapped[str] = mapped_column(Text, nullable=False)
    # The string the embedding model sees (ancestor breadcrumb prepended).
    embedded_text: Mapped[str] = mapped_column(Text, nullable=False)
    # The same clause with only the injected breadcrumb removed -- [M4-01]'s
    # citation type needs a quoted excerpt, which is not `embedded_text`.
    display_text: Mapped[str] = mapped_column(Text, nullable=False)
    # Measures `embedded_text`.
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)

    rule: Mapped[str] = mapped_column(Text, nullable=False)
    clause_type: Mapped[str] = mapped_column(Text, nullable=False)
    type_source: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Genuinely nullable: no server_default, no sentinel. A strict M3-04 filter
    # `WHERE bundle_section = :x` must silently (and testably in SQL) exclude
    # unknown-bundle chunks -- see the [M1-06] cross-note in [M3-04].
    bundle_section: Mapped[str | None] = mapped_column(Text, nullable=True)

    susep_process: Mapped[str] = mapped_column(Text, nullable=False)
    # Not indexed: M3-04 filters insurers by CNPJ, never by name (HDI Seguros
    # vs HDI Global share a brand but are different legal entities).
    insurer: Mapped[str] = mapped_column(Text, nullable=False)
    cnpj: Mapped[str] = mapped_column(Text, nullable=False)
    product_line: Mapped[str] = mapped_column(Text, nullable=False)
    indemnity_regime: Mapped[str] = mapped_column(Text, nullable=False)
    filing_year: Mapped[str] = mapped_column(Text, nullable=False)

    # The dense-retrieval vector for `embedded_text`, per the pinned model
    # contract in `infrastructure.rag.embedding_config` (768-dim, cosine,
    # L2-normalised). `halfvec` half-precision storage follows [M0-08]'s
    # decision. Genuinely nullable: an un-embedded chunk is `NULL`, and
    # `WHERE embedding IS NULL` is the embedding pipeline's resumable cursor.
    # `upsert_chunks` (the metadata write path) deliberately never writes this
    # column -- see `chunk_repository._UPDATE_COLUMNS`. The ANN index over it is
    # a later [M3-02] slice.
    embedding: Mapped[list[float] | None] = mapped_column(
        HALFVEC(EMBEDDING_DIMENSIONS), nullable=True
    )

    __table_args__ = (
        CheckConstraint(
            _in_clause("rule", (member.value for member in ChunkRule)),
            name="rule_valid",
        ),
        CheckConstraint(
            _in_clause("clause_type", (member.value for member in ClauseType)),
            name="clause_type_valid",
        ),
        CheckConstraint(
            _in_clause("type_source", (member.value for member in TypeSource)),
            name="type_source_valid",
        ),
        CheckConstraint(
            _in_clause("source", _SOURCE_VALUES),
            name="source_valid",
        ),
        # One index per field M3-04 filters by, plus the composite for its
        # default path (SUSEP process + insurer CNPJ together). Named
        # explicitly rather than via `index=True` so the migration and the
        # model agree on the exact names with no naming-convention ambiguity.
        # All forward-looking: at ~4,900 chunks Postgres seq-scans in well
        # under a millisecond regardless.
        Index("ix_chunk_clause_type", "clause_type"),
        Index("ix_chunk_bundle_section", "bundle_section"),
        Index("ix_chunk_susep_process", "susep_process"),
        Index("ix_chunk_cnpj", "cnpj"),
        Index("ix_chunk_product_line", "product_line"),
        Index("ix_chunk_susep_process_cnpj", "susep_process", "cnpj"),
    )
