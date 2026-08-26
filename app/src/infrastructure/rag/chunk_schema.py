"""Frozen serialization contract for [domain.chunk.Chunk] -- [M3-01].

Mirrors [infrastructure.parsing.clause_schema]'s split: [domain.chunk.Chunk]
stays a stdlib-only frozen dataclass per that layer's constraint (pydantic
is a forbidden import in domain/application, see tests/architecture/
test_layer_boundaries.py); [ChunkRecord] is the flat, validated row
``scripts/build_chunks.py`` persists to ``build/``.
"""

from typing import Annotated

from pydantic import BaseModel, Field

from domain.chunk import Chunk, ChunkRule
from domain.clause_classification import ClauseType, TypeSource

SCHEMA_VERSION = "v1"

_NonEmptyStr = Annotated[str, Field(min_length=1)]


class ChunkRecord(BaseModel):
    """One flattened, validated chunk row in the chunk corpus."""

    schema_version: str
    chunk_id: _NonEmptyStr
    document_id: _NonEmptyStr
    clause_id: _NonEmptyStr
    source_clause_ids: list[str]
    chunk_index: Annotated[int, Field(ge=0)]
    chunk_count: Annotated[int, Field(ge=1)]
    parent_path: str
    text: str
    char_count: Annotated[int, Field(ge=0)]
    rule: ChunkRule
    clause_type: ClauseType
    type_source: TypeSource
    confidence: float | None
    bundle_section: str | None
    susep_process: _NonEmptyStr
    insurer: _NonEmptyStr
    cnpj: _NonEmptyStr
    product_line: _NonEmptyStr
    indemnity_regime: _NonEmptyStr
    filing_year: _NonEmptyStr


def flatten_chunk(chunk: Chunk) -> ChunkRecord:
    """Flatten a [Chunk] (its provenance already carries the manifest fields)."""
    provenance = chunk.provenance
    return ChunkRecord(
        schema_version=SCHEMA_VERSION,
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        clause_id=chunk.clause_id,
        source_clause_ids=list(chunk.source_clause_ids),
        chunk_index=chunk.chunk_index,
        chunk_count=chunk.chunk_count,
        parent_path=chunk.parent_path,
        text=chunk.text,
        char_count=chunk.char_count,
        rule=chunk.rule,
        clause_type=chunk.clause_type,
        type_source=chunk.type_source,
        confidence=chunk.confidence,
        bundle_section=chunk.bundle_section,
        susep_process=provenance.susep_process,
        insurer=provenance.insurer,
        cnpj=provenance.cnpj,
        product_line=provenance.product_line,
        indemnity_regime=provenance.indemnity_regime,
        filing_year=provenance.process_year,
    )
