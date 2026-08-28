"""Frozen serialization contract for [domain.chunk.Chunk] -- [M3-01].

Mirrors [infrastructure.parsing.clause_schema]'s split: [domain.chunk.Chunk]
stays a stdlib-only frozen dataclass per that layer's constraint (pydantic
is a forbidden import in domain/application, see tests/architecture/
test_layer_boundaries.py); [ChunkRecord] is the flat, validated row
``scripts/build_chunks.py`` persists to ``build/``.
"""

from typing import Annotated, Literal

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
    display_text: str
    char_count: Annotated[int, Field(ge=0)]
    rule: ChunkRule
    clause_type: ClauseType
    type_source: TypeSource
    confidence: float | None
    bundle_section: str | None
    source: Literal["text", "ocr"]
    susep_process: _NonEmptyStr
    insurer: _NonEmptyStr
    cnpj: _NonEmptyStr
    product_line: _NonEmptyStr
    indemnity_regime: _NonEmptyStr
    filing_year: _NonEmptyStr


def _display_text(text: str, parent_path: str) -> str:
    """The chunk text with only the injected ancestor breadcrumb removed.

    ``text`` is what the embedding model sees; [M4-01]'s citation type needs a
    quoted excerpt, which is not that string. [application.use_cases.chunking.
    _render_piece] puts the ``parent_path`` breadcrumb on its own leading line
    when there are ancestors, so removing that one line is exact -- it leaves
    the clause keeping its own heading line -- and is a no-op when the prefix
    is absent (a root-level clause with no ancestors).
    """
    if not parent_path:
        return text
    breadcrumb_line = parent_path + "\n"
    return text.removeprefix(breadcrumb_line)


def flatten_chunk(chunk: Chunk, *, source: Literal["text", "ocr"]) -> ChunkRecord:
    """Flatten a [Chunk] (its provenance already carries the manifest fields).

    ``source`` (``text`` | ``ocr``) is a per-document value from the manifest's
    ``extraction_mode`` column, passed in the same way [infrastructure.parsing.
    clause_schema.flatten_typed_clause] takes it -- no per-clause extraction
    mode exists.
    """
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
        display_text=_display_text(chunk.text, chunk.parent_path),
        char_count=chunk.char_count,
        rule=chunk.rule,
        clause_type=chunk.clause_type,
        type_source=chunk.type_source,
        confidence=chunk.confidence,
        bundle_section=chunk.bundle_section,
        source=source,
        susep_process=provenance.susep_process,
        insurer=provenance.insurer,
        cnpj=provenance.cnpj,
        product_line=provenance.product_line,
        indemnity_regime=provenance.indemnity_regime,
        filing_year=provenance.process_year,
    )
