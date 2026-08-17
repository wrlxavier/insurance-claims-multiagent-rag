"""Frozen serialization contract between parsing and everything downstream.

Not the domain model -- [domain.clause_tree.Clause] and
[domain.clause_classification.TypedClause] stay stdlib-only frozen
dataclasses per that layer's constraint (pydantic/pyarrow are forbidden
imports in domain/application, see tests/architecture/test_layer_boundaries
.py). [ParsedClauseRecord] is the flat, validated row [M1-07]'s
``scripts/build_corpus.py`` persists to ``build/`` -- the wire schema a
re-parse must keep honest, so it fails loudly on any clause missing
required provenance rather than writing it silently.

``SCHEMA_VERSION`` feeds ``build/manifest.json`` (see
[infrastructure.parsing.corpus_artifact]), so a consumer reading the
artefact in isolation still knows which contract produced it.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from domain.clause_classification import ClauseType, TypedClause, TypeSource

SCHEMA_VERSION = "v1"

_NonEmptyStr = Annotated[str, Field(min_length=1)]


class ParsedClauseRecord(BaseModel):
    """One flattened, validated clause row in the parsed corpus."""

    schema_version: str
    clause_id: _NonEmptyStr
    document_id: _NonEmptyStr
    parent_id: str | None
    path: _NonEmptyStr
    title: _NonEmptyStr
    text: str
    clause_type: ClauseType
    type_source: TypeSource
    confidence: float | None
    bundle_section: str | None
    page_start: Annotated[int, Field(ge=1)]
    page_end: Annotated[int, Field(ge=1)]
    source: Literal["text", "ocr"]
    susep_process: _NonEmptyStr
    insurer: _NonEmptyStr
    cnpj: _NonEmptyStr
    product_line: _NonEmptyStr
    indemnity_regime: _NonEmptyStr
    filing_year: _NonEmptyStr


def flatten_typed_clause(
    typed: TypedClause, *, source: Literal["text", "ocr"]
) -> ParsedClauseRecord:
    """Flatten a [TypedClause] plus its manifest provenance into one record.

    ``source`` is supplied by the caller (derived from the manifest's
    ``extraction_mode`` -- see ``scripts/build_corpus.py``) because no
    per-clause extraction mode exists on [Clause]/[ExtractedDocument]: a
    document is routed through exactly one extraction path end to end.
    """
    clause = typed.clause
    provenance = typed.provenance
    return ParsedClauseRecord(
        schema_version=SCHEMA_VERSION,
        clause_id=clause.clause_id,
        document_id=clause.document_id,
        parent_id=clause.parent_id,
        path=clause.path,
        title=clause.title,
        text="\n".join(clause.content_lines),
        clause_type=typed.clause_type,
        type_source=typed.type_source,
        confidence=typed.confidence,
        bundle_section=clause.bundle_section,
        page_start=clause.page_start,
        page_end=clause.page_end,
        source=source,
        susep_process=provenance.susep_process,
        insurer=provenance.insurer,
        cnpj=provenance.cnpj,
        product_line=provenance.product_line,
        indemnity_regime=provenance.indemnity_regime,
        filing_year=provenance.process_year,
    )
