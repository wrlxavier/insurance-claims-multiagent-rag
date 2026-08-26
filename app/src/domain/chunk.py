"""Embeddable chunks recovered from a document's clause tree.

Downstream contract for [M3-01]: chunking the clause tree from [M1-04]
instead of fixed token windows means most clauses map one-to-one onto a
chunk, but a short clause folds into its parent and an over-long clause
splits into several -- so a [Chunk] cannot assume "one clause, one chunk."
``source_clause_ids`` is what keeps that honest: every original clause id
folded into a chunk is recorded there (including the chunk's own anchor
``clause_id``), so a future retriever can roll a chunk-level hit back up to
clause-id granularity for [infrastructure.evaluation.retriever.Retriever]
and ``golden-set-v1`` (both frozen at clause-id granularity, per
[M2-06]/[M2-07]) without re-deriving anything.

Frozen dataclasses only, no third-party imports -- same constraint as
[domain.clause_tree], enforced by tests/architecture/test_layer_boundaries.py.
"""

from dataclasses import dataclass
from enum import Enum

from domain.clause_classification import ClauseProvenance, ClauseType, TypeSource


class ChunkRule(Enum):
    """Which chunking rule produced a given [Chunk]."""

    SINGLE = "single"
    MERGED = "merged"
    ITEM_BOUNDARY_SPLIT = "item_boundary_split"
    SLIDING_WINDOW_SPLIT = "sliding_window_split"


@dataclass(frozen=True)
class Chunk:
    """One embeddable unit of clause text, ready for indexing."""

    document_id: str
    chunk_id: str
    clause_id: str
    source_clause_ids: tuple[str, ...]
    chunk_index: int
    chunk_count: int
    parent_path: str
    text: str
    char_count: int
    rule: ChunkRule
    clause_type: ClauseType
    type_source: TypeSource
    confidence: float | None
    bundle_section: str | None
    provenance: ClauseProvenance


@dataclass(frozen=True)
class ChunkingReport:
    """Per-document chunking summary -- always computed, never silent."""

    document_id: str
    chunk_count: int
    single_count: int
    merged_count: int
    item_boundary_split_count: int
    sliding_window_split_count: int
    min_char_count: int
    p50_char_count: int
    p90_char_count: int
    max_char_count: int
    mean_char_count: float
