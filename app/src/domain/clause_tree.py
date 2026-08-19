"""Parent-referenced clause hierarchy recovered from heading numbering.

Downstream contract for [M1-05] (clause type classification) and [M1-06]
(bundle-document splitting): a sub-clause retrieved in isolation needs its
parent's title as context, so [Clause] carries ``parent_id``/``child_ids``
by string id rather than only nesting objects -- a flat, id-addressable
index (``ClauseTree.all_clauses``) lets a downstream consumer look up "the
parent's title" without holding the whole tree in memory, and keeps the
Parquet cache format a flat row-per-clause table, matching
[domain.extracted_text]'s row-per-span convention.

Frozen dataclasses only, no third-party imports, so this stays importable
with nothing but the standard library -- same constraint as
[domain.extracted_text].
"""

from dataclasses import dataclass
from enum import Enum


class HeadingConvention(Enum):
    """Which corpus convention produced a given [Clause]'s heading."""

    NUMBERED_DECIMAL = "numbered_decimal"
    CLAUSULA_KEYWORD = "clausula_keyword"
    UNNUMBERED_PART = "unnumbered_part"


@dataclass(frozen=True)
class Clause:
    """One node in a document's clause tree."""

    document_id: str
    clause_id: str
    path: str
    numbering_label: str
    title: str
    convention: HeadingConvention
    depth: int
    parent_id: str | None
    child_ids: tuple[str, ...]
    content_lines: tuple[str, ...]
    page_start: int
    page_end: int
    bundle_section: str | None = None
    bundle_confidence: str | None = None
    is_depth_anomaly: bool = False


@dataclass(frozen=True)
class ClauseTreeWarning:
    """A non-fatal segmentation anomaly, surfaced rather than hidden."""

    document_id: str
    page_number: int
    kind: str
    detail: str


@dataclass(frozen=True)
class ClauseTreeReport:
    """Per-document segmentation summary -- always computed, never silent."""

    document_id: str
    filename: str
    clause_count: int
    max_depth: int
    orphan_char_count: int
    total_char_count: int
    orphan_ratio: float
    extraction_mode: str
    warnings: tuple[ClauseTreeWarning, ...]


@dataclass(frozen=True)
class ClauseTree:
    """The full clause hierarchy recovered for one document."""

    document_id: str
    filename: str
    roots: tuple[Clause, ...]
    all_clauses: tuple[Clause, ...]
    report: ClauseTreeReport


class OrphanTextExceedsThresholdError(Exception):
    """Raised when a document's orphan-text ratio exceeds the configured share.

    Fails loudly rather than letting a caller silently persist a broken
    tree -- see ``scripts/build_clause_tree.py``, the only place this is
    raised. [application.use_cases.clause_segmentation.segment_document]
    itself never raises, so it stays a pure, total function like
    [application.use_cases.boilerplate_removal.remove_boilerplate].
    """

    def __init__(
        self,
        *,
        document_id: str,
        filename: str,
        orphan_ratio: float,
        threshold: float,
        clause_count: int,
    ) -> None:
        """Build the message from the document's orphan ratio and threshold."""
        self.document_id = document_id
        self.filename = filename
        self.orphan_ratio = orphan_ratio
        self.threshold = threshold
        self.clause_count = clause_count
        super().__init__(
            f"{filename} (document {document_id}): orphan text ratio "
            f"{orphan_ratio:.3f} exceeds threshold {threshold:.3f} "
            f"({clause_count} clauses recovered)."
        )


class ClauseSizeExceedsThresholdError(Exception):
    """Raised when a clause exceeds the configured page-span/char-count ceiling.

    A loud-failure safeguard for an undetected-heading merge like doc 13's
    20-page "RISCOS EXCLUÍDOS" absorbing 41,000+ characters ([M1-08] sample
    #16), mirroring [OrphanTextExceedsThresholdError]. Only ever raised by
    ``scripts/build_clause_tree.py`` -- [application.use_cases.
    clause_segmentation.find_oversized_clauses] itself never raises.
    """

    def __init__(
        self,
        *,
        document_id: str,
        filename: str,
        oversized_clause_ids: tuple[str, ...],
        max_page_span: int,
        max_char_count: int,
    ) -> None:
        """Build the message from the document's oversized clause ids."""
        self.document_id = document_id
        self.filename = filename
        self.oversized_clause_ids = oversized_clause_ids
        self.max_page_span = max_page_span
        self.max_char_count = max_char_count
        super().__init__(
            f"{filename} (document {document_id}): "
            f"{len(oversized_clause_ids)} clause(s) exceed the configured "
            f"page-span ({max_page_span}) or char-count ({max_char_count}) "
            f"ceiling: {', '.join(oversized_clause_ids)}."
        )
