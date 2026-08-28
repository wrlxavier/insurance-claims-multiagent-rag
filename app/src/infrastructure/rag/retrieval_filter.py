"""The metadata pre-filter that cuts the retrieval search space -- [M3-04].

A claims analyst works a case for a known policy, so the *default* retrieval
path filters chunks to one SUSEP process + insurer CNPJ before anything is
ranked -- not an optional optimisation. ``docs/HYBRID_RETRIEVAL.md`` has the
measured effect (the M3-03 lexical baseline's ~35-point Recall@10 gap is
entirely cross-document leakage this filter removes).

One ``RetrievalFilter`` value drives both retrieval legs: the dense leg
translates it to a SQL ``WHERE`` ([infrastructure.database.chunk_repository.
search_chunks_by_vector]); the in-memory lexical leg applies :meth:`matches` to
each chunk before the chunk->clause roll-up. Filter by CNPJ, never insurer name
-- HDI Seguros (``29980158000157``) and HDI Global (``18096627000153``) share a
brand but are different legal entities ([M1-05]).
"""

from dataclasses import dataclass

from domain.clause_classification import ClauseType
from infrastructure.rag.chunk_schema import ChunkRecord


@dataclass(frozen=True)
class RetrievalFilter:
    """A conjunction of equality pre-filters over chunk metadata.

    Every field is optional; ``None`` means "do not constrain this field". An
    all-``None`` filter matches every chunk -- the unknown-process degradation
    path ([M3-04] DoD item 5), not an error.
    """

    susep_process: str | None = None
    cnpj: str | None = None
    product_line: str | None = None
    bundle_section: str | None = None
    clause_type: ClauseType | None = None
    # When ``bundle_section`` is set, ``strict_bundle=False`` (the default) also
    # keeps chunks whose ``bundle_section`` is ``None``: when a multi-product
    # document's product-start signal is unclear a large share of its clauses
    # land in that bucket ([M1-06] cross-note in [M3-04]), and excluding them
    # silently is the failure mode this guards against. ``strict_bundle=True``
    # is exact equality.
    strict_bundle: bool = False

    @classmethod
    def from_manifest_row(cls, row: dict[str, str]) -> "RetrievalFilter":
        """The default retrieval filter for a document: its SUSEP process + CNPJ.

        ``row`` is a ``data/policies/manifest.csv`` record
        ([infrastructure.parsing.manifest.read_manifest]).
        """
        return cls(susep_process=row["susep_process"], cnpj=row["cnpj"])

    @property
    def is_empty(self) -> bool:
        """True when no field constrains anything (matches every chunk)."""
        return (
            self.susep_process is None
            and self.cnpj is None
            and self.product_line is None
            and self.bundle_section is None
            and self.clause_type is None
        )

    def matches(self, chunk: ChunkRecord) -> bool:
        """Whether ``chunk`` passes every set constraint (the lexical leg)."""
        if self.susep_process is not None and chunk.susep_process != self.susep_process:
            return False
        if self.cnpj is not None and chunk.cnpj != self.cnpj:
            return False
        if self.product_line is not None and chunk.product_line != self.product_line:
            return False
        if self.clause_type is not None and chunk.clause_type != self.clause_type:
            return False
        if self.bundle_section is not None:
            if self.strict_bundle:
                return chunk.bundle_section == self.bundle_section
            return chunk.bundle_section in (self.bundle_section, None)
        return True
