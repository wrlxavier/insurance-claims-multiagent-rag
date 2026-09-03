"""The policy-clause entity [M5-01].

Distinct from ``domain.clause_tree.Clause`` on purpose. That type is an
18-field node in a document's parse tree -- a build-time artifact with
parent/child pointers, page spans and per-line page attribution, cached to
Parquet and never persisted. ``PolicyClause`` is the lean, persistence- and
citation-facing view the application layer works in: the clause of a
*registered product*, identified by a stable id, carrying its business type
and text. [M5-02]'s ``ClauseRepository`` returns these to hydrate and
validate a [domain.citation.Citation]; [M5-03]'s mapper builds one from a
``clause_tree.Clause`` plus its ``TypedClause`` classification and
``ClauseProvenance``.

The DoD names this entity ``Clause``; it is named ``PolicyClause`` here to
avoid shadowing the existing ``Clause`` -- recorded in
``docs/ARCHITECTURE.md``.

Standard library only -- enforced by
tests/architecture/test_layer_boundaries.py.
"""

from dataclasses import dataclass

from domain.clause_classification import ClauseType
from domain.susep_process import SusepProcess


@dataclass(frozen=True)
class PolicyClause:
    """One clause of a registered product, as the service layer sees it."""

    clause_id: str
    susep_process: SusepProcess
    document_id: str
    clause_type: ClauseType
    text: str
    heading: str = ""

    def __post_init__(self) -> None:
        """Reject empty identifiers or text, or a non-``ClauseType`` type."""
        for name in ("clause_id", "document_id", "text"):
            if not getattr(self, name):
                raise ValueError(f"PolicyClause.{name} must not be empty")
        if not isinstance(self.clause_type, ClauseType):
            raise ValueError(
                f"PolicyClause.clause_type must be a ClauseType, "
                f"got {self.clause_type!r}"
            )
