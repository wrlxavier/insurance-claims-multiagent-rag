"""Domain models for clause type classification and provenance."""

from dataclasses import dataclass
from enum import Enum

from domain.clause_tree import Clause


class ClauseType(Enum):
    """The business classification of a clause."""

    COVERAGE = "coverage"
    EXCLUSION = "exclusion"
    CONDITION = "condition"
    DEFINITION = "definition"
    PROCEDURE = "procedure"
    OTHER = "other"


class TypeSource(Enum):
    """The mechanism that assigned the clause type."""

    RULE = "rule"
    LLM = "llm"


@dataclass(frozen=True)
class ClauseProvenance:
    """Document-level provenance metadata attached to every clause."""

    document_id: str
    susep_process: str
    insurer: str
    cnpj: str
    product_line: str
    indemnity_regime: str
    process_year: str


@dataclass(frozen=True)
class TypedClause:
    """A clause enriched with type classification and provenance."""

    clause: Clause
    clause_type: ClauseType
    type_source: TypeSource
    confidence: float | None
    provenance: ClauseProvenance


class MissingProvenanceError(Exception):
    """Raised when a document ID has no corresponding record in the manifest."""

    def __init__(self, document_id: str) -> None:
        """Initialize with the missing document ID."""
        self.document_id = document_id
        super().__init__(
            f"Document ID '{document_id}' not found in manifest.csv. "
            "Cannot attach provenance."
        )
