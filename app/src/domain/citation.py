"""The citation value object [M5-01].

The stdlib-dataclass twin of ``infrastructure.graph.state.Citation``: one
clause an assertion is traceable to. The graph produces the Pydantic
version inside a run; this is the shape the domain entities
([domain.assessment.Assessment]) and the application ports ([M5-02]) speak
in, with the mapper between the two owned by [M5-03].

``excerpt`` is a quoted span of the clause as a human reads it.
``relevance_score`` is the ranker's score for this clause against the
query; a structurally co-retrieved exclusion the ranking itself missed
carries ``0.0`` (``docs/ARCHITECTURE.md``, M3-06).

Standard library only -- enforced by
tests/architecture/test_layer_boundaries.py.
"""

from dataclasses import dataclass

from domain.clause_classification import ClauseType
from domain.susep_process import SusepProcess


@dataclass(frozen=True)
class Citation:
    """One clause an assessment's reasoning is grounded in."""

    clause_id: str
    document_id: str
    susep_process: SusepProcess
    clause_type: ClauseType
    excerpt: str
    relevance_score: float = 0.0

    def __post_init__(self) -> None:
        """Reject empty identifiers or excerpt, a negative score, or a bad type."""
        for name in ("clause_id", "document_id", "excerpt"):
            if not getattr(self, name):
                raise ValueError(f"Citation.{name} must not be empty")
        if self.relevance_score < 0.0:
            raise ValueError(
                f"Citation.relevance_score must be >= 0, got {self.relevance_score}"
            )
        if not isinstance(self.clause_type, ClauseType):
            raise ValueError(
                f"Citation.clause_type must be a ClauseType, got {self.clause_type!r}"
            )
