"""What ``ClaimAssessmentOrchestrator`` hands back to a use case [M5-02].

The orchestrator port hides LangGraph entirely, so its result cannot be the
graph's ``ClaimState`` or its Pydantic ``Recommendation`` -- both live in
``infrastructure`` and the application layer must not import them. This is the
graph-free projection the use cases and ``AssessmentRecord`` work in: the
verdict and prose a reviewer sees, the clauses it is grounded in, the
consistency flags kept beside it, the two retrieval/clarification signals the
[M5-04] read model needs, and whether the run paused at the human checkpoint.

``verdict`` is a required field here even though the graph's ``Recommendation``
has none -- deriving it (from the recommendation node's audit record) is the
adapter's job, done once at the boundary so the application layer never re-reads
graph internals.

``citations`` may be empty: the insufficient-context and clarification-exhausted
paths produce a recommendation with no clauses. That is a valid
``OrchestratorResult`` and a valid ``AssessmentRecord``; it is only not a
persistable domain ``Assessment``.

Standard library and domain types only (enforced by
tests/architecture/test_layer_boundaries.py).
"""

from dataclasses import dataclass

from application.consistency_flag import ConsistencyFlag
from domain.citation import Citation
from domain.verdict import Verdict


@dataclass(frozen=True)
class OrchestratorResult:
    """One assessment run's outcome, with no trace of the graph that produced it."""

    verdict: Verdict
    reasoning: str
    recommended_action: str
    citations: tuple[Citation, ...]
    confidence: float
    consistency_flags: tuple[ConsistencyFlag, ...]
    context_sufficient: bool | None
    clarification_exhausted: bool
    missing_information: tuple[str, ...]
    awaiting_review: bool

    def __post_init__(self) -> None:
        """Reject empty prose, a non-``Verdict`` verdict or a bad confidence."""
        for name in ("reasoning", "recommended_action"):
            if not getattr(self, name):
                raise ValueError(f"OrchestratorResult.{name} must not be empty")
        if not isinstance(self.verdict, Verdict):
            raise ValueError(
                f"OrchestratorResult.verdict must be a domain.verdict.Verdict, "
                f"got {self.verdict!r}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"OrchestratorResult.confidence must be in [0, 1], "
                f"got {self.confidence}"
            )
