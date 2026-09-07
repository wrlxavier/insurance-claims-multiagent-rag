"""The assessment entity [M5-01].

The settled compatibility assessment for one claim against a registered
product -- the stdlib-dataclass twin of
``infrastructure.graph.state.CompatibilityAssessment``, plus the
``recommended_action`` the reviewer acts on (so ``GET /v1/assessments/{id}``
has one entity to return, not two).

Invariants, all checked at construction:

* **at least one citation, always** -- including for an
  ``insufficient_information`` verdict. The graph's abstain-on-empty-
  retrieval path produces a verdict with no citations; that state is *not*
  a persistable ``Assessment`` (it lives in claim state + the audit trail),
  a fact [M5-02]/[M5-03] must account for.
* **the verdict is a ``domain.verdict.Verdict`` member** -- a plain
  dataclass does not type-check its fields, so a bare string would slip
  through without this guard.

Consistency signals are deliberately out of scope: no M5-01 invariant
touches them, and persisting ``ConsistencyReport`` is [M5-03]'s call.

Standard library only -- enforced by
tests/architecture/test_layer_boundaries.py.
"""

from dataclasses import dataclass

from domain.citation import Citation
from domain.errors import CitationRequiredError, VerdictNotPermittedError
from domain.verdict import Verdict


@dataclass(frozen=True)
class Assessment:
    """A compatibility verdict for a claim, grounded in one or more citations."""

    assessment_id: str
    claim_id: str
    verdict: Verdict
    reasoning: str
    citations: tuple[Citation, ...]
    confidence: float
    recommended_action: str

    def __post_init__(self) -> None:
        """Enforce the non-empty fields, the verdict type and the >=1-citation rule."""
        for name in ("assessment_id", "claim_id", "reasoning", "recommended_action"):
            if not getattr(self, name):
                raise ValueError(f"Assessment.{name} must not be empty")
        if not isinstance(self.verdict, Verdict):
            raise VerdictNotPermittedError(
                f"verdict must be a domain.verdict.Verdict, got {self.verdict!r}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Assessment.confidence must be in [0, 1], got {self.confidence}"
            )
        if len(self.citations) < 1:
            raise CitationRequiredError(self.assessment_id)
        if not all(isinstance(c, Citation) for c in self.citations):
            raise ValueError("Assessment.citations must all be Citation instances")
