"""The payload a reviewer submits when editing an assessment [M5-02].

An ``edit`` decision at the human checkpoint replaces the system's opinion with
the analyst's. This is the raw shape of that replacement as it enters
``SubmitHumanDecision`` -- verdict, prose and the clauses the analyst grounds it
in. The use case turns it into a domain ``Assessment`` (which enforces the
>=1-citation rule and the permitted verdict set) and validates every cited
clause against ``ClauseRepository`` before handing it to the orchestrator.

Keeping a 0-citation abstain unchanged is an ``approve``, not an ``edit``: an
``EditedAssessmentInput`` always names at least one clause, because a domain
``Assessment`` always does.

Standard library only -- the application layer imports no ``infrastructure``
(enforced by tests/architecture/test_layer_boundaries.py).
"""

from dataclasses import dataclass

from domain.citation import Citation
from domain.verdict import Verdict


@dataclass(frozen=True)
class EditedAssessmentInput:
    """A reviewer's revised assessment, before it becomes a domain ``Assessment``."""

    verdict: Verdict
    reasoning: str
    recommended_action: str
    citations: tuple[Citation, ...]
    confidence: float

    def __post_init__(self) -> None:
        """Reject empty prose or a non-``Verdict`` verdict (citations: domain's job).

        The >=1-citation and ``[0, 1]`` confidence rules are left to
        ``domain.assessment.Assessment`` so there is exactly one definition of
        each; this guard only covers the fields the use case reads before
        constructing that entity.
        """
        for name in ("reasoning", "recommended_action"):
            if not getattr(self, name):
                raise ValueError(f"EditedAssessmentInput.{name} must not be empty")
        if not isinstance(self.verdict, Verdict):
            raise ValueError(
                f"EditedAssessmentInput.verdict must be a domain.verdict.Verdict, "
                f"got {self.verdict!r}"
            )
