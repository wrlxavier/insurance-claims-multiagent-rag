"""The human-decision entity [M5-01].

What the analyst decided at the checkpoint ([M4-09]) -- the stdlib-dataclass
twin of ``infrastructure.graph.state.HumanDecision``. Recorded alongside,
never overwriting, the system's ``Assessment``.

The M5-01 invariant the graph's version lacks: a decision **always
references the assessment it acted on** (``assessment_id``). ``edited_
assessment`` carries the analyst's revised opinion and is present exactly
when the decision is ``EDIT`` -- and, when present, must revise *that same*
assessment.

Standard library only -- enforced by
tests/architecture/test_layer_boundaries.py.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from domain.assessment import Assessment
from domain.errors import DecisionMustReferenceAssessmentError


class DecisionOutcome(Enum):
    """What the analyst did with the system's recommendation.

    Values match ``infrastructure.graph.state.HumanDecision.decision``'s
    ``Literal`` so [M5-03]'s mapper is a ``DecisionOutcome(value)`` round trip.
    """

    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"


@dataclass(frozen=True)
class HumanDecision:
    """The analyst's recorded decision on one assessment."""

    assessment_id: str
    decision: DecisionOutcome
    decided_at: datetime
    notes: str = ""
    edited_assessment: Assessment | None = None

    def __post_init__(self) -> None:
        """Enforce the assessment reference, the timezone and edit consistency."""
        if not self.assessment_id:
            raise DecisionMustReferenceAssessmentError(
                "HumanDecision.assessment_id must reference the assessment acted on"
            )
        if not isinstance(self.decision, DecisionOutcome):
            raise ValueError(
                f"HumanDecision.decision must be a DecisionOutcome, "
                f"got {self.decision!r}"
            )
        if self.decided_at.tzinfo is None or self.decided_at.utcoffset() is None:
            raise ValueError("HumanDecision.decided_at must be timezone-aware")
        is_edit = self.decision is DecisionOutcome.EDIT
        if is_edit and self.edited_assessment is None:
            raise ValueError("decision EDIT requires an edited_assessment")
        if not is_edit and self.edited_assessment is not None:
            raise ValueError(
                f"decision {self.decision.value!r} must not carry an edited_assessment"
            )
        if (
            self.edited_assessment is not None
            and self.edited_assessment.assessment_id != self.assessment_id
        ):
            raise ValueError(
                "edited_assessment must revise the same assessment the decision "
                f"references ({self.assessment_id!r})"
            )
