"""The assessment as the service stores and serves it [M5-02].

``domain.assessment.Assessment`` is the *grounded* assessment: its >=1-citation
invariant is unconditional, by design (``docs/ARCHITECTURE.md``, M5-01). But the
graph can finish an insufficient-context or clarification-exhausted run with a
recommendation that cites nothing, and ``GET /v1/assessments/{id}`` /
``ListAssessments`` must still serve those -- after the LangGraph thread, and its
checkpoint, may be long gone.

So the unit the ``AssessmentRepository`` persists and the read use cases return
is this application aggregate, not the domain entity. It carries the full
lifecycle: the system's opinion (verdict, prose, citations -- possibly empty),
the retrieval/clarification signals a reviewer needs to judge it, its status,
and -- once settled -- the analyst's ``HumanDecision`` recorded *beside* it,
never over it.

``as_domain_assessment()`` is the bridge back: it builds the domain
``Assessment`` and therefore raises ``CitationRequiredError`` for an abstain
record. That is the invariant doing its job, not a bug -- an abstain is
deliberately not a grounded assessment.

Standard library and domain/application types only (enforced by
tests/architecture/test_layer_boundaries.py).
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from application.consistency_flag import ConsistencyFlag
from application.orchestrator_result import OrchestratorResult
from domain.assessment import Assessment
from domain.citation import Citation
from domain.human_decision import HumanDecision
from domain.verdict import Verdict


class AssessmentStatus(Enum):
    """Where an assessment sits in the review lifecycle."""

    AWAITING_REVIEW = "awaiting_review"
    DECIDED = "decided"


@dataclass(frozen=True)
class AssessmentRecord:
    """One claim's assessment across its whole lifecycle -- the persisted unit."""

    assessment_id: str
    claim_id: str
    verdict: Verdict
    reasoning: str
    recommended_action: str
    citations: tuple[Citation, ...]
    confidence: float
    consistency_flags: tuple[ConsistencyFlag, ...]
    context_sufficient: bool | None
    clarification_exhausted: bool
    missing_information: tuple[str, ...]
    status: AssessmentStatus
    created_at: datetime
    decision: HumanDecision | None = None

    def __post_init__(self) -> None:
        """Enforce the non-empty fields, the types, and the status/decision pairing."""
        for name in ("assessment_id", "claim_id", "reasoning", "recommended_action"):
            if not getattr(self, name):
                raise ValueError(f"AssessmentRecord.{name} must not be empty")
        if not isinstance(self.verdict, Verdict):
            raise ValueError(
                f"AssessmentRecord.verdict must be a domain.verdict.Verdict, "
                f"got {self.verdict!r}"
            )
        if not isinstance(self.status, AssessmentStatus):
            raise ValueError(
                f"AssessmentRecord.status must be an AssessmentStatus, "
                f"got {self.status!r}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"AssessmentRecord.confidence must be in [0, 1], got {self.confidence}"
            )
        if not all(isinstance(c, Citation) for c in self.citations):
            raise ValueError(
                "AssessmentRecord.citations must all be Citation instances"
            )
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("AssessmentRecord.created_at must be timezone-aware")

        is_decided = self.status is AssessmentStatus.DECIDED
        if is_decided and self.decision is None:
            raise ValueError("a DECIDED record must carry a decision")
        if not is_decided and self.decision is not None:
            raise ValueError(
                f"a {self.status.value!r} record must not carry a decision"
            )
        if (
            self.decision is not None
            and self.decision.assessment_id != self.assessment_id
        ):
            raise ValueError(
                "the decision must reference the assessment it settled "
                f"({self.assessment_id!r})"
            )

    @property
    def is_grounded(self) -> bool:
        """Whether this cites >=1 clause -- i.e. projects to a domain ``Assessment``."""
        return len(self.citations) >= 1

    def as_domain_assessment(self) -> Assessment:
        """The grounded projection.

        Raises ``domain.errors.CitationRequiredError`` when ``citations`` is
        empty -- an abstain record is deliberately not a persistable
        ``Assessment``.
        """
        return Assessment(
            assessment_id=self.assessment_id,
            claim_id=self.claim_id,
            verdict=self.verdict,
            reasoning=self.reasoning,
            citations=self.citations,
            confidence=self.confidence,
            recommended_action=self.recommended_action,
        )

    @classmethod
    def from_orchestrator_result(
        cls,
        result: OrchestratorResult,
        *,
        assessment_id: str,
        claim_id: str,
        created_at: datetime,
        status: AssessmentStatus,
        decision: HumanDecision | None = None,
    ) -> "AssessmentRecord":
        """Build a record from an orchestrator run, plus the identity/lifecycle fields.

        The system's opinion (verdict, prose, citations, flags, signals) comes
        straight from ``result``; the caller supplies the ids, the timestamp,
        the status and -- on the resume path -- the analyst's decision.
        """
        return cls(
            assessment_id=assessment_id,
            claim_id=claim_id,
            verdict=result.verdict,
            reasoning=result.reasoning,
            recommended_action=result.recommended_action,
            citations=result.citations,
            confidence=result.confidence,
            consistency_flags=result.consistency_flags,
            context_sufficient=result.context_sufficient,
            clarification_exhausted=result.clarification_exhausted,
            missing_information=result.missing_information,
            status=status,
            created_at=created_at,
            decision=decision,
        )
