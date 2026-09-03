"""Use case: record the analyst's decision and finish the run [M5-02].

Behind ``POST /v1/assessments/{id}/decision`` ([M5-04]). It validates the
decision payload against the domain rules, resumes the paused graph run past its
human checkpoint, and updates the stored record to ``DECIDED`` with the
``HumanDecision`` recorded *beside* the system's opinion -- never overwriting
it.

The system's verdict, prose and citations on the updated record are exactly what
``start`` produced; an ``edit`` lives entirely inside
``record.decision.edited_assessment``. Keeping a 0-citation abstain unchanged is
an ``approve``: an ``edit`` always supplies an ``EditedAssessmentInput``, which
becomes a domain ``Assessment`` and therefore always cites at least one clause
-- and every cited clause is checked against ``ClauseRepository`` before the
graph is resumed.

Ordering mirrors ``SubmitClaim``: the read and the validation happen first, the
orchestrator resume (a long call) runs outside any transaction, and only the
record update is transactional.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from application.assessment_record import AssessmentRecord, AssessmentStatus
from application.edited_assessment_input import EditedAssessmentInput
from application.errors import (
    AssessmentAlreadyDecidedError,
    AssessmentNotFoundError,
    OrchestratorContractError,
    UnknownClauseError,
)
from application.ports.assessment_repository import AssessmentRepository
from application.ports.claim_assessment_orchestrator import ClaimAssessmentOrchestrator
from application.ports.clause_repository import ClauseRepository
from application.ports.clock import Clock
from application.ports.unit_of_work import UnitOfWorkFactory
from domain.assessment import Assessment
from domain.citation import Citation
from domain.human_decision import DecisionOutcome, HumanDecision


@dataclass(frozen=True)
class SubmitHumanDecision:
    """Settle an awaiting-review assessment with the analyst's decision."""

    clock: Clock
    orchestrator: ClaimAssessmentOrchestrator
    assessments: AssessmentRepository
    clauses: ClauseRepository
    uow_factory: UnitOfWorkFactory

    def __call__(
        self,
        *,
        assessment_id: str,
        decision: DecisionOutcome,
        notes: str = "",
        edited: EditedAssessmentInput | None = None,
    ) -> AssessmentRecord:
        """Validate the decision, resume the run, and persist the settled record.

        Raises:
            AssessmentNotFoundError: no record exists for ``assessment_id``.
            AssessmentAlreadyDecidedError: the record is already ``DECIDED``.
            ValueError: an ``edit`` without an ``EditedAssessmentInput``, a
                non-``edit`` carrying one, or a decision the domain rejects.
            domain.errors.CitationRequiredError: an ``edit`` that cites nothing.
            UnknownClauseError: an ``edit`` citing a clause the corpus lacks.
            OrchestratorContractError: the run did not complete on resume.
        """
        record = self.assessments.get(assessment_id)
        if record is None:
            raise AssessmentNotFoundError(assessment_id)
        if record.status is AssessmentStatus.DECIDED:
            raise AssessmentAlreadyDecidedError(assessment_id)

        decided_at = self.clock.now()
        edited_assessment = self._build_edited_assessment(
            assessment_id=assessment_id,
            claim_id=record.claim_id,
            decision=decision,
            edited=edited,
        )
        human_decision = HumanDecision(
            assessment_id=assessment_id,
            decision=decision,
            decided_at=decided_at,
            notes=notes,
            edited_assessment=edited_assessment,
        )

        result = self.orchestrator.resume(
            assessment_id=assessment_id, decision=human_decision
        )
        if result.awaiting_review:
            raise OrchestratorContractError(
                f"resume did not complete the run for assessment {assessment_id!r}"
            )

        updated = AssessmentRecord.from_orchestrator_result(
            result,
            assessment_id=assessment_id,
            claim_id=record.claim_id,
            created_at=record.created_at,
            status=AssessmentStatus.DECIDED,
            decision=human_decision,
        )

        with self.uow_factory() as uow:
            uow.assessments.update(updated)
            uow.commit()

        return updated

    def _build_edited_assessment(
        self,
        *,
        assessment_id: str,
        claim_id: str,
        decision: DecisionOutcome,
        edited: EditedAssessmentInput | None,
    ) -> Assessment | None:
        """Turn an ``EditedAssessmentInput`` into a domain ``Assessment``, or ``None``.

        Enforces the edit/payload pairing here (a friendlier message than the
        one ``HumanDecision`` would give) and defers the >=1-citation and
        verdict rules to ``Assessment`` itself.
        """
        if decision is not DecisionOutcome.EDIT:
            if edited is not None:
                raise ValueError(
                    f"decision {getattr(decision, 'value', decision)!r} must not "
                    "carry an edited assessment"
                )
            return None
        if edited is None:
            raise ValueError("an edit decision requires an edited assessment")

        self._reject_unknown_clauses(edited.citations)
        return Assessment(
            assessment_id=assessment_id,
            claim_id=claim_id,
            verdict=edited.verdict,
            reasoning=edited.reasoning,
            citations=edited.citations,
            confidence=edited.confidence,
            recommended_action=edited.recommended_action,
        )

    def _reject_unknown_clauses(self, citations: Sequence[Citation]) -> None:
        """Raise ``UnknownClauseError`` if any cited clause is not in the corpus."""
        requested = [citation.clause_id for citation in citations]
        found = {clause.clause_id for clause in self.clauses.get_many(requested)}
        missing = [clause_id for clause_id in requested if clause_id not in found]
        if missing:
            raise UnknownClauseError(missing)
