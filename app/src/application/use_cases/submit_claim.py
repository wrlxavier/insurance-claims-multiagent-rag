"""Use case: submit a claim for assessment [M5-02].

Behind ``POST /v1/assessments`` ([M5-04]). It mints the identifiers, builds the
domain ``Claim`` (whose construction validates the narrative and the
timestamp), runs it through the orchestrator up to the human checkpoint, and
persists the result as an ``AssessmentRecord`` in status ``AWAITING_REVIEW``.

Two identifiers, minted separately: a ``claim_id`` (the narrative) and an
``assessment_id`` (this run of it). Re-submitting the same claim is a fresh
``assessment_id`` -- a second run, a second record.

The orchestrator runs *before* the transaction opens: the graph is a long call
and must not hold a database transaction open, and it must pause at the
checkpoint before there is anything worth persisting. A failure to persist after
the graph has paused leaves the run recoverable from its checkpoint but without
a record -- a narrow window [M5-03] closes by writing the record in the same
transaction as the audit trail.
"""

from collections.abc import Callable
from dataclasses import dataclass

from application.assessment_record import AssessmentRecord, AssessmentStatus
from application.errors import OrchestratorContractError
from application.ports.claim_assessment_orchestrator import ClaimAssessmentOrchestrator
from application.ports.clock import Clock
from application.ports.unit_of_work import UnitOfWorkFactory
from domain.claim import Claim
from domain.susep_process import SusepProcess


@dataclass(frozen=True)
class SubmitClaim:
    """Assess a submitted claim and persist the pending result."""

    clock: Clock
    orchestrator: ClaimAssessmentOrchestrator
    uow_factory: UnitOfWorkFactory
    new_id: Callable[[], str]

    def __call__(
        self,
        *,
        raw_text: str,
        policy_ref: SusepProcess | None = None,
        claim_id: str | None = None,
    ) -> AssessmentRecord:
        """Run the claim through assessment and store the awaiting-review record.

        Raises:
            ValueError: the narrative is empty or the clock returned a naive
                datetime (both surface from ``Claim`` construction).
            OrchestratorContractError: the orchestrator did not pause at the
                human checkpoint.
        """
        submitted_at = self.clock.now()
        resolved_claim_id = claim_id or self.new_id()
        assessment_id = self.new_id()

        claim = Claim(
            claim_id=resolved_claim_id,
            raw_text=raw_text,
            submitted_at=submitted_at,
            policy_ref=policy_ref,
        )

        result = self.orchestrator.start(assessment_id=assessment_id, claim=claim)
        if not result.awaiting_review:
            raise OrchestratorContractError(
                f"start did not pause at the human checkpoint for "
                f"assessment {assessment_id!r}"
            )

        record = AssessmentRecord.from_orchestrator_result(
            result,
            assessment_id=assessment_id,
            claim_id=resolved_claim_id,
            created_at=submitted_at,
            status=AssessmentStatus.AWAITING_REVIEW,
        )

        with self.uow_factory() as uow:
            uow.assessments.add(record)
            uow.commit()

        return record
