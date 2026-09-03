"""Port for running a claim through assessment -- LangGraph fully hidden [M5-02].

This is the contract that lets the application layer commission an assessment
without knowing one is produced by a LangGraph state machine. Nothing about the
graph crosses it: no ``ClaimState``, no ``Command``, no ``interrupt``, no
``thread_id``, none of the ``infrastructure.graph`` Pydantic models. In, a
domain ``Claim`` or ``HumanDecision``; out, an
``application.orchestrator_result.OrchestratorResult``.

``assessment_id`` is the sole run key. The implementation ([M5-04]) uses it as
the graph's ``thread_id``, so resuming a specific paused run needs nothing more.
``SubmitClaim`` mints a fresh ``assessment_id`` per submission, so re-submitting
a claim is simply a new run -- the "one claim, a second thread" case
``docs/HUMAN_CHECKPOINT.md`` describes falls out for free.

Both methods raise on an infrastructure failure (the graph, the model, the
database); the use case lets that propagate. A *contract* breach -- ``start``
not pausing, ``resume`` not finishing -- is caught by the use case as
``OrchestratorContractError``.
"""

from typing import Protocol

from application.orchestrator_result import OrchestratorResult
from domain.claim import Claim
from domain.human_decision import HumanDecision


class ClaimAssessmentOrchestrator(Protocol):
    """Commission and resume claim assessments."""

    def start(self, *, assessment_id: str, claim: Claim) -> OrchestratorResult:
        """Assess ``claim`` from scratch, up to the human checkpoint.

        Returns the system's recommendation with ``awaiting_review is True`` --
        the run is paused, waiting for a decision keyed by ``assessment_id``.
        """
        ...

    def resume(
        self, *, assessment_id: str, decision: HumanDecision
    ) -> OrchestratorResult:
        """Resume the paused run ``assessment_id`` with the analyst's decision.

        Runs to completion and returns the final result with
        ``awaiting_review is False``. The system's recommendation is unchanged
        from ``start``; an edit rides in ``decision``, recorded beside it.
        """
        ...
