"""Use case: run one queued assessment on a worker [M5-05].

The background half of ``POST /v1/assessments``. A Redis worker calls this with
an ``assessment_id``; it loads the ``AssessmentJob``, runs the graph to the human
checkpoint through the orchestrator port, and -- on success -- writes the
``AssessmentRecord`` the read endpoints serve, flipping the job to ``SUCCEEDED``
in the *same* transaction.

Failure handling is the DoD's "distinguish transient from real":

- a transient provider fault (rate limit, 5xx, dropped connection -- decided by
  the injected ``is_transient`` predicate) with attempts left: the job goes back
  to ``PENDING`` and a ``TransientAssessmentError`` is raised so the queue
  reschedules it with backoff;
- the same fault with no attempts left, or any real error / contract breach: the
  job is marked ``FAILED`` with the cause preserved on ``JobFailure`` and a
  ``PermanentAssessmentError`` (or ``TransientAssessmentError`` when the budget
  simply ran out) is raised -- the queue dead-letters it.

Idempotent: a redelivered job whose row already reads ``SUCCEEDED`` / ``FAILED``
is a no-op. A job stuck at ``RUNNING`` (the worker died mid-run) is re-run -- the
LangGraph checkpoint means that resumes from the last completed node, not from
scratch.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

from application.assessment_job import AssessmentJob, JobFailure, JobStatus
from application.assessment_record import AssessmentRecord, AssessmentStatus
from application.errors import (
    AssessmentRunError,
    PermanentAssessmentError,
    TransientAssessmentError,
)
from application.ports.claim_assessment_orchestrator import ClaimAssessmentOrchestrator
from application.ports.clock import Clock
from application.ports.unit_of_work import UnitOfWorkFactory
from domain.claim import Claim
from domain.susep_process import SusepProcess

_MAX_CAUSE_CHARS = 2000


@dataclass(frozen=True)
class RunAssessment:
    """Process one queued assessment run to the human checkpoint, or fail it."""

    clock: Clock
    orchestrator: ClaimAssessmentOrchestrator
    uow_factory: UnitOfWorkFactory
    is_transient: Callable[[BaseException], bool]
    max_attempts: int

    def __call__(self, assessment_id: str) -> None:
        """Run the job ``assessment_id``; persist the record or the failure.

        Raises:
            PermanentAssessmentError: the job is unknown, the run hit a real
                error, or the orchestrator broke its contract.
            TransientAssessmentError: the run hit a transient provider fault --
                the queue reschedules (attempts left) or dead-letters it.
        """
        job = self._load(assessment_id)
        if job is None:
            raise PermanentAssessmentError(
                f"no assessment job found for id {assessment_id!r}"
            )
        if job.is_terminal:
            return

        running = self._mark_running(job)
        claim = self._rebuild_claim(running)

        try:
            result = self.orchestrator.start(assessment_id=assessment_id, claim=claim)
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed run error below
            raise self._record_failure(running, exc) from exc

        if not result.awaiting_review:
            breach = _ContractBreachError(
                f"start did not pause at the human checkpoint for "
                f"assessment {assessment_id!r}"
            )
            raise self._record_failure(running, breach) from breach

        record = AssessmentRecord.from_orchestrator_result(
            result,
            assessment_id=assessment_id,
            claim_id=running.claim_id,
            created_at=running.created_at,
            status=AssessmentStatus.AWAITING_REVIEW,
        )
        succeeded = replace(
            running,
            status=JobStatus.SUCCEEDED,
            failure=None,
            updated_at=self.clock.now(),
        )
        with self.uow_factory() as uow:
            uow.assessments.add(record)
            uow.jobs.update(succeeded)
            uow.commit()

    def _load(self, assessment_id: str) -> AssessmentJob | None:
        with self.uow_factory() as uow:
            return uow.jobs.get(assessment_id)

    def _mark_running(self, job: AssessmentJob) -> AssessmentJob:
        running = replace(
            job,
            status=JobStatus.RUNNING,
            attempts=job.attempts + 1,
            failure=None,
            updated_at=self.clock.now(),
        )
        with self.uow_factory() as uow:
            uow.jobs.update(running)
            uow.commit()
        return running

    def _rebuild_claim(self, job: AssessmentJob) -> Claim:
        return Claim(
            claim_id=job.claim_id,
            raw_text=job.raw_text,
            submitted_at=job.submitted_at,
            policy_ref=(
                SusepProcess.parse(job.policy_ref)
                if job.policy_ref is not None
                else None
            ),
        )

    def _record_failure(
        self, running: AssessmentJob, exc: BaseException
    ) -> AssessmentRunError:
        """Persist the job's failure state and return the typed error to raise."""
        transient = not isinstance(exc, _ContractBreachError) and self.is_transient(exc)
        retryable = transient and running.attempts < self.max_attempts
        now = self.clock.now()
        failure = JobFailure(
            kind="transient" if transient else "permanent",
            error_type=type(exc).__name__,
            message=str(exc)[:_MAX_CAUSE_CHARS] or type(exc).__name__,
            failed_at=now,
        )
        failed = replace(
            running,
            status=JobStatus.PENDING if retryable else JobStatus.FAILED,
            failure=failure,
            updated_at=now,
        )
        with self.uow_factory() as uow:
            uow.jobs.update(failed)
            uow.commit()

        error: AssessmentRunError = (
            TransientAssessmentError(str(exc))
            if transient
            else PermanentAssessmentError(str(exc))
        )
        error.__cause__ = exc
        return error


class _ContractBreachError(Exception):
    """``start`` returned a non-paused result -- a permanent failure."""
