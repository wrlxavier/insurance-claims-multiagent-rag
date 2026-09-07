"""Use case: submit a claim for assessment [M5-02, M5-05].

Behind ``POST /v1/assessments``. It mints the identifiers, builds the domain
``Claim`` (whose construction validates the narrative and the timestamp),
persists an ``AssessmentJob`` in ``PENDING``, and hands it to the queue -- then
returns. The graph runs later, on a worker (``RunAssessment``), behind the 202.

Two identifiers, minted separately: a ``claim_id`` (the narrative) and an
``assessment_id`` (this run of it). Re-submitting the same claim is a fresh
``assessment_id`` -- a second run, a second job.

[M5-05] changed the shape: M5-04's version ran ``orchestrator.start`` inline
before responding (a multi-minute call held in the request handler). Now the only
synchronous work is a single-row insert. ``submitted_at`` is stamped here, once,
and stored on the job so a retry re-runs the graph against the *same* loss-date
baseline the consistency checks compare against.

The job is committed *before* it is enqueued: a consumer polling
``GET /v1/assessments/{id}`` must always find the row. The reverse would leave a
window where the id is queued but unreadable. The residual window -- committed but
never enqueued, e.g. Redis is down between the commit and the ``enqueue`` call --
leaves the job ``PENDING`` forever; a reconciler that re-enqueues stale
``PENDING`` jobs is left to operations (noted in ``docs/ASYNC_PROCESSING.md``).
"""

from collections.abc import Callable
from dataclasses import dataclass

from application.assessment_job import AssessmentJob, JobStatus
from application.ports.assessment_queue import AssessmentQueue
from application.ports.clock import Clock
from application.ports.unit_of_work import UnitOfWorkFactory
from domain.claim import Claim
from domain.susep_process import SusepProcess


@dataclass(frozen=True)
class SubmitClaim:
    """Accept a submitted claim, persist it as a pending job, and queue it."""

    clock: Clock
    queue: AssessmentQueue
    uow_factory: UnitOfWorkFactory
    new_id: Callable[[], str]

    def __call__(
        self,
        *,
        raw_text: str,
        policy_ref: SusepProcess | None = None,
        claim_id: str | None = None,
    ) -> AssessmentJob:
        """Store a ``PENDING`` job for the claim and enqueue it for a worker.

        Raises:
            ValueError: the narrative is empty or the clock returned a naive
                datetime (both surface from ``Claim`` construction).
        """
        submitted_at = self.clock.now()
        resolved_claim_id = claim_id or self.new_id()
        assessment_id = self.new_id()

        # Constructed for its validation side effects (empty narrative, naive
        # timestamp); the worker rebuilds it from the persisted fields.
        Claim(
            claim_id=resolved_claim_id,
            raw_text=raw_text,
            submitted_at=submitted_at,
            policy_ref=policy_ref,
        )

        job = AssessmentJob(
            assessment_id=assessment_id,
            claim_id=resolved_claim_id,
            raw_text=raw_text,
            policy_ref=policy_ref.value if policy_ref is not None else None,
            submitted_at=submitted_at,
            status=JobStatus.PENDING,
            attempts=0,
            created_at=submitted_at,
            updated_at=submitted_at,
        )

        with self.uow_factory() as uow:
            uow.jobs.add(job)
            uow.commit()

        self.queue.enqueue(assessment_id)

        return job
