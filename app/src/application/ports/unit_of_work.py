"""Port for the transactional boundary around assessment writes [M5-02].

A ``UnitOfWork`` is one transaction. It exposes the writers the use cases need in
one atomic step -- ``assessments`` (the record), ``audit`` (the trail the resume
path captured, [M5-04]'s transactional fold) and ``jobs`` (the queued-run
lifecycle, [M5-05]: the worker flips a job to ``SUCCEEDED`` in the same
transaction that persists the settled record) -- and nothing else: the clause
corpus is read-only reference data with its own lifecycle, so ``ClauseRepository``
is not part of it.

The use cases take a ``UnitOfWorkFactory`` (``Callable[[], UnitOfWork]``), not a
``UnitOfWork``, so each call opens its own transaction -- which is what
[M5-03]'s session-per-transaction implementation needs, and what lets a test's
in-memory fake give every call an isolated view.

Contract: entering the context begins the unit; ``commit()`` makes its writes
durable; leaving the context without a ``commit()`` -- normally or by exception
-- rolls everything back.
"""

from collections.abc import Callable
from types import TracebackType
from typing import Protocol

from application.ports.assessment_job_repository import AssessmentJobRepository
from application.ports.assessment_repository import AssessmentRepository
from application.ports.audit_trail_writer import AuditTrailWriter


class UnitOfWork(Protocol):
    """One transaction over the assessment store, the audit trail and the job log."""

    assessments: AssessmentRepository
    audit: AuditTrailWriter
    jobs: AssessmentJobRepository

    def __enter__(self) -> "UnitOfWork":
        """Begin the unit and return it."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        """Roll back if the block did not ``commit()``; never suppress an exception."""
        ...

    def commit(self) -> None:
        """Make every write in this unit durable."""
        ...

    def rollback(self) -> None:
        """Discard every write in this unit."""
        ...


UnitOfWorkFactory = Callable[[], UnitOfWork]
"""Opens a fresh ``UnitOfWork`` -- one per use-case invocation."""
