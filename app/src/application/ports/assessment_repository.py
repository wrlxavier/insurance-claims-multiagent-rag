"""Port for persisting and querying assessments [M5-02].

The stored unit is ``application.assessment_record.AssessmentRecord`` -- the
full-lifecycle aggregate, not the domain ``Assessment`` -- because an abstain
outcome (no citations) must be persisted and served, and a domain ``Assessment``
cannot represent one.

Writes (``add`` / ``update``) run inside a ``UnitOfWork``; reads (``get`` /
``list``) do not need a transaction and the use cases call them on a bare
repository. [M5-03] implements both against Postgres.
"""

from typing import Protocol

from application.assessment_record import AssessmentRecord, AssessmentStatus


class AssessmentRepository(Protocol):
    """Store assessment records and query them by claim or status."""

    def add(self, record: AssessmentRecord) -> None:
        """Persist a new record. The ``assessment_id`` is assumed unused."""
        ...

    def update(self, record: AssessmentRecord) -> None:
        """Replace the stored record with the same ``assessment_id``."""
        ...

    def get(self, assessment_id: str) -> AssessmentRecord | None:
        """Return the record for ``assessment_id``, or ``None`` if there is none."""
        ...

    def list(
        self,
        *,
        claim_id: str | None = None,
        status: AssessmentStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[AssessmentRecord, ...]:
        """Return records newest first (by ``created_at``, then ``assessment_id``).

        ``claim_id`` and ``status`` narrow the result when set; ``limit`` and
        ``offset`` page it.
        """
        ...
