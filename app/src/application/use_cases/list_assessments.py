"""Use case: list assessments, newest first [M5-02].

The read behind a future ``GET /v1/assessments`` collection endpoint. A thin
pass-through to ``AssessmentRepository.list`` -- the ordering (newest
``created_at`` first, ``assessment_id`` as the tie-break), the optional
``claim_id`` / ``status`` filters and the ``limit`` / ``offset`` paging are the
repository's contract. Kept as a use case so the presentation layer depends on a
port, not on a repository method directly, and so a later access-control or
projection rule has one place to live.
"""

from dataclasses import dataclass

from application.assessment_record import AssessmentRecord, AssessmentStatus
from application.ports.assessment_repository import AssessmentRepository

_DEFAULT_LIMIT = 50


@dataclass(frozen=True)
class ListAssessments:
    """Return a page of assessment records, most recent first."""

    assessments: AssessmentRepository

    def __call__(
        self,
        *,
        claim_id: str | None = None,
        status: AssessmentStatus | None = None,
        limit: int = _DEFAULT_LIMIT,
        offset: int = 0,
    ) -> tuple[AssessmentRecord, ...]:
        """List records, optionally filtered by claim or status.

        Raises:
            ValueError: ``limit`` is not positive or ``offset`` is negative.
        """
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        if offset < 0:
            raise ValueError(f"offset must not be negative, got {offset}")
        return self.assessments.list(
            claim_id=claim_id, status=status, limit=limit, offset=offset
        )
