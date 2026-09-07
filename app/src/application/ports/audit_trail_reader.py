"""Port for reading a run's durable audit trail [M5-04].

The read behind ``GET /v1/assessments/{id}/audit``. Read-only reference over
the append-only ``audit_event`` table, standalone like ``ClauseRepository`` --
not part of the ``UnitOfWork`` (its one writer is the resume path, through
``UnitOfWork.audit``).

The trail is keyed by the graph ``thread_id``, which is the ``assessment_id``
(``ClaimAssessmentOrchestrator`` uses one as the other). It is written once, in
the ``human_review`` node, *after* the human decision -- so an assessment still
``AWAITING_REVIEW`` has an empty trail, and that is a valid answer, not an
error.

Standard library and application DTO types only (enforced by
tests/architecture/test_layer_boundaries.py).
"""

from typing import Protocol

from application.audit_trail_entry import AuditTrailEntry


class AuditTrailReader(Protocol):
    """Read the durable audit trail for one assessment run."""

    def get_trail(self, assessment_id: str) -> tuple[AuditTrailEntry, ...]:
        """Return the trail for ``assessment_id`` in ``sequence`` order.

        Empty when the run has not been decided yet (the trail is persisted at
        the human checkpoint, not before).
        """
        ...
