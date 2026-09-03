"""Port for persisting a run's audit trail inside a unit of work [M5-04].

Exposed as ``UnitOfWork.audit``. Its one caller is ``SubmitHumanDecision``:
the resume path captures the trail the ``human_review`` node produced (rather
than letting that node commit it on its own) and writes it here, in the same
transaction as the settled ``AssessmentRecord`` -- so a crash between the two
can no longer leave the trail durable while the record still reads
``AWAITING_REVIEW`` (``docs/ARCHITECTURE.md``, the [M5-04] transactional fold).

Insert-only and idempotent on ``(thread_id, sequence)``: the same trail
re-submitted (the self-healing decision retry) rewrites the same rows harmlessly.

Standard library and application DTO types only (enforced by
tests/architecture/test_layer_boundaries.py).
"""

from collections.abc import Sequence
from typing import Protocol

from application.audit_trail_entry import AuditTrailEntry


class AuditTrailWriter(Protocol):
    """Append audit-trail entries within the current transaction."""

    def append(
        self,
        *,
        claim_id: str,
        thread_id: str,
        entries: Sequence[AuditTrailEntry],
    ) -> None:
        """Persist ``entries`` for one run, skipping any already written.

        The caller owns the transaction: this flushes but never commits.
        """
        ...
