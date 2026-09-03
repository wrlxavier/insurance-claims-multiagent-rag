"""Use case: fetch one assessment run's durable audit trail [M5-04].

The read behind ``GET /v1/assessments/{id}/audit``. It resolves the id against
``AssessmentRepository`` first -- an unknown id is a 404, not an empty trail --
then returns the entries the ``AuditTrailReader`` holds for it.

An assessment still ``AWAITING_REVIEW`` has an empty trail: the durable trail is
written once, in the ``human_review`` node, after the analyst decides. That is a
valid answer (an empty list), not an error.
"""

from dataclasses import dataclass

from application.audit_trail_entry import AuditTrailEntry
from application.errors import AssessmentNotFoundError
from application.ports.assessment_repository import AssessmentRepository
from application.ports.audit_trail_reader import AuditTrailReader


@dataclass(frozen=True)
class GetAuditTrail:
    """Return the audit trail for an assessment id, or raise if the id is unknown."""

    assessments: AssessmentRepository
    audit: AuditTrailReader

    def __call__(self, assessment_id: str) -> tuple[AuditTrailEntry, ...]:
        """Fetch the trail for ``assessment_id``.

        Raises:
            AssessmentNotFoundError: no assessment exists for the id.
        """
        if self.assessments.get(assessment_id) is None:
            raise AssessmentNotFoundError(assessment_id)
        return self.audit.get_trail(assessment_id)
