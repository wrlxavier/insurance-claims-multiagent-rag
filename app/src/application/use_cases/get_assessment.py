"""Use case: fetch one assessment by id [M5-02].

The read behind ``GET /v1/assessments/{id}`` ([M5-04]). It returns the stored
``AssessmentRecord`` -- system verdict, recommendation, citations, the
retrieval/clarification signals and, once settled, the analyst's decision -- so
the API never re-enters the graph to answer a read.

An abstain record (no citations) is returned like any other; only an unknown id
is an error.
"""

from dataclasses import dataclass

from application.assessment_record import AssessmentRecord
from application.errors import AssessmentNotFoundError
from application.ports.assessment_repository import AssessmentRepository


@dataclass(frozen=True)
class GetAssessment:
    """Return the assessment record for an id, or raise if there is none."""

    assessments: AssessmentRepository

    def __call__(self, assessment_id: str) -> AssessmentRecord:
        """Fetch the record for ``assessment_id``.

        Raises:
            AssessmentNotFoundError: no record exists for the id.
        """
        record = self.assessments.get(assessment_id)
        if record is None:
            raise AssessmentNotFoundError(assessment_id)
        return record
