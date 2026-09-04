"""Use case: fetch one assessment by id [M5-02, M5-05].

The read behind ``GET /v1/assessments/{id}``. It returns an
``AssessmentReadModel``: the lifecycle ``status`` a caller polls, plus the full
``AssessmentRecord`` once the graph has produced it.

[M5-05] made the read span the whole lifecycle. The id handed back in the 202
must resolve immediately -- while the run is ``pending`` / ``running``, and after
it has ``failed`` -- so this checks the ``AssessmentRepository`` first (a
completed run, ``awaiting_review`` / ``decided``) and falls back to the
``AssessmentJobRepository`` (still queued, or dead-lettered). Only an id neither
store knows is an error.
"""

from dataclasses import dataclass

from application.assessment_read_model import AssessmentReadModel
from application.errors import AssessmentNotFoundError
from application.ports.assessment_job_repository import AssessmentJobRepository
from application.ports.assessment_repository import AssessmentRepository


@dataclass(frozen=True)
class GetAssessment:
    """Return the lifecycle view for an assessment id, or raise if it is unknown."""

    assessments: AssessmentRepository
    jobs: AssessmentJobRepository

    def __call__(self, assessment_id: str) -> AssessmentReadModel:
        """Fetch the read model for ``assessment_id``.

        Raises:
            AssessmentNotFoundError: neither a record nor a job exists for the id.
        """
        record = self.assessments.get(assessment_id)
        if record is not None:
            return AssessmentReadModel.from_record(record)

        job = self.jobs.get(assessment_id)
        if job is not None:
            return AssessmentReadModel.from_job(job)

        raise AssessmentNotFoundError(assessment_id)
