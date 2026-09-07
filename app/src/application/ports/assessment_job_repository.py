"""Port for persisting and reading the queued-run lifecycle [M5-05].

The stored unit is [application.assessment_job.AssessmentJob] -- the run state a
caller polls (``PENDING`` -> ``RUNNING`` -> ``SUCCEEDED`` / ``FAILED``) plus what
the worker needs to rebuild the domain ``Claim``.

``add`` / ``update`` run inside a ``UnitOfWork`` (the worker's success path writes
the settled ``AssessmentRecord`` and flips the job to ``SUCCEEDED`` in one
transaction). ``get`` needs no transaction -- the read endpoints call it on a
bare repository, like ``AssessmentRepository.get``.
"""

from typing import Protocol

from application.assessment_job import AssessmentJob


class AssessmentJobRepository(Protocol):
    """Store queued assessment runs and read one back by id."""

    def add(self, job: AssessmentJob) -> None:
        """Persist a new job. The ``assessment_id`` is assumed unused."""
        ...

    def update(self, job: AssessmentJob) -> None:
        """Replace the stored job with the same ``assessment_id``."""
        ...

    def get(self, assessment_id: str) -> AssessmentJob | None:
        """Return the job for ``assessment_id``, or ``None`` if there is none."""
        ...
