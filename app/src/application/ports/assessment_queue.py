"""Port for handing a persisted assessment run to the worker pool [M5-05].

``SubmitClaim`` persists an ``AssessmentJob`` in ``PENDING`` and then calls this
port to schedule it. The application layer knows nothing of Redis or RQ -- the
adapter ([infrastructure.queue.rq_queue.RqAssessmentQueue]) turns ``enqueue`` into
a Redis-backed job with the retry policy and timeout attached.

``enqueue`` takes only the ``assessment_id``: the worker reloads everything else
from the job row, so a redelivered or retried job always runs against the current
persisted state.
"""

from typing import Protocol


class AssessmentQueue(Protocol):
    """Schedule a persisted assessment run for background processing."""

    def enqueue(self, assessment_id: str) -> None:
        """Queue the ``PENDING`` job ``assessment_id`` for a worker to pick up."""
        ...
