"""The ``AssessmentQueue`` adapter -- an RQ job on a Redis-backed queue [M5-05].

``RqAssessmentQueue.enqueue`` turns a persisted ``PENDING`` job into an RQ job:

- ``job_id`` == ``assessment_id`` -- the id a caller polls is the queue key, so a
  re-submission (a fresh ``assessment_id``) is a distinct job and a redelivery
  reuses the same one;
- ``retry`` -- RQ's native retry-with-backoff, sized from
  ``QueueSettings.assessment_max_retries`` / ``rq_retry_intervals``. RQ retries on
  *any* exception; the worker's ``stop_retry_on_permanent`` handler cancels the
  budget for a real (non-transient) failure so it dead-letters immediately;
- ``job_timeout`` -- a 207-page assessment is minutes of work, not the RQ default.
"""

from __future__ import annotations

import redis
from rq import Queue, Retry

from infrastructure.config.settings import QueueSettings

QUEUE_NAME = "assessments"
DEFAULT_JOB_PATH = "infrastructure.queue.tasks.run_assessment_job"


class RqAssessmentQueue:
    """Schedule assessment runs on an RQ queue. Implements ``AssessmentQueue``."""

    def __init__(
        self,
        queue: Queue,
        *,
        job_path: str = DEFAULT_JOB_PATH,
        max_attempts: int,
        retry_intervals: list[int],
        job_timeout: int,
    ) -> None:
        """Wire the adapter to its RQ queue and retry/timeout policy."""
        self._queue = queue
        self._job_path = job_path
        self._max_attempts = max_attempts
        self._retry_intervals = retry_intervals
        self._job_timeout = job_timeout

    def enqueue(self, assessment_id: str) -> None:
        """Queue ``assessment_id`` for a worker, with retry + timeout attached."""
        retry: Retry | None = None
        if self._max_attempts > 1:
            retry = Retry(
                max=self._max_attempts - 1,
                interval=self._retry_intervals or 0,
            )
        self._queue.enqueue(
            self._job_path,
            assessment_id,
            job_id=assessment_id,
            job_timeout=self._job_timeout,
            retry=retry,
        )


def build_assessment_queue(
    settings: QueueSettings,
    *,
    connection: redis.Redis | None = None,
    job_path: str = DEFAULT_JOB_PATH,
) -> RqAssessmentQueue:
    """Build the queue adapter from settings (opening a Redis connection if needed)."""
    conn = connection or redis.Redis.from_url(settings.redis_url)
    return RqAssessmentQueue(
        Queue(QUEUE_NAME, connection=conn),
        job_path=job_path,
        max_attempts=settings.assessment_max_retries,
        retry_intervals=settings.rq_retry_intervals,
        job_timeout=settings.assessment_job_timeout_seconds,
    )
