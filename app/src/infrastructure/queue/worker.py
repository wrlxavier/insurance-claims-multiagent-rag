"""Run the assessment worker pool -- ``make worker`` [M5-05].

``run_worker`` starts ``QueueSettings.assessment_worker_concurrency`` RQ workers
against the ``assessments`` queue. That number is the DoD's configurable
parallelism bound: each worker runs one assessment at a time, so it is also the
count of concurrent graph runs hitting the LLM provider.

``SimpleWorker`` (no fork per job) is deliberate: the retrieval stack is
expensive to load, and a forking worker would reload it for every job. Each pool
worker loads it once (lazily, in ``tasks._runner_singleton``) and reuses it.

``stop_retry_on_permanent`` is the transient/real split at the queue boundary:
RQ's ``Retry`` reschedules on any exception, so a real
``PermanentAssessmentError`` has its retry budget zeroed here and drops straight
into the ``FailedJobRegistry`` (the dead-letter), traceback preserved. A
``TransientAssessmentError`` is left alone -- RQ retries it with backoff until the
budget runs out, then dead-letters it too.
"""

from __future__ import annotations

import logging
from types import TracebackType

import redis
from rq import SimpleWorker
from rq.job import Job
from rq.worker_pool import WorkerPool

from application.errors import PermanentAssessmentError
from infrastructure.config.settings import (
    get_observability_settings,
    get_queue_settings,
)
from infrastructure.queue.rq_queue import QUEUE_NAME

logger = logging.getLogger(__name__)


def stop_retry_on_permanent(
    job: Job,
    exc_type: type[BaseException],
    exc_value: BaseException,
    traceback: TracebackType | None,
) -> bool:
    """Cancel the retry budget for a real failure so it dead-letters now."""
    if isinstance(exc_value, PermanentAssessmentError):
        job.retries_left = 0
    return True  # let RQ's default failure handling proceed


def run_worker() -> None:
    """Start the worker pool and block until it is stopped."""
    logging.basicConfig(level=get_observability_settings().log_level)
    settings = get_queue_settings()
    connection = redis.Redis.from_url(settings.redis_url)

    pool = WorkerPool(
        [QUEUE_NAME],
        connection=connection,
        num_workers=settings.assessment_worker_concurrency,
        worker_class=SimpleWorker,
        exception_handlers=[stop_retry_on_permanent],
    )
    logger.info(
        "assessment worker pool starting: %d worker(s) on %r",
        settings.assessment_worker_concurrency,
        QUEUE_NAME,
    )
    pool.start()
