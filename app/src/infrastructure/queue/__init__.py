"""The [M5-05] asynchronous-processing queue: RQ over Redis.

``RqAssessmentQueue`` is the ``AssessmentQueue`` port's adapter (used by the API
composition root); ``tasks.run_assessment_job`` is the function RQ runs on a
worker; ``worker.run_worker`` is ``make worker``.
"""

from infrastructure.queue.rq_queue import (
    DEFAULT_JOB_PATH,
    QUEUE_NAME,
    RqAssessmentQueue,
    build_assessment_queue,
)

__all__ = [
    "DEFAULT_JOB_PATH",
    "QUEUE_NAME",
    "RqAssessmentQueue",
    "build_assessment_queue",
]
