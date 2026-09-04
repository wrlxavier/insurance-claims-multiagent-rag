"""The function RQ runs on a worker for one queued assessment [M5-05].

RQ enqueues a dotted path (``infrastructure.queue.tasks.run_assessment_job``) and
each worker process imports and calls it. The heavy dependencies -- the chat
models, the retrieval stack, the LangGraph orchestrator -- are built once per
worker process and reused across jobs, behind a lazy, thread-safe singleton. The
build is deferred to the first job (not import time) so it happens *after* the
worker pool has forked: torch and sentence-transformers do not survive a fork
cleanly, and each worker loading its own copy post-fork is the safe shape.

Tests substitute a different job path (``tests.integration._queue_fakes``) rather
than reaching in here, so this module has no test seam.
"""

from __future__ import annotations

import threading

from application.use_cases.run_assessment import RunAssessment
from infrastructure.bootstrap import build_core_components
from infrastructure.clock import SystemClock
from infrastructure.config.settings import get_queue_settings
from infrastructure.llm_errors import is_transient_llm_error
from infrastructure.observability.correlation import (
    bind_correlation_id,
    generate_correlation_id,
)

_lock = threading.Lock()
_runner: RunAssessment | None = None


def build_runner() -> RunAssessment:
    """Build a fully-wired ``RunAssessment`` from the process-wide composition root."""
    core = build_core_components()
    settings = get_queue_settings()
    return RunAssessment(
        clock=SystemClock(),
        orchestrator=core.orchestrator,
        uow_factory=core.uow_factory,
        is_transient=is_transient_llm_error,
        max_attempts=settings.assessment_max_retries,
    )


def _runner_singleton() -> RunAssessment:
    global _runner
    if _runner is None:
        with _lock:
            if _runner is None:
                _runner = build_runner()
    return _runner


def run_assessment_job(assessment_id: str, correlation_id: str | None = None) -> None:
    """RQ entry point: process one queued assessment run to the human checkpoint.

    ``correlation_id`` is the id the enqueue side ([M5-06]) carried across Redis;
    binding it here means the graph run and its node log lines trace back to the
    originating request. Defaulted so an older queued job (no second arg) still
    runs, with a fresh id.
    """
    with bind_correlation_id(correlation_id or generate_correlation_id()):
        _runner_singleton()(assessment_id)
