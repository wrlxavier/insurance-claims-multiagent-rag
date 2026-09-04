#!/usr/bin/env python3
"""Run the assessment worker pool -- ``make worker`` [M5-05].

The background half of the assessment API. ``POST /v1/assessments`` persists a
job and returns 202; this process drains the ``assessments`` Redis queue,
running each claim through the graph to the human checkpoint.

    make migrate            # the application schema, incl. `assessment_job`
    make setup-checkpointer # the checkpointer's own tables
    make serve              # the API (one shell)
    make worker             # the workers (another shell)

Concurrency is ``ASSESSMENT_WORKER_CONCURRENCY`` (default 2). On a
memory-constrained box keep it at 1 -- each worker loads the embedder and the
cross-encoder. See ``docs/ASYNC_PROCESSING.md``.
"""

from infrastructure.queue.worker import run_worker

if __name__ == "__main__":
    run_worker()
