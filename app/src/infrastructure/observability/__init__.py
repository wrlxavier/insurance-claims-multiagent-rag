"""Operational observability: structured logging, correlation IDs, readiness -- [M5-06].

The three pieces of the M5-06 baseline that are not HTTP routes:

- ``correlation`` -- the per-run id, held in a ``contextvars`` variable so it
  rides along every log line without being threaded through call signatures.
- ``logging`` -- one JSON line per event to stdout (timestamp, level, logger,
  message, correlation id, plus whatever ``extra=`` carries).
- ``readiness`` -- the dependency probes ``GET /ready`` reports on.

Import from the submodules directly (``infrastructure.observability.correlation``
etc.): ``readiness`` reaches into ``infrastructure.database``, and the queue /
graph modules that only need a correlation id should not pull that in. Nothing
here imports FastAPI or RQ -- the middleware and the queue adapter call in, so
``domain`` / ``application`` stay clean.
"""
