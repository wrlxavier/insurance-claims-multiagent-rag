"""The correlation id for one assessment run -- [M5-06].

A correlation id ties an HTTP request to every log line it (and the graph run it
starts) produces. It is accepted from the caller when present and minted when
not, then carried in a module-level ``contextvars.ContextVar`` rather than passed
down every function signature -- the logging filter
([infrastructure.observability.logging.CorrelationIdFilter]) reads it, and so
does anything that needs to tag an outbound call.

Two entry points set it:

- the HTTP middleware ([presentation.middleware.RequestContextMiddleware]) for a
  request, and
- the queue task ([infrastructure.queue.tasks.run_assessment_job]) for a worker
  run, from the id the enqueue side stored on the RQ job.

The graph adapter ([infrastructure.graph.orchestrator]) also binds it for the
duration of a ``.invoke`` so a node that logs picks it up even on the
synchronous resume path.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

# Accepted request headers, in priority order. ``X-Correlation-ID`` is the name
# this service uses on the way out; ``X-Request-ID`` is accepted because it is
# what many proxies and clients already send.
CORRELATION_ID_HEADERS: tuple[str, ...] = ("x-correlation-id", "x-request-id")

# The header this service sets on responses and would send on downstream calls.
CORRELATION_ID_RESPONSE_HEADER = "X-Correlation-ID"

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> str | None:
    """The correlation id bound to the current context, or ``None``."""
    return _correlation_id.get()


def generate_correlation_id() -> str:
    """Mint a fresh correlation id (a bare uuid4 hex string)."""
    return uuid.uuid4().hex


@contextmanager
def bind_correlation_id(value: str) -> Iterator[str]:
    """Bind ``value`` as the correlation id for the duration of the block.

    Resets to the previous value on exit, so nested binds and worker threads
    that reuse a context do not leak an id into the next unit of work.
    """
    token = _correlation_id.set(value)
    try:
        yield value
    finally:
        _correlation_id.reset(token)
