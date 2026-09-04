"""One JSON log line per event, to stdout -- [M5-06].

The M5-06 DoD: "JSON logs to stdout with timestamp, level, path, method, status,
duration". ``path`` / ``method`` / ``status`` / ``duration`` are per-request and
come from the middleware via ``extra=``; the formatter here owns the envelope
(``timestamp`` / ``level`` / ``logger`` / ``message`` / ``correlation_id``) and
passes any ``extra=`` keys straight through.

Hand-rolled rather than a ``python-json-logger`` dependency -- the same
minimal-dependency habit as the rest of the project, and the envelope is a dozen
lines. ``configure_logging`` is idempotent: the API entry point, the worker entry
point and every ``create_app()`` in a test all call it, and the last one wins
without stacking handlers.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from infrastructure.config.settings import ObservabilitySettings
from infrastructure.observability.correlation import get_correlation_id

# Attributes the stdlib puts on every LogRecord. Anything on a record that is
# *not* in here arrived via ``logger.info(..., extra={...})`` and is a field the
# caller wants in the line.
_STANDARD_RECORD_ATTRS = frozenset(logging.makeLogRecord({}).__dict__.keys()) | {
    "message",
    "asctime",
    "taskName",
}

_TEXT_FORMAT = "%(asctime)s %(levelname)s [%(correlation_id)s] %(name)s: %(message)s"


class CorrelationIdFilter(logging.Filter):
    """Stamp every record with the correlation id bound to the current context."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Always keep the record; add ``correlation_id`` if it is not set."""
        if not hasattr(record, "correlation_id"):
            record.correlation_id = get_correlation_id() or "-"
        return True


class JsonFormatter(logging.Formatter):
    """Render a record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        """The envelope, then every ``extra=`` field, then the exception if any."""
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", "-"),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and key != "correlation_id":
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, default=str)


def configure_logging(settings: ObservabilitySettings | None = None) -> None:
    """Point the root logger at stdout with the JSON (or text) formatter.

    Idempotent -- replaces the root handler set rather than adding to it.

    Uvicorn installs its own handlers on ``uvicorn`` / ``uvicorn.access`` /
    ``uvicorn.error`` with ``propagate=False`` and a plain formatter. We hand
    those loggers back to the root handler so every line is JSON;
    ``uvicorn.access`` is then silenced entirely because the per-request line is
    [presentation.middleware.RequestContextMiddleware]'s to emit, structured.
    """
    resolved = settings or ObservabilitySettings()

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(CorrelationIdFilter())
    if resolved.log_format == "text":
        handler.setFormatter(logging.Formatter(_TEXT_FORMAT))
    else:
        handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(resolved.log_level)

    for name in ("uvicorn", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True

    access = logging.getLogger("uvicorn.access")
    access.handlers.clear()
    access.propagate = False
