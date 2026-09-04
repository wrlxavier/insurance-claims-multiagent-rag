"""The JSON log formatter and ``configure_logging`` -- [M5-06]."""

from __future__ import annotations

import json
import logging
import sys

import pytest

from infrastructure.config.settings import ObservabilitySettings
from infrastructure.observability.correlation import bind_correlation_id
from infrastructure.observability.logging import (
    CorrelationIdFilter,
    JsonFormatter,
    configure_logging,
)


def _record(**extra: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


@pytest.mark.unit
def test_formatter_emits_the_dod_envelope() -> None:
    record = _record(correlation_id="c-1")
    payload = json.loads(JsonFormatter().format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "test.logger"
    assert payload["message"] == "hello world"
    assert payload["correlation_id"] == "c-1"
    assert payload["timestamp"].endswith("+00:00")


@pytest.mark.unit
def test_formatter_passes_extra_fields_through() -> None:
    record = _record(
        correlation_id="c-1", path="/ready", method="GET", status=503, duration_ms=1.2
    )
    payload = json.loads(JsonFormatter().format(record))

    assert payload["path"] == "/ready"
    assert payload["method"] == "GET"
    assert payload["status"] == 503
    assert payload["duration_ms"] == 1.2


@pytest.mark.unit
def test_formatter_renders_an_exception() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        exc_info = sys.exc_info()
    record = logging.LogRecord(
        "l", logging.ERROR, __file__, 1, "failed", None, exc_info
    )
    payload = json.loads(JsonFormatter().format(record))

    assert "ValueError: boom" in payload["exception"]


@pytest.mark.unit
def test_filter_fills_correlation_id_from_context() -> None:
    record = _record()
    with bind_correlation_id("bound"):
        CorrelationIdFilter().filter(record)
    assert getattr(record, "correlation_id", None) == "bound"


@pytest.mark.unit
def test_filter_defaults_to_dash_when_unbound() -> None:
    record = _record()
    CorrelationIdFilter().filter(record)
    assert getattr(record, "correlation_id", None) == "-"


@pytest.mark.unit
def test_configure_logging_is_idempotent() -> None:
    root = logging.getLogger()
    original = list(root.handlers)
    try:
        configure_logging(ObservabilitySettings(LOG_FORMAT="json", LOG_LEVEL="WARNING"))
        configure_logging(ObservabilitySettings(LOG_FORMAT="json", LOG_LEVEL="WARNING"))
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JsonFormatter)
        assert root.level == logging.WARNING
    finally:
        root.handlers = original


@pytest.mark.unit
def test_configure_logging_text_format() -> None:
    root = logging.getLogger()
    original = list(root.handlers)
    try:
        configure_logging(ObservabilitySettings(LOG_FORMAT="text"))
        assert not isinstance(root.handlers[0].formatter, JsonFormatter)
    finally:
        root.handlers = original
