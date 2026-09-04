"""``GET /ready`` and the request-context middleware -- [M5-06].

Driven through the same in-memory ``Harness`` as the other endpoint tests
(``tests/unit/presentation/conftest.py``); the readiness probe is the
``FakeReadinessProbe`` whose results the test dictates.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

import pytest

from infrastructure.observability.logging import CorrelationIdFilter, JsonFormatter
from infrastructure.observability.readiness import CheckResult
from tests.unit.presentation.conftest import Harness


class _CapturingHandler(logging.Handler):
    """Collects the JSON string every root log line would be written as."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[dict[str, object]] = []
        self.addFilter(CorrelationIdFilter())
        self.setFormatter(JsonFormatter())

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(json.loads(self.format(record)))


@pytest.fixture
def captured_logs() -> Iterator[_CapturingHandler]:
    handler = _CapturingHandler()
    root = logging.getLogger()
    root.addHandler(handler)
    previous_level = root.level
    root.setLevel(logging.INFO)
    try:
        yield handler
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)


# --------------------------------------------------------------------------- #
# GET /ready
# --------------------------------------------------------------------------- #


def test_ready_returns_200_and_per_check_detail_when_healthy(harness: Harness) -> None:
    response = harness.client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert set(body["checks"]) == {"postgres", "redis", "vector_index"}
    assert body["checks"]["postgres"] == {"status": "ok"}
    assert body["checks"]["vector_index"]["detail"] == "12 embedded chunks"


def test_ready_returns_503_naming_the_degraded_dependency(harness: Harness) -> None:
    harness.readiness.results = [
        CheckResult("postgres", ok=True),
        CheckResult("redis", ok=False, detail="ConnectionError: refused"),
        CheckResult("vector_index", ok=True),
    ]

    response = harness.client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["redis"] == {
        "status": "error",
        "detail": "ConnectionError: refused",
    }
    assert body["checks"]["postgres"] == {"status": "ok"}


# --------------------------------------------------------------------------- #
# Request-context middleware
# --------------------------------------------------------------------------- #


def test_generates_a_correlation_id_when_the_request_has_none(harness: Harness) -> None:
    response = harness.client.get("/health")

    assert response.headers["x-correlation-id"]


def test_echoes_an_inbound_correlation_id(harness: Harness) -> None:
    response = harness.client.get("/health", headers={"X-Correlation-ID": "req-42"})

    assert response.headers["x-correlation-id"] == "req-42"


def test_accepts_x_request_id_as_the_correlation_id(harness: Harness) -> None:
    response = harness.client.get("/health", headers={"X-Request-ID": "trace-9"})

    assert response.headers["x-correlation-id"] == "trace-9"


def test_emits_one_structured_access_line_per_request(
    harness: Harness, captured_logs: _CapturingHandler
) -> None:
    harness.client.get("/health", headers={"X-Correlation-ID": "req-42"})

    access = [line for line in captured_logs.lines if line["message"] == "request"]
    assert len(access) == 1
    line = access[0]
    assert line["path"] == "/health"
    assert line["method"] == "GET"
    assert line["status"] == 200
    assert isinstance(line["duration_ms"], (int, float))
    assert line["correlation_id"] == "req-42"
    assert line["timestamp"] and line["level"] == "INFO"
