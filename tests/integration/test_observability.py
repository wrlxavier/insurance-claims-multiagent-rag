"""Correlation-id propagation from an HTTP request to a graph node log line -- [M5-06].

The M5-06 DoD's propagation test. Real Postgres + the LangGraph Postgres
checkpointer; only the LLM and retriever are faked
(``_checkpoint_fakes.build_fake_context``). The submit path here runs the graph
synchronously inside the request (the ``_InlineQueue``), so a correlation id set
by the middleware must reach the ``infrastructure.graph.node`` log lines the
``build._instrumented`` wrapper emits.
"""

from __future__ import annotations

import itertools
import json
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from application.use_cases.run_assessment import RunAssessment
from infrastructure.clock import SystemClock
from infrastructure.database import (
    create_engine_from_database_url,
    create_session_factory,
    sqlalchemy_unit_of_work_factory,
)
from infrastructure.graph.checkpointer import open_claim_checkpointer
from infrastructure.graph.orchestrator import LangGraphClaimAssessmentOrchestrator
from infrastructure.observability.logging import CorrelationIdFilter, JsonFormatter
from infrastructure.observability.readiness import ReadinessProbe
from presentation.app import create_app
from presentation.dependencies import AppComponents
from tests.integration._checkpoint_fakes import build_fake_context

pytestmark = pytest.mark.integration


class _InlineQueue:
    def __init__(self, run: RunAssessment) -> None:
        self._run = run

    def enqueue(self, assessment_id: str) -> None:
        self._run(assessment_id)


def _lifespan(database_url: str, session_factory: sessionmaker[Session]) -> object:
    ids = (f"obs-{n}" for n in itertools.count(1))
    orchestrator = LangGraphClaimAssessmentOrchestrator(
        context_factory=lambda _session: build_fake_context(),
        session_factory=session_factory,
        database_url=database_url,
    )
    uow_factory = sqlalchemy_unit_of_work_factory(session_factory)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.components = AppComponents(
            clock=SystemClock(),
            queue=_InlineQueue(
                RunAssessment(
                    clock=SystemClock(),
                    orchestrator=orchestrator,
                    uow_factory=uow_factory,
                    is_transient=lambda _exc: False,
                    max_attempts=1,
                )
            ),
            orchestrator=orchestrator,
            uow_factory=uow_factory,
            session_factory=session_factory,
            new_id=lambda: next(ids),
            readiness=ReadinessProbe(
                session_factory=session_factory, redis_ping=lambda: None
            ),
        )
        yield

    return lifespan


class _CapturingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[dict[str, Any]] = []
        self.addFilter(CorrelationIdFilter())
        self.setFormatter(JsonFormatter())

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(json.loads(self.format(record)))


@dataclass
class Api:
    client: TestClient
    node_logs: _CapturingHandler


@pytest.fixture
def api(
    postgres_database_url: str,
    migrated_database: None,
) -> Iterator[Api]:
    with open_claim_checkpointer(postgres_database_url, setup=True):
        pass
    engine = create_engine_from_database_url(postgres_database_url)
    session_factory = create_session_factory(engine=engine)
    app = create_app(
        lifespan=_lifespan(postgres_database_url, session_factory)  # type: ignore[arg-type]
    )

    handler = _CapturingHandler()
    node_logger = logging.getLogger("infrastructure.graph.node")
    node_logger.addHandler(handler)
    node_logger.setLevel(logging.INFO)
    # `migrated_database` runs Alembic, whose `env.py` calls `fileConfig(...)` with
    # the default `disable_existing_loggers=True` -- that flips this logger off in
    # the shared test process (it never happens in the real API/worker process,
    # where migrations are a separate command).
    node_logger.disabled = False
    try:
        with TestClient(app) as client:
            yield Api(client=client, node_logs=handler)
    finally:
        node_logger.removeHandler(handler)
    engine.dispose()


def test_correlation_id_reaches_a_node_log_line(api: Api) -> None:
    response = api.client.post(
        "/v1/assessments",
        json={"raw_text": "Bati o carro."},
        headers={"X-Correlation-ID": "m5-06-itest"},
    )
    assert response.status_code == 202, response.text
    assert response.headers["x-correlation-id"] == "m5-06-itest"

    node_lines = [
        line
        for line in api.node_logs.lines
        if line["message"] in {"node.start", "node.completed"}
    ]
    assert node_lines, "the graph run should have emitted node log lines"
    assert all(line["correlation_id"] == "m5-06-itest" for line in node_lines)
    assert {"intake", "retrieval", "recommendation"} <= {
        line["node"] for line in node_lines
    }


def test_generated_correlation_id_is_consistent_across_the_run(api: Api) -> None:
    response = api.client.post("/v1/assessments", json={"raw_text": "Bati o carro."})

    generated = response.headers["x-correlation-id"]
    assert generated
    node_ids = {
        line["correlation_id"]
        for line in api.node_logs.lines
        if line["message"].startswith("node.")
    }
    assert node_ids == {generated}
