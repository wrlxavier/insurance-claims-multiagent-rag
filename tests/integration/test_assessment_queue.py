"""A queued assessment from submission to completion -- the [M5-05] DoD.

Real Postgres (assessment / job / audit tables + the LangGraph checkpointer) and
real Redis (an RQ queue drained by an in-process burst ``SimpleWorker``). Only
the LLM and the retriever are faked, through ``job_path`` pointing at
``tests/integration/_queue_fakes``.

Three paths:

- submission -> 202 ``pending`` -> worker -> ``awaiting_review`` -> a decision;
- a transient provider failure -> retry with backoff -> success;
- a real failure -> the job dead-letters, cause preserved, RQ ``FailedJobRegistry``.
"""

from __future__ import annotations

import itertools
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pytest
import redis
from fastapi import FastAPI
from fastapi.testclient import TestClient
from rq import Queue, SimpleWorker
from rq.registry import FailedJobRegistry
from sqlalchemy.orm import Session, sessionmaker

from infrastructure.clock import SystemClock
from infrastructure.database import (
    create_engine_from_database_url,
    create_session_factory,
    sqlalchemy_unit_of_work_factory,
)
from infrastructure.graph.checkpointer import open_claim_checkpointer
from infrastructure.graph.orchestrator import LangGraphClaimAssessmentOrchestrator
from infrastructure.queue.rq_queue import QUEUE_NAME, RqAssessmentQueue
from infrastructure.queue.worker import stop_retry_on_permanent
from presentation.app import create_app
from presentation.dependencies import AppComponents
from tests.integration import _queue_fakes
from tests.integration._checkpoint_fakes import build_fake_context

pytestmark = pytest.mark.integration


@dataclass
class Api:
    client: TestClient
    connection: redis.Redis


def _lifespan(
    database_url: str,
    session_factory: sessionmaker[Session],
    connection: redis.Redis,
    job_path: str,
) -> object:
    ids = (f"q-{n}" for n in itertools.count(1))
    orchestrator = LangGraphClaimAssessmentOrchestrator(
        context_factory=lambda _session: build_fake_context(),
        session_factory=session_factory,
        database_url=database_url,
    )
    queue = RqAssessmentQueue(
        Queue(QUEUE_NAME, connection=connection),
        job_path=job_path,
        max_attempts=3,
        retry_intervals=[0, 0],
        job_timeout=60,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.components = AppComponents(
            clock=SystemClock(),
            queue=queue,
            orchestrator=orchestrator,
            uow_factory=sqlalchemy_unit_of_work_factory(session_factory),
            session_factory=session_factory,
            new_id=lambda: next(ids),
        )
        yield

    return lifespan


def _api(
    *,
    database_url: str,
    connection: redis.Redis,
    job_path: str = "tests.integration._queue_fakes.run_assessment_job",
) -> Iterator[Api]:
    with open_claim_checkpointer(database_url, setup=True):
        pass
    engine = create_engine_from_database_url(database_url)
    session_factory = create_session_factory(engine=engine)
    app = create_app(
        lifespan=_lifespan(database_url, session_factory, connection, job_path)  # type: ignore[arg-type]
    )
    with TestClient(app) as client:
        yield Api(client=client, connection=connection)
    engine.dispose()


@pytest.fixture
def api(
    postgres_database_url: str,
    migrated_database: None,
    redis_connection: redis.Redis,
) -> Iterator[Api]:
    yield from _api(database_url=postgres_database_url, connection=redis_connection)


def _drain(connection: redis.Redis, *, rounds: int = 6) -> None:
    """Run burst workers until the queue is exhausted.

    Retries use ``interval=0``, so a failed job is re-enqueued immediately rather
    than scheduled -- a plain loop of burst passes drains them.
    """
    queue = Queue(QUEUE_NAME, connection=connection)
    for _ in range(rounds):
        if not queue.count:
            break
        SimpleWorker(
            [QUEUE_NAME],
            connection=connection,
            exception_handlers=[stop_retry_on_permanent],
        ).work(burst=True)


def _submit(client: TestClient, **body: object) -> str:
    response = client.post(
        "/v1/assessments", json={"raw_text": "Bati o carro.", **body}
    )
    assert response.status_code == 202, response.text
    assert response.json()["status"] == "pending"
    return str(response.json()["assessment_id"])


def test_submission_to_completion(api: Api) -> None:
    assessment_id = _submit(api.client)

    pending = api.client.get(f"/v1/assessments/{assessment_id}").json()
    assert pending["status"] in {"pending", "running"}
    assert pending["verdict"] is None

    _drain(api.connection)

    done = api.client.get(f"/v1/assessments/{assessment_id}").json()
    assert done["status"] == "awaiting_review"
    assert done["verdict"] == "compatible"
    assert done["citations"], "the recommendation should carry a clause"

    decided = api.client.post(
        f"/v1/assessments/{assessment_id}/decision", json={"decision": "approve"}
    )
    assert decided.status_code == 200, decided.text
    assert decided.json()["status"] == "decided"

    trail = api.client.get(f"/v1/assessments/{assessment_id}/audit").json()["entries"]
    assert trail[-1]["action"] == "human_decision:approve"


def test_a_transient_failure_is_retried_then_succeeds(
    postgres_database_url: str,
    migrated_database: None,
    redis_connection: redis.Redis,
) -> None:
    _queue_fakes.reset_flaky_state()
    for api in _api(
        database_url=postgres_database_url,
        connection=redis_connection,
        job_path="tests.integration._queue_fakes.run_flaky_assessment_job",
    ):
        assessment_id = _submit(api.client)
        _drain(redis_connection)

        # The first attempt hit a 429; the retry succeeded.
        assert _queue_fakes._flaky_state["transient_failures"] == 1

        recovered = api.client.get(f"/v1/assessments/{assessment_id}").json()
        assert recovered["status"] == "awaiting_review"
        assert recovered["verdict"] == "compatible"

        # It recovered -- it did NOT dead-letter.
        registry = FailedJobRegistry(QUEUE_NAME, connection=redis_connection)
        assert assessment_id not in registry.get_job_ids()


def test_a_real_failure_dead_letters_with_the_cause(
    postgres_database_url: str,
    migrated_database: None,
    redis_connection: redis.Redis,
) -> None:
    for api in _api(
        database_url=postgres_database_url,
        connection=redis_connection,
        job_path="tests.integration._queue_fakes.run_failing_assessment_job",
    ):
        assessment_id = _submit(api.client)
        _drain(redis_connection)

        failed = api.client.get(f"/v1/assessments/{assessment_id}").json()
        assert failed["status"] == "failed"
        assert "could not be parsed" in failed["error"]

        registry = FailedJobRegistry(QUEUE_NAME, connection=redis_connection)
        assert assessment_id in registry.get_job_ids()
