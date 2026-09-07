"""The assessment API over the full stack -- [M5-04].

Real Postgres (the assessment / decision / audit / job tables and the LangGraph
Postgres checkpointer), the real use cases, the real orchestrator adapter and
its capturing sink -- only the LLM and the retriever are faked
(``tests/integration/_checkpoint_fakes.build_fake_context``: a canned-output
model and a one-clause stub retriever). Proves the M5-04 DoD: submit -> 202 + id,
read state/recommendation/citations, submit a decision and observe the resumed
run, read the audit trail.

The [M5-05] queue is stubbed with an *inline* queue that runs ``RunAssessment``
synchronously on ``enqueue`` -- these tests are about the decision / resume /
audit-fold path, not the queue. ``tests/integration/test_assessment_queue.py``
exercises the real Redis round-trip.
"""

from __future__ import annotations

import itertools
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from application.use_cases.run_assessment import RunAssessment
from infrastructure.clock import SystemClock
from infrastructure.database import (
    create_engine_from_database_url,
    create_session_factory,
    sqlalchemy_unit_of_work_factory,
)
from infrastructure.database.chunk_repository import upsert_chunks
from infrastructure.graph.checkpointer import open_claim_checkpointer
from infrastructure.graph.orchestrator import LangGraphClaimAssessmentOrchestrator
from infrastructure.observability.readiness import ReadinessProbe
from infrastructure.rag.chunk_schema import SCHEMA_VERSION, ChunkRecord
from presentation.app import create_app
from presentation.dependencies import AppComponents
from tests.integration._checkpoint_fakes import CLAUSE_ID, build_fake_context

pytestmark = pytest.mark.integration


def _seed_stub_clause(session: Session) -> None:
    """Persist a chunk for the stub retriever's clause so an ``edit`` can cite it."""
    document_id, _, _ = CLAUSE_ID.partition(":")
    upsert_chunks(
        session,
        [
            ChunkRecord(
                schema_version=SCHEMA_VERSION,
                chunk_id=CLAUSE_ID,
                document_id=document_id,
                clause_id=CLAUSE_ID,
                source_clause_ids=[CLAUSE_ID],
                chunk_index=0,
                chunk_count=1,
                parent_path="1. COBERTURAS",
                text="1.1 A seguradora cobre colisao.",
                display_text="1.1 A seguradora cobre colisao.",
                char_count=31,
                rule="single",
                clause_type="coverage",
                type_source="rule",
                confidence=None,
                bundle_section=None,
                source="text",
                susep_process="15414.900000/2013-00",
                insurer="Seguradora Teste",
                cnpj="00.000.000/0001-00",
                product_line="CASCO",
                indemnity_regime="valor_de_mercado",
                filing_year="2013",
            )
        ],
    )
    session.commit()


class _InlineQueue:
    """Runs the assessment synchronously on ``enqueue`` (no worker, no Redis)."""

    def __init__(self, run: RunAssessment) -> None:
        self._run = run

    def enqueue(self, assessment_id: str) -> None:
        self._run(assessment_id)


def _test_lifespan(database_url: str, session_factory: sessionmaker[Session]) -> object:
    ids = (f"it-{n}" for n in itertools.count(1))
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


@dataclass
class Api:
    """The test client plus the session factory the app runs on."""

    client: TestClient
    session_factory: sessionmaker[Session]


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
        lifespan=_test_lifespan(postgres_database_url, session_factory)  # type: ignore[arg-type]
    )
    with TestClient(app) as test_client:
        yield Api(client=test_client, session_factory=session_factory)
    engine.dispose()


def _submit(client: TestClient, **body: object) -> str:
    response = client.post(
        "/v1/assessments", json={"raw_text": "Bati o carro.", **body}
    )
    assert response.status_code == 202, response.text
    assessment_id = str(response.json()["assessment_id"])
    assert response.headers["location"] == f"/v1/assessments/{assessment_id}"
    return assessment_id


def test_full_flow_approve(api: Api) -> None:
    client = api.client
    assessment_id = _submit(client)

    awaiting = client.get(f"/v1/assessments/{assessment_id}").json()
    assert awaiting["status"] == "awaiting_review"
    assert awaiting["verdict"] == "compatible"
    assert awaiting["citations"], "the recommendation should carry a clause"
    assert awaiting["reasoning"] and awaiting["recommended_action"]

    assert client.get(f"/v1/assessments/{assessment_id}/audit").json()["entries"] == []

    decided = client.post(
        f"/v1/assessments/{assessment_id}/decision",
        json={"decision": "approve", "notes": "conferido"},
    )
    assert decided.status_code == 200, decided.text
    body = decided.json()
    assert body["status"] == "decided"
    assert body["decision"]["decision"] == "approve"
    assert body["decision"]["notes"] == "conferido"

    reread = client.get(f"/v1/assessments/{assessment_id}").json()
    assert reread["status"] == "decided"
    assert reread["verdict"] == awaiting["verdict"]
    assert reread["citations"] == awaiting["citations"]
    assert reread["reasoning"] == awaiting["reasoning"]

    trail = client.get(f"/v1/assessments/{assessment_id}/audit").json()["entries"]
    assert trail, "the resumed run wrote its durable trail"
    assert [e["sequence"] for e in trail] == list(range(len(trail)))
    last = trail[-1]
    assert last["node"] == "human_review"
    assert last["action"] == "human_decision:approve"
    assert last["payload"]["decision"] == "approve"

    # a second decision on a settled assessment is a conflict
    again = client.post(
        f"/v1/assessments/{assessment_id}/decision", json={"decision": "approve"}
    )
    assert again.status_code == 409
    assert again.json()["error"]["code"] == "assessment_already_decided"


def test_full_flow_reject(api: Api) -> None:
    client = api.client
    assessment_id = _submit(client)

    body = client.post(
        f"/v1/assessments/{assessment_id}/decision", json={"decision": "reject"}
    ).json()

    assert body["status"] == "decided"
    trail = client.get(f"/v1/assessments/{assessment_id}/audit").json()["entries"]
    assert trail[-1]["action"] == "human_decision:reject"


def test_decision_edit_records_the_edited_assessment(api: Api) -> None:
    client = api.client
    with api.session_factory() as session:
        _seed_stub_clause(session)

    assessment_id = _submit(client)

    body = client.post(
        f"/v1/assessments/{assessment_id}/decision",
        json={
            "decision": "edit",
            "edited": {
                "verdict": "incompatible",
                "reasoning": "Exclusao aplicavel.",
                "recommended_action": "Negar o sinistro.",
                "confidence": 0.55,
                "citations": [
                    {
                        "clause_id": CLAUSE_ID,
                        "document_id": CLAUSE_ID.split(":")[0],
                        "susep_process": "15414.900000/2013-00",
                        "clause_type": "coverage",
                        "excerpt": "A seguradora cobre colisao.",
                    }
                ],
            },
        },
    )
    assert body.status_code == 200, body.text
    edited = body.json()["decision"]["edited_assessment"]
    assert edited["verdict"] == "incompatible"
    assert body.json()["verdict"] == "compatible"  # system opinion untouched


def test_edit_citing_an_unknown_clause_is_422(api: Api) -> None:
    client = api.client
    assessment_id = _submit(client)

    response = client.post(
        f"/v1/assessments/{assessment_id}/decision",
        json={
            "decision": "edit",
            "edited": {
                "verdict": "incompatible",
                "reasoning": "x",
                "recommended_action": "y",
                "confidence": 0.5,
                "citations": [
                    {
                        "clause_id": "ghost:9",
                        "document_id": "ghost",
                        "susep_process": "15414.900000/2013-00",
                        "clause_type": "exclusion",
                        "excerpt": "t",
                    }
                ],
            },
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unknown_clause"


def test_get_unknown_assessment_is_404(api: Api) -> None:
    response = api.client.get("/v1/assessments/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "assessment_not_found"


def test_the_audit_trail_and_the_record_are_written_atomically(api: Api) -> None:
    assessment_id = _submit(api.client)
    api.client.post(
        f"/v1/assessments/{assessment_id}/decision", json={"decision": "approve"}
    )

    with api.session_factory() as session:
        rows = session.execute(
            text("SELECT count(*) FROM audit_event WHERE thread_id = :tid"),
            {"tid": assessment_id},
        ).scalar_one()
    assert rows > 0  # the fold committed the trail alongside the DECIDED record
