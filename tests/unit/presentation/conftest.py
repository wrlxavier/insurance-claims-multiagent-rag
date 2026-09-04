"""A ``TestClient`` wired to the in-memory application fakes -- [M5-04, M5-05].

The FastAPI app is built with its production lifespan intact but never entered
(``TestClient`` runs lifespan only as a context manager), so the composition
root never touches Postgres, Redis, LLMs or the retriever. Every use-case
provider is overridden to build the interactor against
``tests/unit/application/fakes.py``. A test can reassign ``harness.orchestrator``
before a request to change the canned graph behaviour (the resume path), or
inspect ``harness.queue.enqueued`` to see what ``POST`` scheduled.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from application.assessment_job import AssessmentJob
from application.assessment_record import AssessmentRecord
from application.audit_trail_entry import AuditTrailEntry
from application.use_cases.get_assessment import GetAssessment
from application.use_cases.get_audit_trail import GetAuditTrail
from application.use_cases.list_assessments import ListAssessments
from application.use_cases.submit_claim import SubmitClaim
from application.use_cases.submit_human_decision import SubmitHumanDecision
from domain.policy_clause import PolicyClause
from infrastructure.observability.readiness import CheckResult
from presentation.app import create_app
from presentation.dependencies import (
    get_get_assessment,
    get_get_audit_trail,
    get_list_assessments,
    get_readiness_probe,
    get_submit_claim,
    get_submit_human_decision,
)
from tests.unit.application.fakes import (
    FakeAssessmentQueue,
    FakeClaimAssessmentOrchestrator,
    FixedClock,
    InMemoryAssessmentJobRepository,
    InMemoryAssessmentRepository,
    InMemoryAuditTrailReader,
    InMemoryClauseRepository,
    SequentialIds,
    make_uow_factory,
)


class FakeReadinessProbe:
    """A ``ReadinessProbe`` stand-in whose check results the test dictates."""

    def __init__(self) -> None:
        self.results: list[CheckResult] = [
            CheckResult("postgres", ok=True),
            CheckResult("redis", ok=True),
            CheckResult("vector_index", ok=True, detail="12 embedded chunks"),
        ]

    def check(self) -> list[CheckResult]:
        return list(self.results)


@dataclass
class Harness:
    """Everything a presentation test needs to drive and inspect the API."""

    client: TestClient
    store: dict[str, AssessmentRecord]
    audit_store: dict[str, list[AuditTrailEntry]]
    job_store: dict[str, AssessmentJob]
    queue: FakeAssessmentQueue
    clauses: dict[str, PolicyClause] = field(default_factory=dict)
    clock: FixedClock = field(default_factory=FixedClock)
    ids: SequentialIds = field(default_factory=lambda: SequentialIds("api"))
    orchestrator: FakeClaimAssessmentOrchestrator = field(
        default_factory=FakeClaimAssessmentOrchestrator
    )
    readiness: FakeReadinessProbe = field(default_factory=FakeReadinessProbe)


@pytest.fixture
def harness() -> Iterator[Harness]:
    app = create_app()
    store: dict[str, AssessmentRecord] = {}
    audit_store: dict[str, list[AuditTrailEntry]] = {}
    job_store: dict[str, AssessmentJob] = {}
    h = Harness(
        client=TestClient(app, raise_server_exceptions=False),
        store=store,
        audit_store=audit_store,
        job_store=job_store,
        queue=FakeAssessmentQueue(),
    )

    def _assessments() -> InMemoryAssessmentRepository:
        return InMemoryAssessmentRepository(store)

    def _jobs() -> InMemoryAssessmentJobRepository:
        return InMemoryAssessmentJobRepository(job_store)

    app.dependency_overrides[get_submit_claim] = lambda: SubmitClaim(
        clock=h.clock,
        queue=h.queue,
        uow_factory=make_uow_factory(store, audit_store, job_store),
        new_id=h.ids,
    )
    app.dependency_overrides[get_get_assessment] = lambda: GetAssessment(
        assessments=_assessments(), jobs=_jobs()
    )
    app.dependency_overrides[get_list_assessments] = lambda: ListAssessments(
        assessments=_assessments()
    )
    app.dependency_overrides[get_submit_human_decision] = lambda: SubmitHumanDecision(
        clock=h.clock,
        orchestrator=h.orchestrator,
        assessments=_assessments(),
        clauses=InMemoryClauseRepository(h.clauses.values()),
        uow_factory=make_uow_factory(store, audit_store, job_store),
    )
    app.dependency_overrides[get_get_audit_trail] = lambda: GetAuditTrail(
        assessments=_assessments(),
        jobs=_jobs(),
        audit=InMemoryAuditTrailReader(audit_store),
    )
    app.dependency_overrides[get_readiness_probe] = lambda: h.readiness

    yield h
    app.dependency_overrides.clear()
