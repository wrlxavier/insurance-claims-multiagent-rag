"""FastAPI dependency wiring for the assessment use cases -- [M5-04].

The composition root (``presentation.app.lifespan``) builds the process-wide
singletons -- the clock, the LangGraph orchestrator, the unit-of-work factory,
the session factory, the id minter -- and stashes them in an ``AppComponents``
on ``app.state``. Everything here reads that and assembles a use case *per
request*: a fresh read session for the ``GET`` / list paths, the uow factory for
the writes.

Every provider is a plain function so a test can replace any one of them through
``app.dependency_overrides`` -- overriding ``get_orchestrator`` alone is enough
to run the whole API against ``tests/unit/application/fakes.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Annotated, cast

from fastapi import Depends, Request
from sqlalchemy.orm import Session, sessionmaker

from application.ports.assessment_queue import AssessmentQueue
from application.ports.claim_assessment_orchestrator import ClaimAssessmentOrchestrator
from application.ports.clock import Clock
from application.ports.unit_of_work import UnitOfWorkFactory
from application.use_cases.get_assessment import GetAssessment
from application.use_cases.get_audit_trail import GetAuditTrail
from application.use_cases.list_assessments import ListAssessments
from application.use_cases.submit_claim import SubmitClaim
from application.use_cases.submit_human_decision import SubmitHumanDecision
from infrastructure.database import (
    SqlAlchemyAssessmentJobRepository,
    SqlAlchemyAssessmentRepository,
    SqlAlchemyAuditTrailReader,
    SqlAlchemyClauseRepository,
)
from infrastructure.observability.readiness import ReadinessProbe


@dataclass
class AppComponents:
    """The process-wide singletons the composition root builds once."""

    clock: Clock
    queue: AssessmentQueue
    orchestrator: ClaimAssessmentOrchestrator
    uow_factory: UnitOfWorkFactory
    session_factory: sessionmaker[Session]
    new_id: Callable[[], str]
    readiness: ReadinessProbe


def _components(request: Request) -> AppComponents:
    return cast(AppComponents, request.app.state.components)


def get_read_session(request: Request) -> Iterator[Session]:
    """A per-request read session -- never committed, rolled back on the way out."""
    with _components(request).session_factory() as session:
        yield session
        session.rollback()


def get_clock(request: Request) -> Clock:
    """The wall clock the use cases stamp timestamps from."""
    return _components(request).clock


def get_orchestrator(request: Request) -> ClaimAssessmentOrchestrator:
    """The LangGraph-backed assessment orchestrator (resume path)."""
    return _components(request).orchestrator


def get_queue(request: Request) -> AssessmentQueue:
    """The Redis-backed queue ``SubmitClaim`` hands a new job to."""
    return _components(request).queue


def get_uow_factory(request: Request) -> UnitOfWorkFactory:
    """Opens one write transaction per use-case invocation."""
    return _components(request).uow_factory


def get_new_id(request: Request) -> Callable[[], str]:
    """Mints the claim / assessment identifiers."""
    return _components(request).new_id


def get_readiness_probe(request: Request) -> ReadinessProbe:
    """The dependency probe ``GET /ready`` reports on -- [M5-06]."""
    return _components(request).readiness


_Session = Annotated[Session, Depends(get_read_session)]
_Clock = Annotated[Clock, Depends(get_clock)]
_Orchestrator = Annotated[ClaimAssessmentOrchestrator, Depends(get_orchestrator)]
_Queue = Annotated[AssessmentQueue, Depends(get_queue)]
_UowFactory = Annotated[UnitOfWorkFactory, Depends(get_uow_factory)]
_NewId = Annotated[Callable[[], str], Depends(get_new_id)]


def get_submit_claim(
    clock: _Clock,
    queue: _Queue,
    uow_factory: _UowFactory,
    new_id: _NewId,
) -> SubmitClaim:
    """Wire ``SubmitClaim`` for one request."""
    return SubmitClaim(
        clock=clock,
        queue=queue,
        uow_factory=uow_factory,
        new_id=new_id,
    )


def get_get_assessment(session: _Session) -> GetAssessment:
    """Wire ``GetAssessment`` for one request."""
    return GetAssessment(
        assessments=SqlAlchemyAssessmentRepository(session),
        jobs=SqlAlchemyAssessmentJobRepository(session),
    )


def get_list_assessments(session: _Session) -> ListAssessments:
    """Wire ``ListAssessments`` for one request."""
    return ListAssessments(assessments=SqlAlchemyAssessmentRepository(session))


def get_submit_human_decision(
    clock: _Clock,
    orchestrator: _Orchestrator,
    uow_factory: _UowFactory,
    session: _Session,
) -> SubmitHumanDecision:
    """Wire ``SubmitHumanDecision`` for one request."""
    return SubmitHumanDecision(
        clock=clock,
        orchestrator=orchestrator,
        assessments=SqlAlchemyAssessmentRepository(session),
        clauses=SqlAlchemyClauseRepository(session),
        uow_factory=uow_factory,
    )


def get_get_audit_trail(session: _Session) -> GetAuditTrail:
    """Wire ``GetAuditTrail`` for one request."""
    return GetAuditTrail(
        assessments=SqlAlchemyAssessmentRepository(session),
        jobs=SqlAlchemyAssessmentJobRepository(session),
        audit=SqlAlchemyAuditTrailReader(session),
    )
