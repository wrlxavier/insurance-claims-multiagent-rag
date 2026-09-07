"""The in-memory fakes structurally satisfy the M5-02 ports.

The real assertion is mypy ``--strict`` over this file (the typed bindings
below); the runtime test only guards against a fake losing a method.
"""

import pytest

from application.ports.assessment_repository import AssessmentRepository
from application.ports.claim_assessment_orchestrator import ClaimAssessmentOrchestrator
from application.ports.clause_repository import ClauseRepository
from application.ports.clock import Clock
from application.ports.unit_of_work import UnitOfWork, UnitOfWorkFactory
from tests.unit.application.fakes import (
    FakeClaimAssessmentOrchestrator,
    FixedClock,
    InMemoryAssessmentRepository,
    InMemoryClauseRepository,
    InMemoryUnitOfWork,
    NaiveClock,
    make_uow_factory,
)

_clock: Clock = FixedClock()
_naive_clock: Clock = NaiveClock()
_clauses: ClauseRepository = InMemoryClauseRepository()
_assessments: AssessmentRepository = InMemoryAssessmentRepository({})
_uow: UnitOfWork = InMemoryUnitOfWork({})
_uow_factory: UnitOfWorkFactory = make_uow_factory({})
_orchestrator: ClaimAssessmentOrchestrator = FakeClaimAssessmentOrchestrator()


@pytest.mark.unit
def test_fakes_expose_the_port_methods() -> None:
    assert callable(_clock.now)
    assert callable(_clauses.get_many)
    assert callable(_assessments.list)
    assert callable(_orchestrator.start)
    assert callable(_orchestrator.resume)
    with _uow_factory() as uow:
        assert isinstance(uow.assessments, InMemoryAssessmentRepository)
        uow.commit()
