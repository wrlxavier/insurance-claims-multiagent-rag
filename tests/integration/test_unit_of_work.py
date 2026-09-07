"""The SQLAlchemy unit of work against a real Postgres -- [M5-03].

The port's contract, proven transactionally: ``commit()`` makes writes durable;
leaving the block without one -- normally or by exception -- rolls everything
back; each factory call is its own transaction.
"""

import pytest
from sqlalchemy.orm import Session, sessionmaker

from infrastructure.database.assessment_repository import SqlAlchemyAssessmentRepository
from infrastructure.database.unit_of_work import (
    SqlAlchemyUnitOfWork,
    sqlalchemy_unit_of_work_factory,
)
from tests.unit.application.fakes import make_record

pytestmark = pytest.mark.integration


def test_commit_makes_the_write_durable(
    session_factory: sessionmaker[Session],
) -> None:
    factory = sqlalchemy_unit_of_work_factory(session_factory)

    with factory() as uow:
        uow.assessments.add(make_record(assessment_id="a-commit"))
        uow.commit()

    with factory() as uow:
        assert uow.assessments.get("a-commit") is not None


def test_leaving_without_commit_rolls_back(
    session_factory: sessionmaker[Session],
) -> None:
    factory = sqlalchemy_unit_of_work_factory(session_factory)

    with factory() as uow:
        uow.assessments.add(make_record(assessment_id="a-norollback"))
        # no commit

    with factory() as uow:
        assert uow.assessments.get("a-norollback") is None


def test_an_exception_rolls_back_and_propagates(
    session_factory: sessionmaker[Session],
) -> None:
    factory = sqlalchemy_unit_of_work_factory(session_factory)

    with pytest.raises(RuntimeError, match="boom"):
        with factory() as uow:
            uow.assessments.add(make_record(assessment_id="a-boom"))
            raise RuntimeError("boom")

    with factory() as uow:
        assert uow.assessments.get("a-boom") is None


def test_the_repository_is_bound_to_the_units_session(
    session_factory: sessionmaker[Session],
) -> None:
    with SqlAlchemyUnitOfWork(session_factory) as uow:
        assert isinstance(uow.assessments, SqlAlchemyAssessmentRepository)
        # A write is visible to a read on the same unit before commit...
        uow.assessments.add(make_record(assessment_id="a-samesession"))
        assert uow.assessments.get("a-samesession") is not None
        uow.commit()
