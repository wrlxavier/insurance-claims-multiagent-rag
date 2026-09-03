"""The SQLAlchemy unit of work -- [M5-03].

Implements [application.ports.unit_of_work.UnitOfWork]: one database transaction,
exposing the single repository the assessment use cases write through
(``assessments``). The use cases take a ``UnitOfWorkFactory``
(``Callable[[], UnitOfWork]``), so each invocation opens its own session and its
own transaction -- the session-per-transaction shape the port docstring calls
for.

Contract (from the port): entering the context begins the unit; ``commit()``
makes its writes durable; leaving without a ``commit()`` -- normally or by
exception -- rolls everything back. An exception is never suppressed.
"""

from types import TracebackType

from sqlalchemy.orm import Session, sessionmaker

from application.ports.assessment_repository import AssessmentRepository
from application.ports.unit_of_work import UnitOfWork, UnitOfWorkFactory
from infrastructure.database.assessment_repository import SqlAlchemyAssessmentRepository


class SqlAlchemyUnitOfWork:
    """One transaction over the assessment store, backed by a SQLAlchemy session."""

    # Set on `__enter__`; typed as the port so the class structurally satisfies
    # the `UnitOfWork` protocol (matching `tests/unit/application/fakes.py`).
    assessments: AssessmentRepository

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """Bind the unit to a session factory; the session opens on ``__enter__``."""
        self._session_factory = session_factory
        self._session: Session | None = None
        self._committed = False

    def __enter__(self) -> "UnitOfWork":
        """Open the session and expose the assessment repository bound to it."""
        self._session = self._session_factory()
        self._committed = False
        self.assessments = SqlAlchemyAssessmentRepository(self._session)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Roll back unless ``commit()`` was called, then close. Never suppress."""
        try:
            if not self._committed:
                self.rollback()
        finally:
            assert self._session is not None
            self._session.close()
            self._session = None

    def commit(self) -> None:
        """Make every write in this unit durable."""
        assert self._session is not None, "unit of work is not active"
        self._session.commit()
        self._committed = True

    def rollback(self) -> None:
        """Discard every write in this unit."""
        assert self._session is not None, "unit of work is not active"
        self._session.rollback()


def sqlalchemy_unit_of_work_factory(
    session_factory: sessionmaker[Session],
) -> UnitOfWorkFactory:
    """Build the ``UnitOfWorkFactory`` the use cases take -- one unit per call."""

    def _open() -> UnitOfWork:
        return SqlAlchemyUnitOfWork(session_factory)

    return _open
