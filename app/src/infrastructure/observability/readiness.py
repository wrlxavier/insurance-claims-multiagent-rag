"""The dependency probes behind ``GET /ready`` -- [M5-06].

The M5-06 DoD: "``GET /ready`` checks Postgres, Redis and the vector index;
returns 503 with per-check detail when degraded". Each probe is guarded and
returns a ``CheckResult`` rather than raising, so one dependency being down gives
a 503 that still names which one and why -- an operator reads the body, not just
the status.

Liveness (``GET /health``) stays dependency-free and lives in the route module;
this is only the readiness side.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from infrastructure.database.chunk_repository import assert_chunk_table_ready
from infrastructure.database.session import is_database_reachable


@dataclass(frozen=True)
class CheckResult:
    """The outcome of one readiness probe."""

    name: str
    ok: bool
    detail: str | None = None


class ReadinessProbe:
    """Run the Postgres / Redis / vector-index probes for ``GET /ready``."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        redis_ping: Callable[[], object],
    ) -> None:
        """Wire the probe to the shared session factory and the queue's ``ping``."""
        self._session_factory = session_factory
        self._redis_ping = redis_ping

    def check(self) -> list[CheckResult]:
        """Probe every dependency; order is stable for a readable response body."""
        return [self._postgres(), self._redis(), self._vector_index()]

    def _postgres(self) -> CheckResult:
        if is_database_reachable(self._session_factory):
            return CheckResult("postgres", ok=True)
        return CheckResult("postgres", ok=False, detail="SELECT 1 failed")

    def _redis(self) -> CheckResult:
        try:
            self._redis_ping()
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            return CheckResult("redis", ok=False, detail=f"{type(exc).__name__}: {exc}")
        return CheckResult("redis", ok=True)

    def _vector_index(self) -> CheckResult:
        """Ready == the ``chunk`` table is migrated and has embedded rows."""
        try:
            with self._session_factory() as session:
                assert_chunk_table_ready(session)
                embedded = session.execute(
                    text("SELECT count(*) FROM chunk WHERE embedding IS NOT NULL")
                ).scalar_one()
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            return CheckResult(
                "vector_index", ok=False, detail=f"{type(exc).__name__}: {exc}"
            )
        if embedded == 0:
            return CheckResult(
                "vector_index",
                ok=False,
                detail="no embedded chunks -- run make build-index",
            )
        return CheckResult(
            "vector_index", ok=True, detail=f"{embedded} embedded chunks"
        )
