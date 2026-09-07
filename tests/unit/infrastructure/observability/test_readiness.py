"""The ``ReadinessProbe`` dependency checks -- [M5-06]."""

from __future__ import annotations

from typing import Any

import pytest

from infrastructure.observability import readiness as readiness_module
from infrastructure.observability.readiness import ReadinessProbe


class _FakeSessionFactory:
    """Enough of a ``sessionmaker`` for the probe: a context manager per call."""

    def __init__(self, *, reachable: bool = True, embedded: int = 5) -> None:
        self.reachable = reachable
        self.embedded = embedded

    def __call__(self) -> _FakeSessionFactory:
        return self

    def __enter__(self) -> _FakeSessionFactory:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, *_: Any) -> Any:
        class _Result:
            def __init__(self, value: int) -> None:
                self._value = value

            def scalar_one(self) -> int:
                return self._value

        return _Result(self.embedded)


@pytest.fixture(autouse=True)
def _stub_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the two real DB helpers the probe calls with fakes."""
    monkeypatch.setattr(
        readiness_module,
        "is_database_reachable",
        lambda factory: getattr(factory, "reachable", True),
    )
    monkeypatch.setattr(
        readiness_module, "assert_chunk_table_ready", lambda session: None
    )


def _probe(**kwargs: Any) -> ReadinessProbe:
    defaults: dict[str, Any] = {
        "session_factory": _FakeSessionFactory(),
        "redis_ping": lambda: None,
    }
    defaults.update(kwargs)
    return ReadinessProbe(**defaults)


@pytest.mark.unit
def test_all_checks_pass() -> None:
    results = {r.name: r for r in _probe().check()}

    assert results.keys() == {"postgres", "redis", "vector_index"}
    assert all(r.ok for r in results.values())
    assert results["vector_index"].detail == "5 embedded chunks"


@pytest.mark.unit
def test_postgres_down_is_isolated() -> None:
    results = {
        r.name: r
        for r in _probe(session_factory=_FakeSessionFactory(reachable=False)).check()
    }

    assert results["postgres"].ok is False
    assert results["redis"].ok is True


@pytest.mark.unit
def test_redis_ping_failure_is_reported() -> None:
    def _boom() -> None:
        raise ConnectionError("no route to host")

    results = {r.name: r for r in _probe(redis_ping=_boom).check()}

    assert results["redis"].ok is False
    assert "ConnectionError" in (results["redis"].detail or "")


@pytest.mark.unit
def test_empty_chunk_table_is_not_ready() -> None:
    results = {
        r.name: r
        for r in _probe(session_factory=_FakeSessionFactory(embedded=0)).check()
    }

    assert results["vector_index"].ok is False
    assert "build-index" in (results["vector_index"].detail or "")
