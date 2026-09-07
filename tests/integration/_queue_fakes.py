"""RQ-importable job functions for the [M5-05] queue integration test.

The real job path (``infrastructure.queue.tasks.run_assessment_job``) builds the
whole retrieval stack. These stand-ins wire ``RunAssessment`` to the same fake
graph context ``tests/integration/_checkpoint_fakes`` uses everywhere else -- real
Postgres, real checkpointer, canned model + one-clause retriever -- so the test
exercises the queue round-trip without a GPU.

``RqAssessmentQueue`` is constructed with ``job_path`` pointing at one of these,
so nothing here needs a test seam inside the production queue code.
"""

from __future__ import annotations

import httpx2
import openai
from sqlalchemy.orm import Session, sessionmaker

from application.orchestrator_result import OrchestratorResult
from application.use_cases.run_assessment import RunAssessment
from domain.claim import Claim
from domain.human_decision import HumanDecision
from infrastructure.clock import SystemClock
from infrastructure.config.settings import get_database_settings
from infrastructure.database import (
    create_engine_from_database_url,
    create_session_factory,
    sqlalchemy_unit_of_work_factory,
)
from infrastructure.graph.orchestrator import LangGraphClaimAssessmentOrchestrator
from infrastructure.llm_errors import is_transient_llm_error
from infrastructure.observability.correlation import (
    bind_correlation_id,
    generate_correlation_id,
)
from tests.integration._checkpoint_fakes import build_fake_context

_MAX_ATTEMPTS = 3
# State survives across job executions in the same process (the burst
# SimpleWorker the test drives is in-process).
_flaky_state = {"transient_failures": 0}


def reset_flaky_state() -> None:
    """Call at the start of a test that uses ``run_flaky_assessment_job``."""
    _flaky_state["transient_failures"] = 0


def _database_url() -> str:
    return get_database_settings().sqlalchemy_database_url


def _session_factory() -> sessionmaker[Session]:
    return create_session_factory(
        engine=create_engine_from_database_url(_database_url())
    )


def _fake_orchestrator(
    session_factory: sessionmaker[Session],
) -> LangGraphClaimAssessmentOrchestrator:
    return LangGraphClaimAssessmentOrchestrator(
        context_factory=lambda _session: build_fake_context(),
        session_factory=session_factory,
        database_url=_database_url(),
    )


def _run(
    orchestrator: object, assessment_id: str, correlation_id: str | None = None
) -> None:
    session_factory = _session_factory()
    with bind_correlation_id(correlation_id or generate_correlation_id()):
        RunAssessment(
            clock=SystemClock(),
            orchestrator=orchestrator,  # type: ignore[arg-type]
            uow_factory=sqlalchemy_unit_of_work_factory(session_factory),
            is_transient=is_transient_llm_error,
            max_attempts=_MAX_ATTEMPTS,
        )(assessment_id)


def run_assessment_job(assessment_id: str, correlation_id: str | None = None) -> None:
    """Happy-path job: run the fake graph to the human checkpoint."""
    _run(_fake_orchestrator(_session_factory()), assessment_id, correlation_id)


class _WrappingOrchestrator:
    """Delegates ``resume`` to a real fake-context orchestrator; ``start`` varies."""

    def __init__(self, inner: LangGraphClaimAssessmentOrchestrator) -> None:
        self._inner = inner

    def start(self, *, assessment_id: str, claim: Claim) -> OrchestratorResult:
        raise NotImplementedError

    def resume(
        self, *, assessment_id: str, decision: HumanDecision
    ) -> OrchestratorResult:
        return self._inner.resume(assessment_id=assessment_id, decision=decision)


class _FlakyOrchestrator(_WrappingOrchestrator):
    def start(self, *, assessment_id: str, claim: Claim) -> OrchestratorResult:
        if _flaky_state["transient_failures"] == 0:
            _flaky_state["transient_failures"] += 1
            response = httpx2.Response(
                429, request=httpx2.Request("POST", "https://openrouter/api")
            )
            cause = openai.RateLimitError("429", response=response, body=None)
            raise RuntimeError("compatibility node failed") from cause
        return self._inner.start(assessment_id=assessment_id, claim=claim)


class _PermanentlyFailingOrchestrator(_WrappingOrchestrator):
    def start(self, *, assessment_id: str, claim: Claim) -> OrchestratorResult:
        raise ValueError("the claim narrative could not be parsed")


def run_flaky_assessment_job(
    assessment_id: str, correlation_id: str | None = None
) -> None:
    """Fails transiently once, then succeeds -- exercises retry-with-backoff."""
    _run(
        _FlakyOrchestrator(_fake_orchestrator(_session_factory())),
        assessment_id,
        correlation_id,
    )


def run_failing_assessment_job(
    assessment_id: str, correlation_id: str | None = None
) -> None:
    """Fails with a real error every time -- exercises the dead-letter path."""
    _run(
        _PermanentlyFailingOrchestrator(_fake_orchestrator(_session_factory())),
        assessment_id,
        correlation_id,
    )
