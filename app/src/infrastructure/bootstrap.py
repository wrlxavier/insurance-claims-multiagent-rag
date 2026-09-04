"""The shared composition root for the API and the worker -- [M5-05].

Before M5-05 the FastAPI ``lifespan`` was the only place the heavy process-wide
singletons were built (the DB engine, the two chat models, the retrieval stack,
the LangGraph orchestrator). The worker (``make worker``) needs the same set, so
that construction lives here and both entry points call it.

``build_core_components`` is deliberately framework-free: no FastAPI, no RQ. The
API wraps its result with the request-scoped bits (the clock, the id minter, the
queue); the worker wraps it with a ``RunAssessment``.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from application.ports.unit_of_work import UnitOfWorkFactory
from infrastructure.config.llm_client_factory import build_chat_model
from infrastructure.config.settings import (
    DatabaseSettings,
    LlmSettings,
    ObservabilitySettings,
    get_database_settings,
    get_llm_settings,
    get_observability_settings,
)
from infrastructure.database import (
    create_engine_from_settings,
    create_session_factory,
    sqlalchemy_unit_of_work_factory,
)
from infrastructure.graph.checkpointer import open_claim_checkpointer
from infrastructure.graph.context import (
    NO_CLASSIFIER,
    GraphContext,
    InjectionClassifierPort,
)
from infrastructure.graph.orchestrator import LangGraphClaimAssessmentOrchestrator
from infrastructure.observability.tracing import RunTracer, build_tracer
from infrastructure.rag.retriever_factory import (
    build_graph_retriever,
    load_retriever_components,
)


@dataclass
class CoreComponents:
    """The process-wide singletons the API and the worker both need."""

    engine: Engine
    session_factory: sessionmaker[Session]
    orchestrator: LangGraphClaimAssessmentOrchestrator
    uow_factory: UnitOfWorkFactory
    tracer: RunTracer

    def dispose(self) -> None:
        """Release the connection pool and stop the tracer -- call on shutdown."""
        self.engine.dispose()
        self.tracer.shutdown()


def build_core_components(
    *,
    database_settings: DatabaseSettings | None = None,
    llm_settings: LlmSettings | None = None,
    observability_settings: ObservabilitySettings | None = None,
    probe_checkpointer: bool = True,
) -> CoreComponents:
    """Build the DB engine, the chat models, the retriever and the orchestrator.

    ``probe_checkpointer`` opens the LangGraph Postgres checkpointer once so a
    missing schema fails fast with an actionable "run ``make setup-checkpointer``"
    message rather than on the first job.

    The tracer ([M5-07]) is process-wide like the models: one Langfuse client per
    API or worker process, or the no-op when tracing is not configured. It is
    built here rather than per run because the SDK batches and exports on a
    background thread -- one per assessment would be a thread per assessment.
    """
    db = database_settings or get_database_settings()
    llm = llm_settings or get_llm_settings()
    observability = observability_settings or get_observability_settings()
    tracer = build_tracer(observability=observability, llm=llm)

    engine = create_engine_from_settings(db)
    session_factory = create_session_factory(engine=engine)

    if probe_checkpointer:
        with open_claim_checkpointer(db.sqlalchemy_database_url):
            pass

    fast_model = build_chat_model(
        llm,
        llm.llm_model_fast,
        provider_order=llm.llm_fast_provider_order,
        allow_fallbacks=llm.llm_fast_allow_fallbacks,
    )
    reasoning_model = build_chat_model(
        llm,
        llm.llm_model_reasoning,
        provider_order=llm.llm_reasoning_provider_order,
        allow_fallbacks=llm.llm_reasoning_allow_fallbacks,
    )
    retriever_components = load_retriever_components()
    classifier = _load_classifier(llm)

    def context_factory(session: Session) -> GraphContext:
        return GraphContext(
            fast_model=fast_model,
            reasoning_model=reasoning_model,
            retriever=build_graph_retriever(
                session, retriever_components, tracer=tracer
            ),
            llm_settings=llm,
            audit_sink=None,
            tracer=tracer,
            classifier=classifier,
        )

    orchestrator = LangGraphClaimAssessmentOrchestrator(
        context_factory=context_factory,
        session_factory=session_factory,
        database_url=db.sqlalchemy_database_url,
        tracer=tracer,
    )

    return CoreComponents(
        engine=engine,
        session_factory=session_factory,
        orchestrator=orchestrator,
        uow_factory=sqlalchemy_unit_of_work_factory(session_factory),
        tracer=tracer,
    )


def _load_classifier(llm: LlmSettings) -> InjectionClassifierPort:
    """The optional prompt-injection classifier -- [M5-08 Appendix].

    ``NO_CLASSIFIER`` (the no-op) unless
    ``PROMPT_INJECTION_CLASSIFIER_ENABLED=true``; deferred import so the
    optional ``embed`` group is only required where the classifier is
    actually turned on, mirroring ``infrastructure.rag.retriever_factory``'s
    ``_load_reranker``/``_load_query_embedder``.
    """
    if not llm.prompt_injection_classifier_enabled:
        return NO_CLASSIFIER

    from infrastructure.guardrails.local_prompt_injection_classifier import (
        LocalPromptInjectionClassifier,
    )

    return LocalPromptInjectionClassifier(
        model_id=llm.prompt_injection_classifier_model,
        threshold=llm.prompt_injection_classifier_threshold,
    )
