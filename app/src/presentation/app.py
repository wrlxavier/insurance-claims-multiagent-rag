"""The FastAPI app and its composition root -- [M5-04].

``create_app()`` mounts the routers and the error edge. ``_default_lifespan``
is the composition root: it builds the process-wide singletons (the DB engine
and session factory, the two chat models, the retrieval components, the
LangGraph orchestrator, the clock, the id minter), stashes them on
``app.state.components`` for the dependency layer, and disposes the engine on
shutdown. A checkpointer probe at startup turns a missing checkpointer schema
into an actionable "run ``make setup-checkpointer``" message before the first
request rather than after it.

``lifespan`` is injectable so a test can supply its own composition (real
Postgres, a fake-context orchestrator) without the heavy model / retriever load
-- see ``tests/integration/test_assessment_api.py``. Unit tests skip the
lifespan entirely and override the use-case providers
(``tests/unit/presentation/conftest.py``).

Run locally with ``make serve`` (``uvicorn presentation.app:app``).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI
from sqlalchemy.orm import Session

from infrastructure.clock import SystemClock
from infrastructure.config.llm_client_factory import build_chat_model
from infrastructure.config.settings import (
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
from infrastructure.graph.context import GraphContext
from infrastructure.graph.orchestrator import LangGraphClaimAssessmentOrchestrator
from infrastructure.rag.retriever_factory import (
    build_graph_retriever,
    load_retriever_components,
)
from presentation.dependencies import AppComponents
from presentation.errors import register_exception_handlers
from presentation.routes import assessments, health

logger = logging.getLogger(__name__)

_LifespanFactory = Callable[[FastAPI], AbstractAsyncContextManager[None]]


@asynccontextmanager
async def _default_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the production composition root and tear it down on shutdown."""
    obs = get_observability_settings()
    logging.basicConfig(level=obs.log_level)

    db = get_database_settings()
    engine = create_engine_from_settings(db)
    session_factory = create_session_factory(engine=engine)

    # Fail fast with the actionable message if the checkpointer schema is absent.
    with open_claim_checkpointer(db.sqlalchemy_database_url):
        pass

    llm = get_llm_settings()
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

    def context_factory(session: Session) -> GraphContext:
        return GraphContext(
            fast_model=fast_model,
            reasoning_model=reasoning_model,
            retriever=build_graph_retriever(session, retriever_components),
            llm_settings=llm,
            audit_sink=None,
        )

    app.state.components = AppComponents(
        clock=SystemClock(),
        orchestrator=LangGraphClaimAssessmentOrchestrator(
            context_factory=context_factory,
            session_factory=session_factory,
            database_url=db.sqlalchemy_database_url,
        ),
        uow_factory=sqlalchemy_unit_of_work_factory(session_factory),
        session_factory=session_factory,
        new_id=lambda: str(uuid4()),
    )
    logger.info("assessment API composition root ready")
    try:
        yield
    finally:
        engine.dispose()


def create_app(*, lifespan: _LifespanFactory | None = None) -> FastAPI:
    """Build the FastAPI app; ``lifespan`` overrides the production composition."""
    app = FastAPI(
        title="Insurance claim assessment API",
        version="1",
        lifespan=lifespan or _default_lifespan,
    )
    app.include_router(health.router)
    app.include_router(assessments.router)
    register_exception_handlers(app)
    return app


app = create_app()
