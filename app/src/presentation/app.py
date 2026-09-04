"""The FastAPI app and its composition root -- [M5-04].

``create_app()`` mounts the routers and the error edge. ``_default_lifespan``
is the composition root: it calls ``infrastructure.bootstrap.build_core_components``
(the heavy singletons -- DB engine, chat models, retrieval stack, LangGraph
orchestrator -- shared with the [M5-05] worker), adds the request-scoped bits
(the clock, the id minter, the Redis-backed ``AssessmentQueue``), stashes them on
``app.state.components``, and disposes the engine on shutdown. ``build_core_components``
probes the checkpointer at startup so a missing schema is an actionable "run
``make setup-checkpointer``" message before the first request, not after.

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

from infrastructure.bootstrap import build_core_components
from infrastructure.clock import SystemClock
from infrastructure.config.settings import (
    get_observability_settings,
    get_queue_settings,
)
from infrastructure.observability.logging import configure_logging
from infrastructure.observability.readiness import ReadinessProbe
from infrastructure.queue import build_assessment_queue
from presentation.dependencies import AppComponents
from presentation.errors import register_exception_handlers
from presentation.middleware import RequestContextMiddleware
from presentation.routes import assessments, health

logger = logging.getLogger(__name__)

_LifespanFactory = Callable[[FastAPI], AbstractAsyncContextManager[None]]


@asynccontextmanager
async def _default_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the production composition root and tear it down on shutdown."""
    configure_logging(get_observability_settings())

    core = build_core_components()
    queue = build_assessment_queue(get_queue_settings())

    app.state.components = AppComponents(
        clock=SystemClock(),
        queue=queue,
        orchestrator=core.orchestrator,
        uow_factory=core.uow_factory,
        session_factory=core.session_factory,
        new_id=lambda: str(uuid4()),
        readiness=ReadinessProbe(
            session_factory=core.session_factory,
            redis_ping=queue.ping,
        ),
    )
    logger.info("assessment API composition root ready")
    try:
        yield
    finally:
        core.dispose()


def create_app(*, lifespan: _LifespanFactory | None = None) -> FastAPI:
    """Build the FastAPI app; ``lifespan`` overrides the production composition."""
    app = FastAPI(
        title="Insurance claim assessment API",
        version="1",
        lifespan=lifespan or _default_lifespan,
    )
    app.add_middleware(RequestContextMiddleware)
    app.include_router(health.router)
    app.include_router(assessments.router)
    register_exception_handlers(app)
    return app


app = create_app()
