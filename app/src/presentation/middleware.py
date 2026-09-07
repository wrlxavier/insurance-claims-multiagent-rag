"""The request-context middleware -- correlation id + one access line [M5-06].

A pure ASGI middleware, not ``BaseHTTPMiddleware``: it needs to (a) set the
correlation-id ``contextvars`` variable in the *same* task that runs the route
and its sync-threadpool handler, so a graph run started inside the request picks
it up, and (b) see the real response status by wrapping ``send``.
``BaseHTTPMiddleware`` runs the app in a child task and would defeat (a).

Per request:

- take the correlation id from ``X-Correlation-ID`` / ``X-Request-ID`` or mint one;
- bind it for the duration of the call;
- echo it on the response as ``X-Correlation-ID``;
- emit exactly one ``presentation.access`` log line -- ``path`` / ``method`` /
  ``status`` / ``duration_ms`` -- which the JSON formatter renders with the
  ``timestamp`` / ``level`` / ``correlation_id`` envelope the DoD asks for.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from infrastructure.observability.correlation import (
    CORRELATION_ID_HEADERS,
    CORRELATION_ID_RESPONSE_HEADER,
    bind_correlation_id,
    generate_correlation_id,
)

_Scope = dict[str, Any]
_Message = dict[str, Any]
_Receive = Callable[[], Awaitable[_Message]]
_Send = Callable[[_Message], Awaitable[None]]
_App = Callable[[_Scope, _Receive, _Send], Awaitable[None]]

_access_logger = logging.getLogger("presentation.access")


def _incoming_correlation_id(scope: _Scope) -> str:
    """The caller's correlation id from a known header, or a fresh one."""
    raw_headers: list[tuple[bytes, bytes]] = scope["headers"]
    headers = {key.decode("latin-1").lower(): value for key, value in raw_headers}
    for name in CORRELATION_ID_HEADERS:
        header = headers.get(name)
        if header is not None:
            value = header.decode("latin-1").strip()
            if value:
                return value
    return generate_correlation_id()


class RequestContextMiddleware:
    """Bind a correlation id per request and log one structured access line."""

    def __init__(self, app: _App) -> None:
        """Wrap the downstream ASGI app."""
        self._app = app

    async def __call__(self, scope: _Scope, receive: _Receive, send: _Send) -> None:
        """Bind the correlation id, run the app, emit the access line."""
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        correlation_id = _incoming_correlation_id(scope)
        header_value = correlation_id.encode("latin-1")
        status_code = 500
        started = time.perf_counter()

        async def send_wrapper(message: _Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = message.setdefault("headers", [])
                headers.append(
                    (CORRELATION_ID_RESPONSE_HEADER.encode("latin-1"), header_value)
                )
            await send(message)

        with bind_correlation_id(correlation_id):
            try:
                await self._app(scope, receive, send_wrapper)
            finally:
                _access_logger.info(
                    "request",
                    extra={
                        "path": scope["path"],
                        "method": scope["method"],
                        "status": status_code,
                        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    },
                )
