"""Liveness and readiness endpoints -- [M5-04], [M5-06].

``GET /health`` returns 200 without touching any dependency, so a load balancer
can tell the process is up.

``GET /ready`` ([M5-06]) probes Postgres, Redis and the vector index and returns
per-check detail: 200 ``{"status": "ready", ...}`` when all pass, 503
``{"status": "degraded", ...}`` naming the check that failed and why.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Response

from infrastructure.observability.readiness import CheckResult, ReadinessProbe
from presentation.dependencies import get_readiness_probe

router = APIRouter(tags=["health"])

_ReadinessProbe = Annotated[ReadinessProbe, Depends(get_readiness_probe)]


@router.get("/health")
def health() -> dict[str, str]:
    """Report that the process is alive."""
    return {"status": "ok"}


def _check_body(result: CheckResult) -> dict[str, str]:
    body = {"status": "ok" if result.ok else "error"}
    if result.detail is not None:
        body["detail"] = result.detail
    return body


@router.get("/ready")
def ready(probe: _ReadinessProbe, response: Response) -> dict[str, object]:
    """Probe every dependency; 503 with per-check detail when one is degraded."""
    results = probe.check()
    healthy = all(result.ok for result in results)
    response.status_code = 200 if healthy else 503
    return {
        "status": "ready" if healthy else "degraded",
        "checks": {result.name: _check_body(result) for result in results},
    }
