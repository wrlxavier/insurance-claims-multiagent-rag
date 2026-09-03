"""Liveness endpoint -- [M5-04].

``GET /health`` returns 200 without touching any dependency, so a load balancer
can tell the process is up. Readiness (``/ready`` with per-check detail on
Postgres / Redis / the vector index) is [M5-06]'s.
"""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    """Report that the process is alive."""
    return {"status": "ok"}
