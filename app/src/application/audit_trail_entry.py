"""The audit-trail entry DTO the application layer speaks in [M5-04].

The graph-free projection of one ``infrastructure.graph.state.AuditEvent`` row
of a run's durable trail. It exists for two callers:

- ``GetAuditTrail`` / ``AuditTrailReader`` -- the read model behind
  ``GET /v1/assessments/{id}/audit``;
- ``OrchestratorResult.audit_records`` -- the trail the resume path captures so
  the use case can persist it in the *same* transaction as the settled record
  (``docs/ARCHITECTURE.md``, the [M5-04] transactional fold), instead of the
  graph node committing it on its own.

``TokenUsage`` is flattened into the three integer fields, matching
``AuditEventRow``; ``payload`` is the optional JSON detail the human-review
event carries (the analyst's whole decision).

Standard library and typing only -- the application layer never imports
Pydantic or ``infrastructure`` (enforced by
tests/architecture/test_layer_boundaries.py).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class AuditTrailEntry:
    """One entry of a graph run's durable audit trail, framework-free."""

    sequence: int
    timestamp: datetime
    node: str
    action: str
    model: str | None = None
    model_version: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    confidence: float | None = None
    node_input: str | None = None
    payload: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        """Reject a negative sequence or empty node/action."""
        if self.sequence < 0:
            raise ValueError(
                f"AuditTrailEntry.sequence must not be negative, got {self.sequence}"
            )
        for name in ("node", "action"):
            if not getattr(self, name):
                raise ValueError(f"AuditTrailEntry.{name} must not be empty")
