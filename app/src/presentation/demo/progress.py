"""A live read of one run's pipeline position, from the LangGraph checkpoint -- [M6-03].

The ``/v1`` API only reports a coarse ``pending`` / ``running`` / ``awaiting_review``
status, and the durable audit trail is empty until a decision is submitted. But
the graph's ``audit_trail`` state channel accumulates one ``AuditEvent`` per node
*during* the run, and it lives in the checkpoint -- which the checkpoint
serializer already knows how to rebuild (``AuditEvent`` is on its allowlist). So
the demo can compile the graph against the shared checkpointer and call
``get_state`` to see which nodes have run.

Read-only and best-effort: any failure (missing checkpointer schema, a dead
connection, serializer drift) is reported as ``available=False`` and the SPA
falls back to a plain elapsed timer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# The graph's nodes in pipeline order (``infrastructure.graph.build``). The SPA
# collapses the three assessment nodes into one step and hides the two
# clarification nodes until they are actually seen -- all three of
# ``clarification`` / ``clarification_exhausted`` / ``injection_scan`` are
# conditional.
PIPELINE: tuple[str, ...] = (
    "intake",
    "clarification",
    "clarification_exhausted",
    "retrieval",
    "compatibility",
    "consistency",
    "injection_scan",
    "recommendation",
    "human_review",
)


@dataclass(frozen=True)
class ProgressView:
    """What ``read_progress`` found -- mapped 1:1 onto ``schemas.ProgressResponse``."""

    available: bool
    started: bool
    nodes_seen: list[str]
    next_nodes: list[str]
    interrupted: bool


def _flat(*, available: bool, started: bool) -> ProgressView:
    """A no-detail view -- unavailable, or available-but-not-started."""
    return ProgressView(
        available=available,
        started=started,
        nodes_seen=[],
        next_nodes=[],
        interrupted=False,
    )


def read_progress(assessment_id: str, database_url: str | None = None) -> ProgressView:
    """Read ``assessment_id``'s position in the pipeline from the checkpoint.

    ``database_url`` defaults to the service database, exactly as
    ``LangGraphClaimAssessmentOrchestrator`` does. Never raises.
    """
    from infrastructure.graph.build import build_claim_graph
    from infrastructure.graph.checkpointer import open_claim_checkpointer

    try:
        with open_claim_checkpointer(database_url) as saver:
            graph = build_claim_graph().compile(checkpointer=saver)
            snapshot = graph.get_state({"configurable": {"thread_id": assessment_id}})
    except Exception:
        logger.warning(
            "demo: could not read run progress for %s", assessment_id, exc_info=True
        )
        return _flat(available=False, started=False)

    if snapshot.created_at is None:
        return _flat(available=True, started=False)

    values = snapshot.values if isinstance(snapshot.values, dict) else {}
    return ProgressView(
        available=True,
        started=True,
        nodes_seen=_nodes_from_audit_trail(values.get("audit_trail")),
        next_nodes=[str(node) for node in snapshot.next],
        interrupted=bool(snapshot.interrupts),
    )


def _nodes_from_audit_trail(audit_trail: object) -> list[str]:
    """Distinct node names, in first-seen order, from the ``audit_trail`` channel.

    Each entry is an ``AuditEvent`` once the serializer has rebuilt it, but a
    type the serializer was not told about degrades to a plain ``dict`` -- so
    both shapes are read.
    """
    if not isinstance(audit_trail, list):
        return []
    seen: list[str] = []
    for event in audit_trail:
        node = _event_node(event)
        if node is not None and node not in seen:
            seen.append(node)
    return seen


def _event_node(event: Any) -> str | None:
    node = getattr(event, "node", None)
    if node is None and isinstance(event, dict):
        node = event.get("node")
    return node if isinstance(node, str) and node else None
