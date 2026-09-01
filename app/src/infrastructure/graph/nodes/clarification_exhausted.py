"""Terminal node for a clarification loop that ran out of rounds ([M4-03]).

Reached only when ``route_after_intake`` sees ``missing_information`` still
non-empty after ``MAX_CLARIFICATION_ROUNDS`` rounds. It records one fact --
``clarification_exhausted = True`` -- so downstream ([M4-08] consolidation,
[M4-10] failure cataloguing) can tell "the claimant never supplied enough"
apart from "retrieval missed" (``context_sufficient``, [M4-04]) without
re-deriving it from the round counter. The gaps themselves are already in
``missing_information``; this node does not touch them.

Deterministic: no model is consulted, so the ``AuditEvent`` leaves ``model``
and ``token_usage`` as ``None``.
"""

from langgraph.runtime import Runtime

from infrastructure.graph.context import GraphContext
from infrastructure.graph.state import AuditEvent, ClaimState


def clarification_exhausted(
    state: ClaimState, runtime: Runtime[GraphContext]
) -> dict[str, object]:
    """Mark the loop as exhausted; the open gaps stay listed in state."""
    gaps = sorted(state.get("missing_information") or [])
    audit_event = AuditEvent(
        node="clarification_exhausted",
        action="exhaust_clarification_budget",
        node_input=f"gaps={gaps}",
    )
    return {
        "clarification_exhausted": True,
        "audit_trail": [audit_event],
    }
