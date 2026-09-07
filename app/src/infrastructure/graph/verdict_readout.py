"""Read the effective verdict from the recommendation node's audit event -- [M4-08].

``state.Recommendation`` has no ``verdict`` field by [M4-08]'s design: everything
load-bearing is derived, and the derivation the recommendation node settled on is
recorded in its own ``AuditEvent`` as ``posture=<p> verdict=<v> ...``. Anything
that needs the end-to-end verdict -- ``scripts/eval_end_to_end.py``, and the
[M5-04] orchestrator adapter building an ``OrchestratorResult`` -- reads it from
there rather than re-deriving it, so both see exactly what the graph decided.

``verdict=`` is authoritative; ``posture=`` is the fallback, and also says *why*
an abstention happened, not only that it did.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from domain.verdict import Verdict
from infrastructure.graph.state import AuditEvent

_RECOMMENDATION_NODE = "recommendation"

_POSTURE_RE = re.compile(r"posture=(\S+)")
_VERDICT_RE = re.compile(r"verdict=(\S+)")

# Maps the recommendation node's posture onto the M0-06 verdict vocabulary. The
# posture carries more than the verdict (it distinguishes claimant gaps from a
# retrieval miss from an inconclusive assessment) -- all three collapse to
# insufficient_information here.
_POSTURE_VERDICT: dict[str, Verdict] = {
    "compatible": Verdict.COMPATIBLE,
    "incompatible": Verdict.INCOMPATIBLE,
    "inconclusive": Verdict.INSUFFICIENT_INFORMATION,
    "claimant_gaps": Verdict.INSUFFICIENT_INFORMATION,
    "retrieval_miss": Verdict.INSUFFICIENT_INFORMATION,
    "no_assessment": Verdict.INSUFFICIENT_INFORMATION,
}

_VERDICT_VALUES = {member.value for member in Verdict}


def _last_recommendation_input(audit_trail: Sequence[AuditEvent]) -> str | None:
    """The ``node_input`` of the last recommendation-node event, if any."""
    for event in reversed(list(audit_trail)):
        if event.node == _RECOMMENDATION_NODE and event.node_input:
            return event.node_input
    return None


def posture_of(audit_trail: Sequence[AuditEvent]) -> str | None:
    """The posture string the recommendation node recorded, or ``None``."""
    node_input = _last_recommendation_input(audit_trail)
    if node_input is None:
        return None
    match = _POSTURE_RE.search(node_input)
    return match.group(1) if match else None


def effective_verdict(audit_trail: Sequence[AuditEvent]) -> Verdict | None:
    """The verdict the recommendation node settled on, or ``None`` if unreadable.

    ``verdict=`` in the node's audit input wins; failing that, ``posture=``
    mapped through :data:`_POSTURE_VERDICT`.
    """
    node_input = _last_recommendation_input(audit_trail)
    if node_input is None:
        return None

    verdict_match = _VERDICT_RE.search(node_input)
    if verdict_match and verdict_match.group(1) in _VERDICT_VALUES:
        return Verdict(verdict_match.group(1))

    posture_match = _POSTURE_RE.search(node_input)
    if posture_match:
        return _POSTURE_VERDICT.get(posture_match.group(1))
    return None
