"""The human checkpoint: pause before anything is final -- [M4-09].

The graph's last node. It surfaces the whole recommendation, stops, and does not
continue until a person has approved, edited or rejected it. The analyst's
decision is recorded *alongside* the system's opinion -- ``recommendation`` is
never written here, so an edit lives in ``human_decision.edited_recommendation``
and the original stays exactly as the machine produced it.

**Order inside this node is load-bearing.** LangGraph re-runs an interrupted node
from the top on resume: everything above the ``interrupt()`` call runs twice
(once on the invocation that hits the pause, once on the invocation that
resumes), everything below it once. So the node is deliberately pure until the
pause -- read state, build a payload -- and does its one side effect, the durable
audit write, strictly afterwards.

**It also must not raise after the pause.** A resume value is stored in the
checkpoint's pending writes; if the node then raises, that same value is replayed
on every later ``Command(resume=...)`` and the thread can never be finished --
the decision would be lost for good. Two consequences:

* a malformed decision **re-asks** instead of raising: the node calls
  ``interrupt()`` again with the validation error attached, and the next resume
  supplies a fresh value. This is not a graph loop -- every iteration hands
  control back to the caller, so the node cannot spin -- and the interrupt
  sequence stays deterministic (call *n* always returns resume value *n*), which
  is the condition the LangGraph documentation puts on repeated interrupts;
* a failing audit sink **degrades**: it is logged and recorded as an audit event
  of its own, but the decision is still returned. Graph state is itself durable
  through the checkpointer, so nothing is lost that a re-run cannot rewrite.

See ``docs/HUMAN_CHECKPOINT.md``.
"""

import logging
from collections.abc import Sequence

from langgraph.config import get_config
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from pydantic import ValidationError

from infrastructure.graph.context import AuditTrailSink, GraphContext
from infrastructure.graph.state import (
    SCHEMA_VERSION,
    AuditEvent,
    AuditRecord,
    ClaimState,
    HumanDecision,
    Recommendation,
)

logger = logging.getLogger(__name__)

DECISION_OPTIONS = ("approve", "edit", "reject")

_NODE_INPUT_PREVIEW_CHARS = 200
_ERROR_PREVIEW_CHARS = 500


def human_review(
    state: ClaimState, runtime: Runtime[GraphContext]
) -> dict[str, object]:
    """Pause for the analyst, then record their decision and the durable trail."""
    recommendation = state.get("recommendation")
    if recommendation is None:
        # Every path through the graph runs the recommendation node first, so
        # this is an assembly error, not a runtime one. Raised before the pause,
        # where there is no resume value to strand.
        raise ValueError(
            "human_review reached with no recommendation in state -- the "
            "recommendation node must run before the human checkpoint."
        )

    payload = _review_payload(state, recommendation)

    # --- everything above this line must stay free of side effects ---
    decision = _await_decision(payload)
    # --- everything below runs exactly once per completed checkpoint ---

    decision_event = AuditEvent(
        node="human_review",
        action=f"human_decision:{decision.decision}",
        node_input=(
            f"decision={decision.decision} "
            f"edited={decision.edited_recommendation is not None} "
            f"notes_len={len(decision.notes)}"
        )[:_NODE_INPUT_PREVIEW_CHARS],
    )
    events = [decision_event]

    failure = _persist_audit_trail(
        runtime.context.audit_sink,
        claim_id=state["claim_id"],
        prior_events=state.get("audit_trail") or [],
        decision_event=decision_event,
        decision=decision,
    )
    if failure is not None:
        events.append(failure)

    return {"human_decision": decision, "audit_trail": events}


def _review_payload(
    state: ClaimState, recommendation: Recommendation
) -> dict[str, object]:
    """The full recommendation, plus the context a reviewer needs to judge it.

    Plain JSON-serializable values: this crosses the checkpoint (and, in [M5-04],
    an HTTP boundary), and it is what an interrupted ``invoke`` hands back under
    ``__interrupt__``. ``recommendation`` is dumped in ``mode="json"`` so the
    caller does not need this project's Pydantic models to read it.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_id": state["claim_id"],
        "recommendation": recommendation.model_dump(mode="json"),
        "context_sufficient": state.get("context_sufficient"),
        "clarification_exhausted": bool(state.get("clarification_exhausted")),
        "missing_information": sorted(state.get("missing_information") or []),
        "decision_options": list(DECISION_OPTIONS),
    }


def _await_decision(payload: dict[str, object]) -> HumanDecision:
    """Interrupt until a well-formed decision arrives, then return it.

    The resume value may be a ``HumanDecision`` (an in-process caller) or a
    mapping (anything that crossed a process or HTTP boundary); Pydantic accepts
    both, including a nested ``edited_recommendation``, and enforces the
    "``edit`` carries a revision" rule ``HumanDecision`` already declares.

    An invalid value re-opens the checkpoint with the error attached rather than
    raising -- see the module docstring for why raising here is unrecoverable.
    """
    prompt = payload
    while True:
        # `interrupt()` stays outside the `try`, and the `except` names one
        # exception type: it pauses by *raising*, so a bare try/except around it
        # would swallow the pause itself.
        raw = interrupt(prompt)
        try:
            return HumanDecision.model_validate(raw)
        except ValidationError as exc:
            prompt = {**payload, "error": str(exc)[:_ERROR_PREVIEW_CHARS]}


def _persist_audit_trail(
    sink: AuditTrailSink | None,
    *,
    claim_id: str,
    prior_events: Sequence[AuditEvent],
    decision_event: AuditEvent,
    decision: HumanDecision,
) -> AuditEvent | None:
    """Write the run's whole trail to durable storage. Return a failure event, if any.

    The trail is written once, here, rather than incrementally by every node:
    this is the single point every path reaches, it is the only place a side
    effect is safe from the interrupt's re-execution, and the human decision --
    the part a compliance reader is actually after -- does not exist until now.
    The cost is that a run abandoned before the checkpoint leaves no
    ``audit_event`` rows; the checkpointer holds that state until the thread is
    resumed, which is what makes the trade acceptable.

    ``decision`` rides along in the decision event's ``payload`` so notes,
    ``decided_at`` and any ``edited_recommendation`` are readable in SQL.
    """
    if sink is None:
        return None

    records = [AuditRecord(event) for event in prior_events]
    records.append(AuditRecord(decision_event, decision.model_dump(mode="json")))

    try:
        sink.record(claim_id=claim_id, thread_id=_thread_id(), records=records)
    except Exception as exc:  # noqa: BLE001 - never lose the decision to a sink failure
        logger.exception("failed to persist the audit trail for claim %s", claim_id)
        return AuditEvent(
            node="human_review",
            action="persist_audit_trail_failed",
            node_input=f"{type(exc).__name__}: {exc}"[:_NODE_INPUT_PREVIEW_CHARS],
        )
    return None


def _thread_id() -> str:
    """The LangGraph thread this run belongs to -- the audit trail's row key.

    Read from the run config rather than ``runtime``, which does not expose it in
    LangGraph 1.2. Distinct from ``claim_id``: one claim re-submitted for a
    second assessment is a second thread with a trail of its own, and keying on
    the claim would make the two collide.
    """
    return str(get_config()["configurable"]["thread_id"])
