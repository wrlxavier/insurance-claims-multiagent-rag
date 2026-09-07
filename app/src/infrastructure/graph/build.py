"""Assembly of the agent graph -- the first edges land here ([M4-03]).

[M4-03] is the first issue that needs graph topology (a loop), so this module
is where ``StateGraph`` wiring lives from now on. Every later M4 node issue
extends it: [M4-04] added the retrieval node on the ``"proceed"`` branch,
[M4-05] pointed the ``"assess"`` branch at the compatibility node, [M4-07]
fanned that branch out to the two assessment nodes as fixed parallel branches
with a fan-in, [M4-08] added the recommendation node as the single terminal
consolidation point every path routes through, [M4-09] appended the human
checkpoint behind it.

``build_claim_graph()`` returns the **uncompiled** ``StateGraph`` on purpose:
the caller (a test today, the composition root later) owns ``.compile()``, so
[M4-09] compiles it as
``build_claim_graph().compile(checkpointer=pg_saver)`` without a second entry
point here.

Topology after [M5-08 Appendix]::

    START -> intake -> route_after_intake
        "proceed"       -> retrieval -> route_after_retrieval
                               sufficient   -> {compatibility, consistency,
                                                 injection_scan}
                               insufficient -> recommendation
        "clarification" -> clarification -> intake   (loop)
        "exhausted"     -> clarification_exhausted -> recommendation

    {compatibility, consistency, injection_scan}
        -> recommendation -> human_review -> END

``human_review`` is the one node with an edge to ``END``.

Every node is registered through ``_instrumented`` ([M5-06]): a wrapper that
brackets each run with a correlation-id-tagged ``node.start`` / ``node.completed``
log line (and ``node.failed`` on a real exception). It is applied here, not in
the node files, so the ``(state, runtime) -> dict`` convention and its
enforcement test stay untouched.

**Compiling this graph without a checkpointer is a mistake, not an option.**
``human_review`` calls ``interrupt()``, and LangGraph does not complain when
there is nowhere to persist the pause -- ``.invoke()`` simply returns early with
``__interrupt__`` set and the run can never be resumed. Every caller passes a
checkpointer and a ``thread_id``: ``infrastructure.graph.checkpointer`` for the
real one, ``InMemorySaver`` for a unit test.

The loop is self-capping: ``route_after_intake`` sends the claim to
``clarification_exhausted`` once ``clarification_rounds`` reaches
``MAX_CLARIFICATION_ROUNDS`` with gaps still open. Termination is a property of
this router plus the cap -- the graph never relies on LangGraph's
``recursion_limit`` / ``GraphRecursionError``.

``route_after_retrieval`` acts on the [M3-07] gate flag the retrieval node
recorded. A sufficient context fans out -- one superstep, the compatibility
([M4-05]) and consistency ([M4-06]) nodes plus the optional injection-scan
node ([M5-08 Appendix]) -- and all three converge on the recommendation node.
This is LangGraph's *fixed parallel branches* primitive (a conditional edge to
three known nodes, then a fan-in), **not** ``Send``, which is for dynamic
map-reduce over a variable number of items. ``compatibility`` and
``consistency`` write disjoint state channels; ``injection_scan`` writes only
``audit_trail``, the one channel all three can write in the same superstep,
which carries a reducer (``state.append_audit_events``) for exactly this
reason.

``injection_scan`` is a fixed member of this fan-out regardless of whether
the optional classifier is configured -- see
``infrastructure.graph.context.InjectionClassifierPort``/``NO_CLASSIFIER``
and ``infrastructure.graph.nodes.injection_scan``. Keeping the graph's
topology static and reading the toggle only at the composition root (never
here) is deliberate: ``build_claim_graph()`` stays deterministic, and every
existing test/eval script that builds a bare ``GraphContext`` exercises the
same three-branch fan-out it always did, just with a classifier that is a
guaranteed no-op.

The recommendation node ([M4-08]) is the single terminal consolidation point:
every terminal path routes through it (both assessment branches, the
insufficient-context path -- ``route_after_retrieval`` now returns
``["recommendation"]`` there -- and ``clarification_exhausted``). It reads
``compatibility`` / ``consistency`` when present and ``context_sufficient`` /
``clarification_exhausted`` otherwise, and always emits one
``state.Recommendation``. See ``docs/PARALLEL_ASSESSMENT.md`` for the measured
wall-clock gain and ``docs/RECOMMENDATION_NODE.md`` for the node.

The human checkpoint ([M4-09]) sits between that and ``END``, unconditionally --
it is the product behaviour, not a mode, so there is no flag that removes it.
It surfaces the recommendation, pauses on ``interrupt()``, records the analyst's
approve / edit / reject beside (never over) the system's opinion, and writes the
audit trail somewhere a compliance reader can query. See
``docs/HUMAN_CHECKPOINT.md``.
"""

import logging
import time
from collections.abc import Callable
from typing import Literal, cast

from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from infrastructure.graph.context import GraphContext
from infrastructure.graph.nodes.clarification import clarification
from infrastructure.graph.nodes.clarification_exhausted import clarification_exhausted
from infrastructure.graph.nodes.compatibility import compatibility
from infrastructure.graph.nodes.consistency import consistency
from infrastructure.graph.nodes.human_review import human_review
from infrastructure.graph.nodes.injection_scan import injection_scan
from infrastructure.graph.nodes.intake import intake
from infrastructure.graph.nodes.recommendation import recommendation
from infrastructure.graph.nodes.retrieval import retrieval
from infrastructure.graph.state import ClaimState

# How many clarification rounds a claim gets before the loop gives up and
# terminates as insufficient_information. Product behaviour, defined and tested
# in code (like application.use_cases.llm_retry_defaults' retry constants), not
# a deployment knob -- there is no .env pin.
MAX_CLARIFICATION_ROUNDS = 2

_ClaimGraph = StateGraph[ClaimState, GraphContext, ClaimState, ClaimState]

_Node = Callable[[ClaimState, Runtime[GraphContext]], dict[str, object]]

# One log line at the start and end of every node run, tagged with the run's
# correlation id ([M5-06]). Emitting it here -- once, around every node -- rather
# than in each node file keeps the `(state, runtime) -> dict` node convention
# untouched and gives per-node timing for free (a head start on [M5-10]).
_node_logger = logging.getLogger("infrastructure.graph.node")


def _instrumented[NodeT: _Node](node: NodeT) -> NodeT:
    """Wrap a node so every run brackets itself with a correlation-tagged log line."""

    def wrapper(state: ClaimState, runtime: Runtime[GraphContext]) -> dict[str, object]:
        correlation_id = runtime.context.correlation_id or "-"
        _node_logger.info(
            "node.start",
            extra={"node": node.__name__, "correlation_id": correlation_id},
        )
        started = time.perf_counter()
        try:
            result = node(state, runtime)
        except GraphBubbleUp:
            # `interrupt()` in `human_review` and other LangGraph control flow --
            # not a failure, let it bubble silently.
            raise
        except Exception:
            _node_logger.exception(
                "node.failed",
                extra={
                    "node": node.__name__,
                    "correlation_id": correlation_id,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
            raise
        _node_logger.info(
            "node.completed",
            extra={
                "node": node.__name__,
                "correlation_id": correlation_id,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        return result

    wrapper.__name__ = node.__name__
    wrapper.__qualname__ = getattr(node, "__qualname__", node.__name__)
    wrapper.__doc__ = node.__doc__
    return cast("NodeT", wrapper)


def route_after_intake(
    state: ClaimState,
) -> Literal["proceed", "clarification", "exhausted"]:
    """Decide what happens after an intake pass.

    - no ``missing_information`` -> ``"proceed"`` (to the retrieval node).
    - gaps, and rounds left -> ``"clarification"``.
    - gaps, and the cap reached -> ``"exhausted"``.
    """
    if not (state.get("missing_information") or []):
        return "proceed"
    if state.get("clarification_rounds", 0) >= MAX_CLARIFICATION_ROUNDS:
        return "exhausted"
    return "clarification"


def route_after_retrieval(state: ClaimState) -> list[str]:
    """Branch on the [M3-07] gate's verdict, which the retrieval node recorded.

    ``context_sufficient is False`` -> ``["recommendation"]`` (the node produces
    an ``insufficient_information`` recommendation directly, with no assessment
    branches -- the optional injection scan is skipped on this path too, since
    there is no retrieved-clause set for it to score); anything else (the gate
    said the context settles the question, or -- defensively -- no flag was
    written) -> ``["compatibility", "consistency", "injection_scan"]``, the
    fixed parallel fan-out to the two assessment nodes ([M4-05]/[M4-06]) plus
    the optional, advisory-only injection scan ([M5-08 Appendix]) in one
    superstep. Returning a list of node names is LangGraph's fan-out primitive;
    all three branches then converge on the recommendation node ([M4-08]).
    """
    if state.get("context_sufficient") is False:
        return ["recommendation"]
    return ["compatibility", "consistency", "injection_scan"]


def build_claim_graph() -> _ClaimGraph:
    """Wire the graph as it stands after [M4-09]; the caller compiles it.

    The caller must supply a checkpointer to ``.compile()`` and a ``thread_id``
    to every invocation -- see the module docstring.
    """
    builder: _ClaimGraph = StateGraph(ClaimState, context_schema=GraphContext)
    builder.add_node("intake", _instrumented(intake))
    builder.add_node("clarification", _instrumented(clarification))
    builder.add_node("clarification_exhausted", _instrumented(clarification_exhausted))
    builder.add_node("retrieval", _instrumented(retrieval))
    builder.add_node("compatibility", _instrumented(compatibility))
    builder.add_node("consistency", _instrumented(consistency))
    builder.add_node("injection_scan", _instrumented(injection_scan))
    builder.add_node("recommendation", _instrumented(recommendation))
    builder.add_node("human_review", _instrumented(human_review))

    builder.add_edge(START, "intake")
    builder.add_conditional_edges(
        "intake",
        route_after_intake,
        {
            "clarification": "clarification",
            "exhausted": "clarification_exhausted",
            "proceed": "retrieval",
        },
    )
    builder.add_edge("clarification", "intake")
    builder.add_edge("clarification_exhausted", "recommendation")
    builder.add_conditional_edges(
        "retrieval",
        route_after_retrieval,
        ["compatibility", "consistency", "injection_scan", "recommendation"],
    )
    # Fan-in on the recommendation node ([M4-08]): it runs once after both
    # assessment branches (plus the optional injection scan, [M5-08 Appendix])
    # complete on the sufficient-context path, and once directly on the
    # insufficient-context / clarification-exhausted paths.
    builder.add_edge("compatibility", "recommendation")
    builder.add_edge("consistency", "recommendation")
    builder.add_edge("injection_scan", "recommendation")
    # The human checkpoint ([M4-09]) is the last thing before END, so no path
    # reaches a final state without a person having seen the recommendation.
    builder.add_edge("recommendation", "human_review")
    builder.add_edge("human_review", END)
    return builder
