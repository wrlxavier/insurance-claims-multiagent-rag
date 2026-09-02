"""Assembly of the agent graph -- the first edges land here ([M4-03]).

[M4-03] is the first issue that needs graph topology (a loop), so this module
is where ``StateGraph`` wiring lives from now on. Every later M4 node issue
extends it: [M4-04] added the retrieval node on the ``"proceed"`` branch,
[M4-05] pointed the ``"assess"`` branch at the compatibility node, [M4-07]
fanned that branch out to the two assessment nodes as fixed parallel branches
with a fan-in, [M4-08] added the recommendation node as the single terminal
consolidation point every path routes through, [M4-09] passes a Postgres
``checkpointer`` into ``.compile()``.

``build_claim_graph()`` returns the **uncompiled** ``StateGraph`` on purpose:
the caller (a test today, the composition root later) owns ``.compile()``, so
[M4-09] adds ``build_claim_graph().compile(checkpointer=pg_saver)`` without a
second entry point here.

Topology after [M4-08]::

    START -> intake -> route_after_intake
        "proceed"       -> retrieval -> route_after_retrieval
                               sufficient   -> {compatibility, consistency}
                               insufficient -> recommendation
        "clarification" -> clarification -> intake   (loop)
        "exhausted"     -> clarification_exhausted -> recommendation

    {compatibility, consistency} -> recommendation -> END

``recommendation`` is the one node with an edge to ``END``.

The loop is self-capping: ``route_after_intake`` sends the claim to
``clarification_exhausted`` once ``clarification_rounds`` reaches
``MAX_CLARIFICATION_ROUNDS`` with gaps still open. Termination is a property of
this router plus the cap -- the graph never relies on LangGraph's
``recursion_limit`` / ``GraphRecursionError``.

``route_after_retrieval`` acts on the [M3-07] gate flag the retrieval node
recorded. A sufficient context fans out -- one superstep, both the compatibility
([M4-05]) and consistency ([M4-06]) nodes -- and the two converge on the
recommendation node. This is LangGraph's *fixed parallel branches* primitive (a
conditional edge to two known nodes, then a fan-in), **not** ``Send``, which is
for dynamic map-reduce over a variable number of items. The nodes write disjoint
state channels (``compatibility`` vs ``consistency``); ``audit_trail``, the one
channel both write in the superstep, carries a reducer
(``state.append_audit_events``).

The recommendation node ([M4-08]) is the single terminal consolidation point:
every terminal path routes through it (both assessment branches, the
insufficient-context path -- ``route_after_retrieval`` now returns
``["recommendation"]`` there -- and ``clarification_exhausted``). It reads
``compatibility`` / ``consistency`` when present and ``context_sufficient`` /
``clarification_exhausted`` otherwise, and always emits one
``state.Recommendation`` before ``END``. See ``docs/PARALLEL_ASSESSMENT.md`` for
the measured wall-clock gain and ``docs/RECOMMENDATION_NODE.md`` for the node.
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from infrastructure.graph.context import GraphContext
from infrastructure.graph.nodes.clarification import clarification
from infrastructure.graph.nodes.clarification_exhausted import clarification_exhausted
from infrastructure.graph.nodes.compatibility import compatibility
from infrastructure.graph.nodes.consistency import consistency
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
    branches); anything else (the gate said the context settles the question,
    or -- defensively -- no flag was written) -> ``["compatibility",
    "consistency"]``, the fixed parallel fan-out to the two assessment nodes
    ([M4-05]/[M4-06]) in one superstep. Returning a list of node names is
    LangGraph's fan-out primitive; both branches then converge on the
    recommendation node ([M4-08]).
    """
    if state.get("context_sufficient") is False:
        return ["recommendation"]
    return ["compatibility", "consistency"]


def build_claim_graph() -> _ClaimGraph:
    """Wire the graph as it stands after [M4-08]; the caller compiles it."""
    builder: _ClaimGraph = StateGraph(ClaimState, context_schema=GraphContext)
    builder.add_node("intake", intake)
    builder.add_node("clarification", clarification)
    builder.add_node("clarification_exhausted", clarification_exhausted)
    builder.add_node("retrieval", retrieval)
    builder.add_node("compatibility", compatibility)
    builder.add_node("consistency", consistency)
    builder.add_node("recommendation", recommendation)

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
        ["compatibility", "consistency", "recommendation"],
    )
    # Fan-in on the recommendation node ([M4-08]): it runs once after both
    # assessment branches complete on the sufficient-context path, and once
    # directly on the insufficient-context / clarification-exhausted paths.
    builder.add_edge("compatibility", "recommendation")
    builder.add_edge("consistency", "recommendation")
    builder.add_edge("recommendation", END)
    return builder
