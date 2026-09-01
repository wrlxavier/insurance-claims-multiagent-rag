"""Assembly of the agent graph -- the first edges land here ([M4-03]).

[M4-03] is the first issue that needs graph topology (a loop), so this module
is where ``StateGraph`` wiring lives from now on. Every later M4 node issue
extends it: [M4-04] retargets the ``"proceed"`` branch from ``END`` to a
retrieval node, [M4-07] fans out to the two assessment nodes, [M4-08] adds the
recommendation node, [M4-09] passes a Postgres ``checkpointer`` into
``.compile()``.

``build_claim_graph()`` returns the **uncompiled** ``StateGraph`` on purpose:
the caller (a test today, the composition root later) owns ``.compile()``, so
[M4-09] adds ``build_claim_graph().compile(checkpointer=pg_saver)`` without a
second entry point here.

Topology after [M4-03]::

    START -> intake -> route_after_intake
        "proceed"       -> END
        "clarification" -> clarification -> intake   (loop)
        "exhausted"     -> clarification_exhausted -> END

The loop is self-capping: ``route_after_intake`` sends the claim to
``clarification_exhausted`` once ``clarification_rounds`` reaches
``MAX_CLARIFICATION_ROUNDS`` with gaps still open. Termination is a property of
this router plus the cap -- the graph never relies on LangGraph's
``recursion_limit`` / ``GraphRecursionError``.
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from infrastructure.graph.context import GraphContext
from infrastructure.graph.nodes.clarification import clarification
from infrastructure.graph.nodes.clarification_exhausted import clarification_exhausted
from infrastructure.graph.nodes.intake import intake
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

    - no ``missing_information`` -> ``"proceed"`` (to ``END`` until [M4-04]).
    - gaps, and rounds left -> ``"clarification"``.
    - gaps, and the cap reached -> ``"exhausted"``.
    """
    if not (state.get("missing_information") or []):
        return "proceed"
    if state.get("clarification_rounds", 0) >= MAX_CLARIFICATION_ROUNDS:
        return "exhausted"
    return "clarification"


def build_claim_graph() -> _ClaimGraph:
    """Wire the graph as it stands after [M4-03]; the caller compiles it."""
    builder: _ClaimGraph = StateGraph(ClaimState, context_schema=GraphContext)
    builder.add_node("intake", intake)
    builder.add_node("clarification", clarification)
    builder.add_node("clarification_exhausted", clarification_exhausted)

    builder.add_edge(START, "intake")
    builder.add_conditional_edges(
        "intake",
        route_after_intake,
        {
            "clarification": "clarification",
            "exhausted": "clarification_exhausted",
            "proceed": END,
        },
    )
    builder.add_edge("clarification", "intake")
    builder.add_edge("clarification_exhausted", END)
    return builder
