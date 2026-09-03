"""LangGraph orchestration.

Graph definition, nodes, edges, state schema, and the Postgres
checkpointer wiring. This is the only place in the codebase that should
import langgraph.

Import submodules directly -- ``infrastructure.graph.build`` for the graph
assembly (``build_claim_graph``, ``MAX_CLARIFICATION_ROUNDS``),
``infrastructure.graph.state`` for the pure data contract (no langgraph),
``infrastructure.graph.checkpointer`` for the Postgres checkpointer the graph
compiles against ([M4-09]), ``infrastructure.graph.nodes.<name>`` for a node.
This package ``__init__`` stays empty so importing the state schema does not
pull in langgraph.
"""
