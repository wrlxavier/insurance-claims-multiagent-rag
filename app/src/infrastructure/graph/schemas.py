"""Structured-output schemas for graph nodes -- [M4-01b].

One frozen Pydantic ``<Node>Output`` per LLM node -- the exact shape passed to
``llm.with_structured_output(...)``. Kept separate from the state sub-models in
``state.py``: the node maps its output schema onto the state model (for
example, the model returns clause-id strings and the node hydrates ``Citation``
objects with the retrieved clauses' provenance). Empty until the first node
lands ([M4-02]).
"""
