"""One module per graph node -- [M4-02] onward.

A node is a module-level function

    def <name>(state: ClaimState, runtime: Runtime[GraphContext]) -> dict[str, object]

that reads ``state``, never mutates it, and returns only the keys it changed
(a partial ``ClaimState`` update) including at least one ``AuditEvent`` for
``audit_trail``. Any other function in a node module is a private ``_``-prefixed
helper. No class-based nodes.

The full authoring convention -- structured-output schemas, prompt builders,
"how to add a node" -- is in ``docs/ARCHITECTURE.md`` ([M4-01b]);
``tests/architecture/test_graph_node_conventions.py`` enforces the shape.
"""
