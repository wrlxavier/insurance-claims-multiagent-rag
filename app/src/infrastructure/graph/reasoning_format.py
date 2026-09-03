"""The rendered-reasoning contract between the graph and its evaluations [M4-10].

``state.CompatibilityAssessment.reasoning`` is a plain string, so the
structured ``(statement, clause_ids)`` pairs the compatibility node ([M4-05])
reasons in do not survive into graph state. That makes the rendered format a
**contract**, not a display detail: [M4-10] has to check the DoD's "every
assertion carries a clause id" property from a completed run's state alone, and
the only way back to the assertions is to parse what was rendered.

:func:`render_reasoning` and :func:`parse_reasoning` are exact inverses, pinned
by a round-trip test. They live here rather than in ``nodes/compatibility.py``
because a public function in the ``nodes`` package is required by
``tests/architecture/test_graph_node_conventions.py`` to *be* a node -- pure
helpers belong beside ``state.py``, the way ``consistency_checks.py`` does.
"""

import re

from infrastructure.graph.schemas import ReasonedAssertion

# The placeholder an assertion with no clause id renders as. Only reachable on
# an `insufficient_information` verdict: `compatibility._grounding_errors`
# rejects a citation-free assertion on any settled one.
_NO_CLAUSES = "—"

NO_ASSERTIONS_REASONING = "Nenhuma afirmação fundamentada foi produzida."

_REASONING_LINE_RE = re.compile(
    r"^\d+\.\s+(?P<statement>.+?)\s+\[cláusulas:\s*(?P<cited>[^\]]*)\]$"
)


def render_reasoning(assertions: list[ReasonedAssertion]) -> str:
    """Render the assertion list into the plain ``reasoning`` string state holds.

    One numbered line per assertion, its clause ids in a trailing bracket. See
    the module docstring for why the format is a contract.
    """
    if not assertions:
        return NO_ASSERTIONS_REASONING
    lines = []
    for index, assertion in enumerate(assertions, start=1):
        cited = ", ".join(assertion.clause_ids) if assertion.clause_ids else _NO_CLAUSES
        lines.append(f"{index}. {assertion.statement.strip()} [cláusulas: {cited}]")
    return "\n".join(lines)


def parse_reasoning(reasoning: str) -> list[ReasonedAssertion]:
    """Recover the assertions :func:`render_reasoning` wrote; ``[]`` if none.

    The inverse of :func:`render_reasoning`. A line that does not match the
    rendered shape is skipped rather than raising: an *abstaining* assessment
    writes free prose into the same field by design
    (``nodes/compatibility._abstain``), and that prose is not malformed output
    to be rejected -- it simply carries no assertions.
    """
    assertions: list[ReasonedAssertion] = []
    for line in reasoning.splitlines():
        match = _REASONING_LINE_RE.match(line.strip())
        if match is None:
            continue
        cited = match.group("cited").strip()
        clause_ids = (
            []
            if cited == _NO_CLAUSES
            else [part.strip() for part in cited.split(",") if part.strip()]
        )
        assertions.append(
            ReasonedAssertion(
                statement=match.group("statement").strip(), clause_ids=clause_ids
            )
        )
    return assertions
