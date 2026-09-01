"""Prompt builder for the compatibility assessment node ([M4-05]).

``build_compatibility_prompt`` returns the system instruction -- wrapped in
``with_scope_preamble`` so the ``docs/SCOPE.md`` constraint rides along -- that
turns the retrieved clauses plus what intake extracted into a
``schemas.CompatibilityOutput``. The claim narrative itself is a separate human
message the node adds; per the [M4-01b] convention, no prompt text lives in the
node function.

Two rules the prompt carries that the DoD names explicitly:

- every assertion must list the ``clause_ids`` it rests on -- the node rejects
  and retries an output that breaks this, it is not patched afterwards;
- retrieved exclusions must be weighed against retrieved coverage *in the
  reasoning*: when a coverage clause seems to include the event but an exclusion
  removes it, the assertions must say so and cite both.
"""

from infrastructure.graph.prompts.scope_preamble import with_scope_preamble
from infrastructure.graph.state import Citation, ExtractedEntities

_MAX_EXCERPT_CHARS = 700


def _known_facts(entities: ExtractedEntities | None) -> str:
    """The entity summary block -- what intake extracted, or a placeholder."""
    if entities is None:
        return "- (intake não extraiu nada estruturado)"
    pairs = [
        ("tipo de evento", entities.event_type),
        ("data", entities.event_date),
        ("descrição", entities.description),
        ("valor estimado", entities.estimated_amount),
        ("veículo", entities.vehicle_info),
        ("processo SUSEP", entities.susep_process),
        ("ramo do produto", entities.product_line),
    ]
    stated = [f"- {label}: {value}" for label, value in pairs if value is not None]
    return "\n".join(stated) or "- (nada de concreto no relato)"


def _clause_block(citations: list[Citation]) -> str:
    """The numbered clause list the assertions must cite from."""
    if not citations:
        return "(nenhuma cláusula recuperada)"
    lines = []
    for citation in citations:
        excerpt = citation.excerpt.strip()[:_MAX_EXCERPT_CHARS]
        lines.append(f"[{citation.clause_id}] ({citation.clause_type.value}) {excerpt}")
    return "\n".join(lines)


def build_compatibility_prompt(
    entities: ExtractedEntities | None,
    citations: list[Citation],
) -> str:
    """Return the compatibility node's system prompt: preamble + assessment task."""
    body = f"""\
You are given one insurance claim and a numbered list of clauses retrieved from \
the registered product's general and special conditions. Decide whether the \
described event is consistent with those conditions, reasoning only from the \
clauses listed -- do not rely on outside knowledge of what a policy usually says.

The verdict is exactly one of:
- compatible: a retrieved clause shows the event fits the product's conditions.
- incompatible: a retrieved clause -- typically an exclusion -- rules the event out.
- insufficient_information: the retrieved clauses do not settle the question. \
Return this rather than guessing whenever the clauses are silent, ambiguous, or \
only tangentially related.

Rules for the reasoning:
- Break it into assertions, one point each. Every assertion must list, in its \
`clause_ids`, the ids of the retrieved clauses that support that exact point, \
copied verbatim from the list below. An assertion with no clause id is invalid.
- When the list contains both a coverage clause that appears to include the \
event and an exclusion that would remove it, weigh the two against each other in \
an explicit assertion and cite both clause ids. Do not stop at the coverage side.
- If you cannot ground a point in a retrieved clause, do not make the point.
- confidence is how firmly the retrieved clauses settle the question, 0 to 1.

What intake extracted from the claim:
{_known_facts(entities)}

Retrieved clauses:
{_clause_block(citations)}
"""
    return with_scope_preamble(body)
