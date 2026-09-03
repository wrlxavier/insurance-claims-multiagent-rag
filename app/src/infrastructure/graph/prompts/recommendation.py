"""Prompt builder for the recommendation node's justification leg ([M4-08]).

``build_recommendation_prompt`` returns the system instruction -- wrapped in
``with_scope_preamble`` so the ``docs/SCOPE.md`` constraint rides along -- that
asks the fast model for the **prose summary only**. Everything load-bearing in
``state.Recommendation`` (the recommended action, the aggregated citations, the
consistency flags, the confidence) is computed by the node from upstream state;
the model is given the compatibility finding, the clauses it rests on and the
consistency signals, and asked to render them into one scannable paragraph.

The model is called only on the path where a compatibility assessment exists.
The claimant-gaps and retrieval-miss paths use a deterministic template in the
node and never reach this builder. Per the [M4-01b] convention no prompt text
lives in the node function; ``_known_facts`` / ``_clause_block`` mirror the
helpers in ``prompts/compatibility.py``.
"""

from infrastructure.graph.prompts.scope_preamble import with_scope_preamble
from infrastructure.graph.state import (
    Citation,
    CompatibilityAssessment,
    ConsistencySignal,
    ExtractedEntities,
)

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
    """The numbered clause list the justification may reference, and only this."""
    if not citations:
        return "(nenhuma cláusula sustenta a avaliação)"
    lines = []
    for citation in citations:
        excerpt = citation.excerpt.strip()[:_MAX_EXCERPT_CHARS]
        lines.append(f"[{citation.clause_id}] ({citation.clause_type.value}) {excerpt}")
    return "\n".join(lines)


def _flags_block(flags: list[ConsistencySignal]) -> str:
    """The consistency signals, rendered as caveats -- never part of the verdict."""
    if not flags:
        return "- (nenhum ponto de atenção registrado)"
    return "\n".join(f"- [{flag.severity}] {flag.detail}" for flag in flags)


def build_recommendation_prompt(
    entities: ExtractedEntities | None,
    compatibility: CompatibilityAssessment,
    flags: list[ConsistencySignal],
    citations: list[Citation],
) -> str:
    """Return the recommendation node's system prompt: preamble + summary task."""
    body = f"""\
The compatibility step and the consistency step have both run on this claim. \
Write one short paragraph that consolidates what they found, for a human \
reviewer who will read it in seconds and then decide. You are summarising \
existing findings -- you are not re-assessing the claim and you introduce no \
new fact.

Shape the paragraph in this order:
1. The compatibility finding first: state the verdict \
(`{compatibility.verdict.value}`) and, in one clause, why.
2. Then the clauses it rests on, by id, copied verbatim from the list below. \
Name only ids that appear in that list; if the list is empty, say the \
assessment is not clause-grounded.
3. Then the consistency attention points as caveats, if any -- flagged for the \
reviewer to check, kept separate from the verdict. If there are none, say the \
account is internally coherent.

Rules:
- Brazilian Portuguese, at most four sentences.
- No real-world coverage outcome: do not say the claim is approved, rejected, \
payable, or resolved. You are describing consistency with a registered \
product's conditions, nothing more.
- Do not restate the raw narrative; assume the reviewer has it.

The compatibility verdict: {compatibility.verdict.value}
The compatibility reasoning:
{compatibility.reasoning.strip() or "(sem raciocínio registrado)"}

What intake extracted from the claim:
{_known_facts(entities)}

Clauses the assessment rests on (the only ids you may cite):
{_clause_block(citations)}

Consistency attention points:
{_flags_block(flags)}
"""
    return with_scope_preamble(body)
