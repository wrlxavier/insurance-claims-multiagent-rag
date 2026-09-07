"""Shared prompt fragments for entity facts and retrieved clauses ([M5-08]).

``known_facts_block`` and ``clause_block`` were four near-identical private
helpers duplicated across ``compatibility.py``, ``recommendation.py``,
``consistency.py`` and ``clarification.py``. Consolidated here for two
reasons: the duplication itself, and — the reason this module exists under
M5-08 rather than being left alone — a single place to wrap the untrusted span
of each block in ``untrusted_content.wrap_untrusted`` so a future prompt
builder gets the same guard by construction rather than by remembering to.

Both fields intake extracts (``known_facts_block``) and clause excerpts
(``clause_block``) trace back to text this project does not control: the
former to the claim narrative, the latter to a third-party PDF. Only the
untrusted span goes inside the tag — labels, ids and clause types are
structural values this code produced, not text the model or a PDF wrote, so
they stay outside it.
"""

from infrastructure.graph.prompts.untrusted_content import wrap_untrusted
from infrastructure.graph.state import Citation, ExtractedEntities

_MAX_EXCERPT_CHARS = 700


def known_facts_block(entities: ExtractedEntities | None) -> str:
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
    if not stated:
        return "- (nada de concreto no relato)"
    return wrap_untrusted("intake_extracted_facts", "\n".join(stated))


def clause_block(citations: list[Citation], *, empty_message: str) -> str:
    """The numbered clause list, each excerpt wrapped as untrusted content."""
    if not citations:
        return empty_message
    lines = []
    for citation in citations:
        excerpt = citation.excerpt.strip()[:_MAX_EXCERPT_CHARS]
        wrapped = wrap_untrusted("retrieved_clause", excerpt)
        lines.append(f"[{citation.clause_id}] ({citation.clause_type.value}) {wrapped}")
    return "\n".join(lines)
