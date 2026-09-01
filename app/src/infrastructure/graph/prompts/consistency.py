"""Prompt builder for the consistency node's semantic-judgement leg ([M4-06]).

``build_consistency_prompt`` returns the system instruction -- wrapped in
``with_scope_preamble`` so the ``docs/SCOPE.md`` constraint rides along -- that
asks the fast model for the *judgement* half of the consistency check:
narrative coherence, description vs. stated event type, and vagueness where a
claimant would normally give detail. The arithmetic half (dates, amounts, field
contradictions, product-line/event-type collisions) is
``infrastructure.graph.consistency_checks`` and never reaches the model. Per the
[M4-01b] convention no prompt text lives in the node function.

Two things the prompt is careful about:

- **No verdict.** The scope preamble ends by naming the verdict vocabulary the
  *other* nodes use; this node produces none, so the body says so explicitly to
  remove the tension.
- **Not a fraud detector.** The prompt forbids speculation about intent -- the
  node flags internal inconsistencies for a reviewer, nothing more.
"""

from infrastructure.graph.prompts.scope_preamble import with_scope_preamble
from infrastructure.graph.state import ExtractedEntities


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


def build_consistency_prompt(entities: ExtractedEntities | None) -> str:
    """Return the consistency node's system prompt: preamble + judgement task."""
    body = f"""\
You are given one insurance claim narrative and the structured record an intake \
step extracted from it. Read them together and list the ways the claim is \
internally inconsistent -- where the story does not hold together on its own \
terms. You produce a list of signals for a human reviewer and nothing else.

You produce NO verdict. The verdict vocabulary in the section above describes \
what other parts of the system do; your only output here is a list of signals. \
Do not judge whether the claim is consistent with the product's conditions, \
whether it should be paid, or whether it is valid. Do not speculate about fraud \
or the claimant's intent -- that is neither your task nor something this data \
supports.

Flag only these kinds of inconsistency:
- narrative_coherence: the sequence of events described cannot have happened as \
told, or a later statement contradicts an earlier one.
- description_event_type_mismatch: the free description does not match the \
event type intake recorded (for example, the description is of a mechanical \
breakdown but the event type says collision).
- unexpected_vagueness: a material loss is described with none of the detail a \
claimant would normally give -- no location, no sequence, no specifics -- so \
that the account cannot be assessed on its face.

Rules:
- Most narratives are internally coherent. Return an empty list when nothing is \
genuinely inconsistent; do not manufacture a signal.
- At most a handful of signals. One sentence each, in Brazilian Portuguese, \
naming the specific problem.
- severity is `attention` when a reviewer should look before proceeding, `info` \
for a minor note.

What intake extracted from the claim:
{_known_facts(entities)}
"""
    return with_scope_preamble(body)
