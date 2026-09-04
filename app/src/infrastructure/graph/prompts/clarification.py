"""Prompt builder for the clarification node ([M4-03]).

``build_clarification_prompt`` returns the system instruction -- wrapped in
``with_scope_preamble`` -- that turns a set of ``missing_information`` tags plus
what intake already extracted into one concrete question per gap. Per the
[M4-01b] convention no prompt text lives in the node function.

``CLARIFICATION_FALLBACK_TEMPLATES`` is the deterministic backstop: one question
per tag, used for any gap the model leaves unaddressed and for every gap when
the LLM call fails all its retries. A question generator has a sane fallback
value (unlike intake's extraction), so the loop can always make progress.
"""

from infrastructure.graph.prompts.prompt_fragments import known_facts_block
from infrastructure.graph.prompts.scope_preamble import with_scope_preamble
from infrastructure.graph.state import ClarificationQuestion, ExtractedEntities

# tag -> what to ask when the fact is missing. Keys match
# schemas.MissingInfoTag. Phrased as the claimant would be asked, in informal
# Brazilian Portuguese, mirroring the conditions in
# prompts.intake.MISSING_INFO_GUIDE.
CLARIFICATION_FALLBACK_TEMPLATES: dict[str, str] = {
    "data_evento_vigencia": ("Em que dia (ou por volta de quando) o evento aconteceu?"),
    "ambito_geografico": (
        "Onde o evento aconteceu? Foi na sua cidade, numa viagem pelo país ou "
        "fora do Brasil?"
    ),
    "uso_do_veiculo": (
        "No momento do evento, o veículo estava sendo usado para transporte de "
        "passageiros por aplicativo, transporte de carga ou algum outro uso "
        "remunerado?"
    ),
    "valor_franquia_limite": (
        "Qual a gravidade do dano ou da perda? Se possível, dê uma estimativa "
        "do valor do prejuízo."
    ),
    "tipo_evento_condicao": (
        "O que exatamente causou o dano? Descreva se foi uma batida, um ato de "
        "vandalismo, uma falha do próprio veículo ou outra coisa."
    ),
}


def build_clarification_prompt(
    entities: ExtractedEntities | None,
    missing_information: list[str],
    asked_questions: list[ClarificationQuestion],
) -> str:
    """Return the clarification node's system prompt: preamble + question task."""
    gaps = "\n".join(f"- {tag}" for tag in missing_information)
    already_asked = (
        "\n".join(f"- ({q.field}) {q.question}" for q in asked_questions)
        or "- (nenhuma ainda)"
    )
    body = f"""\
An intake step has read one insurance claim narrative and found that some \
load-bearing facts are missing. Your job is to write the questions that would \
get those facts from the claimant. Do not assess or answer the claim.

What intake already knows:
{known_facts_block(entities)}

The missing facts, by tag -- write exactly one question for each:
{gaps}

Questions already put to the claimant on earlier rounds (still unanswered):
{already_asked}

Rules:
- Exactly one question per tag above, in the `questions` list, each with its \
`field` set to the tag it addresses.
- Write in informal Brazilian Portuguese, the way a claims assistant would ask.
- Make each question specific to this claim: refer to what the claimant \
already said. Never a generic "envie mais detalhes" or "poderia detalhar".
- If a tag was already asked, rephrase it more concretely rather than \
repeating the same wording.
"""
    return with_scope_preamble(body)
