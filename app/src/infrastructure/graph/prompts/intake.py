"""Prompt builder for the intake node ([M4-02]).

``build_intake_prompt`` returns the system instruction -- wrapped in
``with_scope_preamble`` so the ``docs/SCOPE.md`` constraint rides along with
every call -- that turns a free-text Brazilian-Portuguese claim narrative into
an ``IntakeOutput``. The claim text itself is a separate human message the node
adds; per the [M4-01b] convention, no prompt text lives in the node function.
"""

from infrastructure.graph.prompts.scope_preamble import with_scope_preamble

# code -> one-line description, taken from data/README.md so the model
# classifies against the same definitions the corpus documentation uses. Keys
# match schemas.ProductLineCode.
PRODUCT_LINE_GUIDE: dict[str, str] = {
    "CASCO": (
        "damage to the insured's own vehicle from an external event -- collision, "
        "fire, theft, weather, vandalism, total loss"
    ),
    "RCF-A": (
        "optional third-party liability -- damage the insured causes to other "
        "people or their property"
    ),
    "ASSIST": (
        "the claimant needs a service after a breakdown or incident -- towing, "
        "locksmith, a stand-in car, a taxi, a hotel, glass repair -- rather than "
        "indemnification for vehicle damage"
    ),
    "GAR.EST": (
        "a mechanical or electrical part failed on its own, with no crash or "
        "external cause, and the claimant wants it repaired under warranty"
    ),
    "CARTA VERDE": ("an event during cross-border travel within Mercosur countries"),
}

# tag -> the narrow condition under which intake should flag it. Keys match
# schemas.MissingInfoTag. Deliberately conservative: before retrieval the node
# cannot know which clause gates a given claim, so a tag is only justified when
# the gap plainly blocks an analyst -- most narratives get an empty list. The
# residual recall gap on the amount / cause / location tags is measured in
# docs/INTAKE_EXTRACTION.md and is caught later by retrieval ([M4-04]) or the
# clarification loop ([M4-03]).
MISSING_INFO_GUIDE: dict[str, str] = {
    "data_evento_vigencia": (
        "the narrative gives no time reference at all -- not even a vague one "
        "like 'a few weeks ago'"
    ),
    "ambito_geografico": (
        "the narrative does not say where the event happened -- no place, and no "
        "indication of whether it was local, on a trip, or abroad"
    ),
    "uso_do_veiculo": (
        "the narrative hints the vehicle may have been in paid transport, "
        "app-based or cargo use but does not confirm it; with no such hint, "
        "assume private use and do not tag"
    ),
    "valor_franquia_limite": (
        "the narrative gives no sense at all of how severe the damage or loss is "
        "-- no amount and no words like 'minor', 'a scratch', 'serious', "
        "'wrecked' or 'total loss' -- so it cannot be weighed against a "
        "deductible or a sub-limit"
    ),
    "tipo_evento_condicao": (
        "the narrative itself says it is unsure what caused the damage, or names "
        "two clearly different possible causes (an impact vs vandalism vs a "
        "falling object; a mechanical fault vs an external hit)"
    ),
}


def build_intake_prompt() -> str:
    """Return the intake node's system prompt: scope preamble + extraction task."""
    product_lines = "\n".join(
        f"- {code}: {description}" for code, description in PRODUCT_LINE_GUIDE.items()
    )
    missing_tags = "\n".join(
        f"- {tag}: {description}" for tag, description in MISSING_INFO_GUIDE.items()
    )
    body = f"""\
You are given one insurance claim narrative, written in informal Brazilian \
Portuguese by the person making the claim. Read it and extract a structured \
record. Do not answer or assess the claim -- only read it.

Rules:
- Fill a field only from what the narrative actually says. Leave it null \
rather than infer or invent a value.
- The SUSEP process, the insurer and the policy especially: if the narrative \
does not name them, leave susep_process null and do not pick a document.
- estimated_amount is a number in BRL only when the narrative states an amount.
- event_date may be an absolute date or the relative phrase the narrative uses; \
it is null only when there is no time reference at all.
- description is one or two factual sentences.

Classify the event against exactly one of the five registered product lines in \
the corpus, or null when it fits none or is too vague to place:
{product_lines}
Guidance: damage to the insured's own vehicle from an external cause is CASCO \
even when the person is writing about a different insurer or product. Choose \
ASSIST when the narrative reads as a request for help or a service rather than \
a claim for vehicle damage. Choose GAR.EST only for a self-caused part failure \
with no crash or external cause.

missing_information is usually empty. Add a tag only when its condition below \
is met -- a fact that would stop an analyst from even starting, not merely one \
that more precision would improve. When in doubt, leave it out.
{missing_tags}
"""
    return with_scope_preamble(body)
