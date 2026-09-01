"""Structured-output schemas for graph nodes -- [M4-01b].

One frozen Pydantic ``<Node>Output`` per LLM node -- the exact shape passed to
``llm.with_structured_output(...)``. Kept separate from the state sub-models in
``state.py``: the node maps its output schema onto the state model (for
example, the model returns clause-id strings and the node hydrates ``Citation``
objects with the retrieved clauses' provenance).

``IntakeOutput`` ([M4-02]) is the first: it maps onto ``ExtractedEntities`` and
routes ``missing_information`` to its own ``ClaimState`` channel.
``ClarificationOutput`` ([M4-03]) is the second: the clarification node maps
each ``ClarificationQuestionItem`` onto a ``state.ClarificationQuestion``.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# The five product lines in the corpus. Canonical source is the ``product_line``
# column of ``data/policies/manifest.csv`` -- there is no enum in the codebase.
# The intake node ([M4-02]) classifies each claim's event against this closed
# set.
ProductLineCode = Literal["CASCO", "RCF-A", "ASSIST", "GAR.EST", "CARTA VERDE"]

# The load-bearing facts an intake pass can find missing from a claim narrative,
# before any retrieval runs. The string values match
# ``infrastructure.evaluation.synthetic_claims_schema.MissingFactType`` verbatim
# so the [M4-10] eval can score intake's output against each synthetic claim's
# labelled ``missing_fact_type`` -- redeclared here rather than imported, to keep
# the graph package independent of the evaluation package. A unit test
# (``tests/unit/infrastructure/graph/test_intake.py``) guards the match.
MissingInfoTag = Literal[
    "ambito_geografico",
    "uso_do_veiculo",
    "data_evento_vigencia",
    "valor_franquia_limite",
    "tipo_evento_condicao",
]


class IntakeOutput(BaseModel):
    """The intake node's ([M4-02]) structured read of the raw claim narrative.

    The exact shape passed to ``fast_model.with_structured_output(...)``. The
    node copies the entity fields onto ``state.ExtractedEntities`` and sends
    ``missing_information`` to its own ``ClaimState`` channel. Every entity
    field is optional: the model leaves it ``None`` when the narrative does not
    state the fact, rather than inventing a value.
    """

    model_config = ConfigDict(frozen=True)

    event_type: str | None = Field(
        default=None,
        description=(
            "The kind of event described -- e.g. collision, theft, fire, glass "
            "breakage, mechanical failure, a roadside assistance request. Null "
            "when the narrative does not make the event type clear."
        ),
    )
    event_date: str | None = Field(
        default=None,
        description=(
            "When the event happened, as stated -- an absolute date or the "
            "relative phrase the narrative uses ('faz umas duas semanas'). "
            "Null only when there is no time reference at all. Do not infer a "
            "date."
        ),
    )
    description: str | None = Field(
        default=None,
        description=(
            "A concise, neutral summary of what happened, including where it "
            "happened when the narrative says so. Null only when the narrative "
            "carries no detail."
        ),
    )
    estimated_amount: float | None = Field(
        default=None,
        description=(
            "The estimated loss amount in BRL, as a number, only when the "
            "narrative states one. Null otherwise -- do not estimate."
        ),
    )
    vehicle_info: str | None = Field(
        default=None,
        description=(
            "What the narrative says about the vehicle: make, model, year, how "
            "it was being used (private, app-based transport, cargo), or its "
            "condition. Null when the narrative says nothing about the vehicle."
        ),
    )
    susep_process: str | None = Field(
        default=None,
        description=(
            "The SUSEP process number (format NNNNN.NNNNNN/NNNN-NN) only when "
            "the narrative explicitly states one. Null otherwise -- never guess "
            "a process, an insurer, or a policy."
        ),
    )
    product_line: ProductLineCode | None = Field(
        default=None,
        description=(
            "Which registered product line the described event belongs to, "
            "read from the event itself rather than copied from the text. Null "
            "only when the event fits none of the five lines or is too vague to "
            "place. Damage to the insured's own vehicle is CASCO even when the "
            "person is writing about a different product."
        ),
    )
    missing_information: list[MissingInfoTag] = Field(
        default_factory=list,
        description=(
            "The load-bearing facts needed to judge whether the event is "
            "consistent with a registered product's conditions and which the "
            "narrative does not provide. Add a tag only when the fact is both "
            "absent and material; an empty list means the narrative is "
            "complete enough to proceed."
        ),
    )


class ClarificationQuestionItem(BaseModel):
    """One (gap, question) pair the clarification node ([M4-03]) produces.

    ``field`` is the ``missing_information`` tag the question is meant to
    close; ``question`` is the concrete thing to ask the claimant. The node
    maps this onto ``state.ClarificationQuestion`` and guarantees exactly one
    item per input gap -- filling any tag the model omits from a per-tag
    fallback template.
    """

    model_config = ConfigDict(frozen=True)

    field: MissingInfoTag
    question: str = Field(
        description=(
            "One specific question, in informal Brazilian Portuguese, whose "
            "answer would close exactly this gap and no other. Refer to what "
            "the claimant already said. Never a generic 'send more details'."
        )
    )


class ClarificationOutput(BaseModel):
    """The clarification node's ([M4-03]) structured set of questions.

    The exact shape passed to ``fast_model.with_structured_output(...)`` -- one
    ``ClarificationQuestionItem`` per gap in the current ``missing_information``
    list.
    """

    model_config = ConfigDict(frozen=True)

    questions: list[ClarificationQuestionItem]
