"""The insufficient-context gate -- [M3-07].

Make "the corpus does not contain this" a first-class outcome. The policy corpus
is *condições gerais* of registered SUSEP products; by construction it holds no
policy-instance facts -- deductibles, insured amounts, premiums, policy periods,
endorsements (``docs/SCOPE.md``). A confident, fluent, well-cited answer to a
question whose answer was never in the corpus is the most expensive error this
project can make, and M4's assessment nodes trust whatever retrieval hands them.

This module is a pure decision function over **retrieval signals already
computed by the pipeline** -- it never calls a retriever, an embedder or the
reranker, and it imports nothing from ``infrastructure.evaluation`` (the same
split ``reranking_retriever`` / ``exclusion_co_retrieval`` keep). Whoever runs
retrieval -- ``scripts/eval_insufficient_context_gate.py`` for the [M3-07]
calibration now, M4's retrieval node ([M4-04]) later -- assembles a
[GateSignals] and calls [evaluate_gate].

The rule, calibrated on ``golden-set-v1`` (``docs/INSUFFICIENT_CONTEXT_GATE.md``):
abstain when nothing was retrieved, **or** the rank-1 reranked clause scores
below ``TOP_SCORE_ABSTAIN_THRESHOLD`` (a poor topical match), **or** the
question asks for a *specific* policy-instance value of a fact type
``docs/SCOPE.md`` says is absent and even the best retrieved clause scores below
the stricter ``INSTANCE_VALUE_TOP_SCORE_THRESHOLD`` (a clause that discusses the
fact without stating the number -- the domain's core failure mode).

The pinned thresholds and their ``config_fingerprint`` live in
``infrastructure.rag.insufficient_context_config``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from domain.clause_classification import ClauseType
from infrastructure.rag.insufficient_context_config import (
    INSTANCE_VALUE_TOP_SCORE_THRESHOLD,
    TOP_SCORE_ABSTAIN_THRESHOLD,
)

_CLOSEST_CLAUSE_LIMIT = 3


class MissingFactCategory(Enum):
    """The kind of policy-instance fact an abstained question was asking for.

    The five non-``OTHER`` values are exactly ``docs/SCOPE.md``'s "what the
    corpus is not" list, and match the ``FACT_TYPES`` the [M2-05]
    ``unanswerable`` questions were authored from
    (``scripts/unanswerable_question_selection.py``). This is a *label on an
    abstention*, for the structured result and for M4's downstream message.
    """

    DEDUCTIBLE = "deductible"
    INSURED_AMOUNT = "insured_amount"
    PREMIUM = "premium"
    POLICY_PERIOD = "policy_period"
    ENDORSEMENT = "endorsement"
    OTHER = "other"


class GateTrigger(Enum):
    """Which rule produced an abstention (empty string when the context sufficed)."""

    NONE = ""
    NO_CONTEXT = "no_context"
    LOW_RELEVANCE = "low_relevance"
    UNVERIFIED_INSTANCE_VALUE = "unverified_instance_value"


# Priority-ordered: the first category with a keyword hit in the question wins.
# Endorsement outranks the others because an endorsement question ("prêmio
# adicional cobrado no endosso ...", "limite ... após o endosso ...") is about
# a policy-instance change even when it also names a premium or a limit -- this
# matches the [M2-05] authoring of unanswerable-007..-010. Patterns are matched
# case-insensitively against the raw (accented) question text.
_CATEGORY_PATTERNS: tuple[tuple[MissingFactCategory, tuple[str, ...]], ...] = (
    (MissingFactCategory.ENDORSEMENT, (r"endosso", r"aditivo")),
    (
        MissingFactCategory.INSURED_AMOUNT,
        (
            r"import[âa]ncia segurada",
            r"limite m[áa]ximo de indeniza",
            r"limite m[áa]ximo de garantia",
            r"\blmi\b",
            r"capital segurado",
        ),
    ),
    (
        MissingFactCategory.POLICY_PERIOD,
        (r"vig[êe]ncia", r"vigor", r"prazo de cobertura"),
    ),
    (MissingFactCategory.PREMIUM, (r"pr[êe]mio",)),
    (MissingFactCategory.DEDUCTIBLE, (r"franquia",)),
)

_COMPILED_CATEGORY_PATTERNS: tuple[
    tuple[MissingFactCategory, tuple[re.Pattern[str], ...]], ...
] = tuple(
    (category, tuple(re.compile(p, re.IGNORECASE) for p in patterns))
    for category, patterns in _CATEGORY_PATTERNS
)

# A question that asks for a *concrete* value or date -- "qual é o valor exato /
# nominal / total / em reais", "montante", "em qual data", "número do endosso",
# a "contratada" / "fixada" figure -- as opposed to one that asks how a fact is
# structured / calculated / when it applies. The negative set below excludes the
# rule / manner / quantity / yes-no phrasings that a general-conditions document
# genuinely does answer. Calibrated against the ``golden-set-v1``
# ``unanswerable`` subset and the answerable questions that mention the same
# terms (docs/INSUFFICIENT_CONTEXT_GATE.md); a fitted classifier, not a
# held-out result -- golden-set-v2 must re-check it.
_INSTANCE_VALUE_RE = re.compile(
    r"\bvalor\s+(?:exato|nominal|total|estipulad|devido|em reais)"
    r"|\bmontante\b|\bem reais\b|\bexat[oa]s?\b"
    r"|em qual data|a partir de qual dia|n[úu]mero do endosso"
    r"|\bcontratad[oa]\b|\bfixad[oa]\b|estipulad[oa] (?:após|para|no)",
    re.IGNORECASE,
)
_STRUCTURAL_RE = re.compile(
    r"de que forma|\bcomo\b|\bquais\b|\bquant[oa]s\b|\bse pode|\bse poder"
    r"|\bonde\b|é sempre|\bpode ser\b|h[áa] cobertura|é obrigad"
    r"|qual é a porcentagem",
    re.IGNORECASE,
)

_CATEGORY_PHRASING: dict[MissingFactCategory, str] = {
    MissingFactCategory.DEDUCTIBLE: "a policy-instance deductible (franquia) amount",
    MissingFactCategory.INSURED_AMOUNT: (
        "a policy-instance insured amount / limite máximo de indenização"
    ),
    MissingFactCategory.PREMIUM: "a policy-instance premium (prêmio) amount",
    MissingFactCategory.POLICY_PERIOD: "policy-instance coverage dates (vigência)",
    MissingFactCategory.ENDORSEMENT: "a fact about a specific endorsement (endosso)",
    MissingFactCategory.OTHER: "a fact",
}


def classify_missing_information(question: str) -> MissingFactCategory:
    """Best-effort label for the policy-instance fact ``question`` asks for.

    Deterministic keyword match, priority order per ``_CATEGORY_PATTERNS``.
    Returns [MissingFactCategory.OTHER] when nothing matches -- still an honest
    "policy-instance fact absent by construction" outcome.
    """
    for category, patterns in _COMPILED_CATEGORY_PATTERNS:
        if any(pattern.search(question) for pattern in patterns):
            return category
    return MissingFactCategory.OTHER


def asks_for_instance_value(question: str) -> bool:
    """Whether ``question`` asks for a concrete value/date rather than a rule."""
    return bool(_INSTANCE_VALUE_RE.search(question)) and not bool(
        _STRUCTURAL_RE.search(question)
    )


def needs_verified_instance_value(question: str) -> bool:
    """Whether ``question`` wants a specific value of a ``docs/SCOPE.md``-absent fact.

    Those questions get the stricter ``INSTANCE_VALUE_TOP_SCORE_THRESHOLD``: the
    corpus can plausibly hold the *rule* behind a deductible / limit / premium /
    period / endorsement, so a mediocre-scoring clause is often a real answer to
    a rule question -- but never a source for the instance *number*.
    """
    return classify_missing_information(
        question
    ) is not MissingFactCategory.OTHER and asks_for_instance_value(question)


@dataclass(frozen=True)
class GateSignals:
    """The retrieval signals the gate decides on, for one question.

    Assembled by the caller from the pipeline it already ran -- hybrid RRF +
    cross-encoder rerank, filtered to the question's SUSEP process + CNPJ. The
    reranker (``Alibaba-NLP/gte-multilingual-reranker-base``) emits a sigmoid
    score in [0, 1]; ``reranked_scores`` is every scored candidate, sorted
    descending, and ``top_score`` is its first element (``0.0`` when nothing was
    retrieved).
    """

    top_score: float
    reranked_scores: tuple[float, ...]
    retrieved_clause_ids: tuple[str, ...]
    retrieved_clause_types: tuple[ClauseType, ...]
    k_requested: int
    n_returned: int


@dataclass(frozen=True)
class InsufficientContextResult:
    """The gate's verdict for one question.

    ``sufficient`` is the only field that matters when the context settles the
    question. On an abstention the rest name *what is missing* rather than
    leaving a bare refusal ([M3-07] DoD): which rule fired, the fact category, a
    plain-language explanation in the ``docs/SCOPE.md`` framing, and the closest
    clauses the retriever did surface (so a reviewer can see what it weighed).
    """

    sufficient: bool
    top_score: float
    threshold: float
    trigger: GateTrigger = GateTrigger.NONE
    missing_category: MissingFactCategory | None = None
    explanation: str = ""
    closest_clause_ids: tuple[str, ...] = field(default_factory=tuple)


def _build_explanation(
    category: MissingFactCategory,
    signals: GateSignals,
    trigger: GateTrigger,
    threshold: float,
) -> str:
    """One sentence naming the gap, in the ``docs/SCOPE.md`` framing."""
    phrasing = _CATEGORY_PHRASING[category]
    if trigger is GateTrigger.NO_CONTEXT:
        retrieval_note = (
            "no clause in the question's SUSEP process + CNPJ partition matched"
        )
    elif trigger is GateTrigger.UNVERIFIED_INSTANCE_VALUE:
        retrieval_note = (
            f"the question asks for a specific figure and the best retrieved "
            f"clause scored only {signals.top_score:.3f} (< {threshold:.3f}) -- a "
            "clause that discusses the fact, not one that states the contracted "
            "value"
        )
    else:
        retrieval_note = (
            f"the top retrieved clause scored {signals.top_score:.3f}, below the "
            f"{threshold:.3f} relevance floor"
        )
    return (
        "The retrieved context does not settle this question: it asks for "
        f"{phrasing}, which a registered product's general conditions (condições "
        "gerais) do not contain by construction -- that fact exists only in an "
        f"individual policy (docs/SCOPE.md). Here, {retrieval_note}."
    )


def evaluate_gate(
    question: str,
    signals: GateSignals,
    *,
    top_score_threshold: float = TOP_SCORE_ABSTAIN_THRESHOLD,
    instance_value_threshold: float = INSTANCE_VALUE_TOP_SCORE_THRESHOLD,
) -> InsufficientContextResult:
    """Decide whether the retrieved context settles ``question``.

    Abstains (``sufficient=False``) when the retriever returned nothing, when the
    rank-1 reranked score is below ``top_score_threshold``, or when the question
    asks for a specific policy-instance value
    ([needs_verified_instance_value]) and the rank-1 score is below the stricter
    ``instance_value_threshold``. Both defaults are the pinned
    ``infrastructure.rag.insufficient_context_config`` values;
    ``scripts/eval_insufficient_context_gate.py`` overrides them while sweeping.
    """
    if signals.n_returned == 0:
        trigger, threshold = GateTrigger.NO_CONTEXT, top_score_threshold
    elif signals.top_score < top_score_threshold:
        trigger, threshold = GateTrigger.LOW_RELEVANCE, top_score_threshold
    elif signals.top_score < instance_value_threshold and needs_verified_instance_value(
        question
    ):
        trigger, threshold = (
            GateTrigger.UNVERIFIED_INSTANCE_VALUE,
            instance_value_threshold,
        )
    else:
        return InsufficientContextResult(
            sufficient=True,
            top_score=signals.top_score,
            threshold=top_score_threshold,
        )

    category = classify_missing_information(question)
    return InsufficientContextResult(
        sufficient=False,
        top_score=signals.top_score,
        threshold=threshold,
        trigger=trigger,
        missing_category=category,
        explanation=_build_explanation(category, signals, trigger, threshold),
        closest_clause_ids=signals.retrieved_clause_ids[:_CLOSEST_CLAUSE_LIMIT],
    )
