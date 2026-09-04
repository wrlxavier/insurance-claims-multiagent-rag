"""The recommendation node: consolidate the assessment for a human ([M4-08]).

The single terminal node. Every path the graph can take reaches it -- the
sufficient-context path after the compatibility ([M4-05]) and consistency
([M4-06]) branches converge, the insufficient-retrieval path
(``context_sufficient is False``), and the exhausted clarification loop
(``clarification_exhausted``) -- and it emits one ``state.Recommendation`` for a
reviewer.

**It consolidates; it does not re-decide.** The load-bearing fields are computed
here, in Python, from upstream state:

- ``recommended_action`` -- a fixed template per *posture* (the effective verdict
  plus why we are here). Framed as "route to a human reviewer" / "ask the
  claimant", never a real-world coverage outcome.
- ``citations`` -- a deduplicated copy of ``compatibility.citations`` (which
  [M4-05] already guarantees is a subset of what retrieval returned). The node
  never constructs a ``Citation`` and the model's output schema has no citation
  field, so a citation no upstream node produced cannot appear -- this is
  structural, not a prompt instruction.
- ``consistency_flags`` -- ``consistency.signals`` verbatim, kept separate from
  the compatibility verdict. Attention points, not part of the decision.
- ``confidence`` -- derived from ``compatibility.confidence`` and clamped: an
  ``insufficient_information`` effective verdict is capped at
  ``_INSUFFICIENT_CONFIDENCE_CEILING``, and an unresolved ``attention`` flag caps
  it at ``_ATTENTION_FLAG_CONFIDENCE_CEILING``. An abstaining upstream verdict
  cannot become a confident recommendation.

Only ``justification`` -- the prose paragraph a reviewer scans -- comes from the
model, and only on the path where a real compatibility assessment exists. The
fast model is used (like the consistency node's semantic leg): this is a summary,
not the legal reasoning, which the compatibility node already did. On the
claimant-gaps and retrieval-miss paths, or if the model call fails every retry,
a deterministic template stands in -- the node never raises (it is the terminal
consolidation point).
"""

import time
from collections.abc import Callable
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.runtime import Runtime

from application.use_cases.llm_retry_defaults import (
    DEFAULT_LLM_RETRY_DELAY_SECONDS,
    DEFAULT_LLM_RETRY_MAX_ATTEMPTS,
)
from domain.verdict import Verdict
from infrastructure.graph.context import GraphContext
from infrastructure.graph.prompts.recommendation import build_recommendation_prompt
from infrastructure.graph.prompts.untrusted_content import wrap_untrusted
from infrastructure.graph.schemas import RecommendationOutput
from infrastructure.graph.state import (
    AuditEvent,
    Citation,
    ClaimState,
    CompatibilityAssessment,
    ConsistencySignal,
    ExtractedEntities,
    Recommendation,
    TokenUsage,
)

# Confidence ceilings -- product behaviour, defined and tested in code (like
# ``compatibility.MAX_GROUNDING_ATTEMPTS`` / ``build.MAX_CLARIFICATION_ROUNDS``),
# not deployment knobs.
#
# An effective verdict of insufficient_information (compatibility abstained,
# retrieval missed, or the claimant never supplied enough) can never be a
# confident recommendation.
_INSUFFICIENT_CONFIDENCE_CEILING = 0.3
# An unresolved `attention` consistency flag caps how confident the
# recommendation may be -- the reviewer still has something to check.
_ATTENTION_FLAG_CONFIDENCE_CEILING = 0.7

_NODE_INPUT_PREVIEW_CHARS = 200

# The "posture" is the effective verdict plus the reason we reached this node.
_POSTURE_CLAIMANT_GAPS = "claimant_gaps"
_POSTURE_RETRIEVAL_MISS = "retrieval_miss"
_POSTURE_COMPATIBLE = "compatible"
_POSTURE_INCOMPATIBLE = "incompatible"
_POSTURE_INCONCLUSIVE = "inconclusive"
_POSTURE_NO_ASSESSMENT = "no_assessment"

_ASSESSED_POSTURES = frozenset(
    {_POSTURE_COMPATIBLE, _POSTURE_INCOMPATIBLE, _POSTURE_INCONCLUSIVE}
)

_VERDICT_POSTURE = {
    Verdict.COMPATIBLE: _POSTURE_COMPATIBLE,
    Verdict.INCOMPATIBLE: _POSTURE_INCOMPATIBLE,
    Verdict.INSUFFICIENT_INFORMATION: _POSTURE_INCONCLUSIVE,
}

_RECOMMENDED_ACTION: dict[str, str] = {
    _POSTURE_CLAIMANT_GAPS: (
        "Solicitar ao segurado as informações pendentes antes de retomar a "
        "análise: {gaps}."
    ),
    _POSTURE_RETRIEVAL_MISS: (
        "Encaminhar para revisão manual de cláusulas: a recuperação automática "
        "não localizou condições que resolvam a questão."
    ),
    _POSTURE_COMPATIBLE: (
        "Encaminhar para revisão humana: nenhuma cláusula de exclusão recuperada "
        "contradiz o evento; confirmar com as cláusulas citadas."
    ),
    _POSTURE_INCOMPATIBLE: (
        "Encaminhar para revisão humana com prioridade: uma cláusula recuperada "
        "indica incompatibilidade entre o evento e as condições do produto "
        "registrado."
    ),
    _POSTURE_INCONCLUSIVE: (
        "Encaminhar para revisão humana: as cláusulas recuperadas não resolvem a "
        "questão; avaliação inconclusiva."
    ),
    _POSTURE_NO_ASSESSMENT: (
        "Encaminhar para revisão humana: o fluxo automático terminou sem uma "
        "avaliação de compatibilidade."
    ),
}

_POSTURE_SENTENCE: dict[str, str] = {
    _POSTURE_CLAIMANT_GAPS: (
        "O segurado não forneceu informações suficientes para avaliar o evento."
    ),
    _POSTURE_RETRIEVAL_MISS: (
        "A recuperação automática não trouxe cláusulas que resolvam a questão."
    ),
    _POSTURE_COMPATIBLE: (
        "A avaliação de compatibilidade indicou que o evento é compatível com as "
        "condições do produto registrado."
    ),
    _POSTURE_INCOMPATIBLE: (
        "A avaliação de compatibilidade indicou que o evento é incompatível com "
        "as condições do produto registrado."
    ),
    _POSTURE_INCONCLUSIVE: (
        "A avaliação de compatibilidade foi inconclusiva diante das cláusulas "
        "recuperadas."
    ),
    _POSTURE_NO_ASSESSMENT: (
        "Nenhuma avaliação de compatibilidade foi produzida para este sinistro."
    ),
}


def recommendation(
    state: ClaimState, runtime: Runtime[GraphContext]
) -> dict[str, object]:
    """Consolidate the upstream findings into one ``state.Recommendation``."""
    context = runtime.context
    compatibility = state.get("compatibility")
    consistency = state.get("consistency")
    entities = state.get("entities")
    missing_information = sorted(state.get("missing_information") or [])
    clarification_exhausted = bool(state.get("clarification_exhausted"))
    context_sufficient = state.get("context_sufficient")

    effective_verdict, posture = _posture(
        clarification_exhausted, context_sufficient, compatibility
    )
    citations = _aggregate_citations(compatibility)
    flags = list(consistency.signals) if consistency is not None else []
    confidence = _confidence(effective_verdict, compatibility, flags)
    action = _recommended_action(posture, missing_information)

    if compatibility is not None and posture in _ASSESSED_POSTURES:
        drafted, raw_message, llm_failed = _draft_justification(
            context, state["raw_claim_text"], entities, compatibility, flags, citations
        )
        justification = drafted or _fallback_justification(
            posture, compatibility, citations, flags, missing_information
        )
        model = None if llm_failed else context.llm_settings.llm_model_fast
    else:
        justification = _fallback_justification(
            posture, compatibility, citations, flags, missing_information
        )
        raw_message, llm_failed, model = None, False, None

    recommendation_obj = Recommendation(
        recommended_action=action,
        justification=justification,
        citations=citations,
        consistency_flags=flags,
        confidence=confidence,
    )
    audit_event = AuditEvent(
        node="recommendation",
        action="consolidate",
        model=model,
        token_usage=_token_usage(raw_message),
        confidence=confidence,
        node_input=(
            f"posture={posture} verdict={effective_verdict.value} "
            f"n_clauses={len(citations)} n_flags={len(flags)} "
            f"llm_failed={llm_failed}"
        )[:_NODE_INPUT_PREVIEW_CHARS],
    )
    return {"recommendation": recommendation_obj, "audit_trail": [audit_event]}


def _posture(
    clarification_exhausted: bool,
    context_sufficient: bool | None,
    compatibility: CompatibilityAssessment | None,
) -> tuple[Verdict, str]:
    """The effective verdict and why we are at this node.

    Precedence: the claimant never supplied enough (``clarification_exhausted``)
    outranks a retrieval miss, which outranks a real assessment. The first two
    are always ``insufficient_information``.
    """
    if clarification_exhausted:
        return Verdict.INSUFFICIENT_INFORMATION, _POSTURE_CLAIMANT_GAPS
    if context_sufficient is False:
        return Verdict.INSUFFICIENT_INFORMATION, _POSTURE_RETRIEVAL_MISS
    if compatibility is not None:
        return compatibility.verdict, _VERDICT_POSTURE[compatibility.verdict]
    return Verdict.INSUFFICIENT_INFORMATION, _POSTURE_NO_ASSESSMENT


def _aggregate_citations(
    compatibility: CompatibilityAssessment | None,
) -> list[Citation]:
    """Deduplicate ``compatibility.citations`` by clause id, order preserved.

    These are the clauses the verdict actually rests on -- [M4-05] hydrated them
    from what retrieval returned, so every one is already upstream-produced. The
    node builds no ``Citation`` of its own.
    """
    if compatibility is None:
        return []
    seen: set[str] = set()
    unique: list[Citation] = []
    for citation in compatibility.citations:
        if citation.clause_id not in seen:
            seen.add(citation.clause_id)
            unique.append(citation)
    return unique


def _confidence(
    effective_verdict: Verdict,
    compatibility: CompatibilityAssessment | None,
    flags: list[ConsistencySignal],
) -> float:
    """Derive the recommendation confidence, clamped so it cannot overstate."""
    base = compatibility.confidence if compatibility is not None else 0.0
    if effective_verdict is Verdict.INSUFFICIENT_INFORMATION:
        return min(base, _INSUFFICIENT_CONFIDENCE_CEILING)
    if any(flag.severity == "attention" for flag in flags):
        return min(base, _ATTENTION_FLAG_CONFIDENCE_CEILING)
    return base


def _recommended_action(posture: str, missing_information: list[str]) -> str:
    """The fixed action template for this posture."""
    template = _RECOMMENDED_ACTION[posture]
    if posture == _POSTURE_CLAIMANT_GAPS:
        return template.format(gaps=", ".join(missing_information) or "—")
    return template


def _draft_justification(
    context: GraphContext,
    raw_claim_text: str,
    entities: ExtractedEntities | None,
    compatibility: CompatibilityAssessment,
    flags: list[ConsistencySignal],
    citations: list[Citation],
) -> tuple[str, object, bool]:
    """Ask the fast model for the prose paragraph; degrade to ``""`` on failure.

    Returns ``(justification, raw_message, llm_failed)``. A transient failure is
    retried inside ``_invoke_with_retry``; if it still fails, or the response
    does not parse (or is blank), the caller falls back to a deterministic
    template rather than raising.
    """
    structured: Runnable[Any, Any] = context.fast_model.with_structured_output(
        RecommendationOutput, include_raw=True
    )
    messages: list[Any] = [
        SystemMessage(
            build_recommendation_prompt(entities, compatibility, flags, citations)
        ),
        HumanMessage(wrap_untrusted("claim_narrative", raw_claim_text)),
    ]
    try:
        result = _invoke_with_retry(structured, messages)
    except Exception:  # noqa: BLE001 - a terminal node degrades, it does not fail
        return "", None, True

    parsed = cast("RecommendationOutput | None", result.get("parsed"))
    if parsed is None or not parsed.justification.strip():
        return "", result.get("raw"), True
    return parsed.justification.strip(), result.get("raw"), False


def _fallback_justification(
    posture: str,
    compatibility: CompatibilityAssessment | None,
    citations: list[Citation],
    flags: list[ConsistencySignal],
    missing_information: list[str],
) -> str:
    """A deterministic justification: the posture, the clauses, the caveats."""
    parts = [_POSTURE_SENTENCE[posture]]
    if compatibility is not None and compatibility.reasoning.strip():
        parts.append(compatibility.reasoning.strip().replace("\n", " "))
    ids = ", ".join(citation.clause_id for citation in citations)
    parts.append(f"Cláusulas consideradas: {ids or '—'}.")
    if flags:
        parts.append(
            "Pontos de atenção: " + "; ".join(flag.detail for flag in flags) + "."
        )
    if posture == _POSTURE_CLAIMANT_GAPS and missing_information:
        parts.append("Informações pendentes: " + ", ".join(missing_information) + ".")
    return " ".join(parts)


def _invoke_with_retry(
    chain: Runnable[Any, Any],
    messages: list[Any],
    *,
    max_attempts: int = DEFAULT_LLM_RETRY_MAX_ATTEMPTS,
    delay_seconds: float = DEFAULT_LLM_RETRY_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Invoke ``chain``, retrying a transient failure before re-raising.

    Same shape as ``nodes.consistency._invoke_with_retry`` (kept module-local
    per the [M4-01b] node convention). The caller turns the final re-raise into
    a deterministic-only justification -- this node, like consistency, proceeds
    without the model.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return cast(dict[str, object], chain.invoke(messages))
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised below
            last_exc = exc
            if attempt < max_attempts:
                sleep(delay_seconds)
    assert last_exc is not None
    raise last_exc


def _token_usage(raw_message: object) -> TokenUsage | None:
    """Build ``TokenUsage`` from a raw AI message's ``usage_metadata``, if present."""
    usage = getattr(raw_message, "usage_metadata", None)
    if not isinstance(usage, dict):
        return None
    input_tokens = _as_int(usage.get("input_tokens"))
    output_tokens = _as_int(usage.get("output_tokens"))
    total_tokens = _as_int(usage.get("total_tokens")) or input_tokens + output_tokens
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _as_int(value: object) -> int:
    return value if isinstance(value, int) else 0
