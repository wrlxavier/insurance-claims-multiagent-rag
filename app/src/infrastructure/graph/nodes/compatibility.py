"""The compatibility assessment node: retrieved clauses -> a grounded verdict ([M4-05]).

One call on the **reasoning** model (``runtime.context.reasoning_model``, pinned
to its own OpenRouter route by ``LlmSettings.llm_reasoning_provider_order``),
structured output into ``schemas.CompatibilityOutput``, mapped onto
``state.CompatibilityAssessment``.

The rule that shapes this node: **every assertion in the reasoning must cite at
least one retrieved clause id.** The model returns its reasoning as a list of
``(statement, clause_ids)`` pairs; the node checks each pair against the clause
ids retrieval actually returned and, for a ``compatible`` / ``incompatible``
verdict, rejects and retries an output with an ungrounded assertion -- it is
never patched afterwards. After ``MAX_GROUNDING_ATTEMPTS`` still-ungrounded
returns the verdict degrades to ``insufficient_information``: an answer that
cannot be tied to a clause is not one this system may state as compatible or
incompatible. A transient (network) failure still propagates, as in
``nodes/intake.py``.

When retrieval returned nothing at all the node makes no model call -- there is
nothing to reason over, so the verdict is ``insufficient_information`` directly.
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
from infrastructure.graph.prompts.compatibility import build_compatibility_prompt
from infrastructure.graph.schemas import CompatibilityOutput, ReasonedAssertion
from infrastructure.graph.state import (
    AuditEvent,
    ClaimState,
    CompatibilityAssessment,
    TokenUsage,
)

# How many times the node will ask the model to re-ground its reasoning before
# giving up and returning insufficient_information. Product behaviour, defined
# and tested in code (like ``build.MAX_CLARIFICATION_ROUNDS`` /
# ``retrieval.RETRIEVAL_K``), not a deployment knob.
MAX_GROUNDING_ATTEMPTS = 3

# Confidence recorded when the node abstains without a real assessment -- no
# clauses to reason over, or the model never grounded its reasoning. The
# ``insufficient_information`` verdict is deliberate; the 0.0 says "no
# compatible/incompatible judgement was reached", which a reviewer should see.
_ABSTAIN_CONFIDENCE = 0.0

_NODE_INPUT_PREVIEW_CHARS = 200


def compatibility(
    state: ClaimState, runtime: Runtime[GraphContext]
) -> dict[str, object]:
    """Assess whether the claim's event is consistent with the retrieved clauses."""
    context = runtime.context
    citations = list(state.get("citations") or [])
    entities = state.get("entities")

    if not citations:
        return _abstain(
            "Nenhuma cláusula foi recuperada; não há base para avaliar o evento.",
            node_input="no_citations",
        )

    valid_ids = {citation.clause_id for citation in citations}
    structured: Runnable[Any, Any] = context.reasoning_model.with_structured_output(
        CompatibilityOutput, include_raw=True
    )
    messages: list[Any] = [
        SystemMessage(build_compatibility_prompt(entities, citations)),
        HumanMessage(state["raw_claim_text"]),
    ]

    result, parsed, errors, retries = _invoke_grounded(structured, messages, valid_ids)

    if errors or parsed is None:
        return _abstain(
            "A avaliação não pôde ser fundamentada nas cláusulas recuperadas após "
            f"{MAX_GROUNDING_ATTEMPTS} tentativas.",
            node_input=(
                f"ungrounded_after={MAX_GROUNDING_ATTEMPTS} errors={errors[:3]}"
            ),
            model=context.llm_settings.llm_model_reasoning,
            token_usage=_token_usage(result.get("raw")),
        )

    verdict = Verdict(parsed.verdict)
    cited_ids = {cid for a in parsed.assertions for cid in a.clause_ids}
    hydrated = [c for c in citations if c.clause_id in cited_ids]

    assessment = CompatibilityAssessment(
        verdict=verdict,
        reasoning=_render_reasoning(parsed.assertions),
        citations=hydrated,
        confidence=parsed.confidence,
    )
    audit_event = AuditEvent(
        node="compatibility",
        action="assess",
        model=context.llm_settings.llm_model_reasoning,
        token_usage=_token_usage(result.get("raw")),
        confidence=parsed.confidence,
        node_input=(
            f"verdict={verdict.value} n_clauses={len(citations)} "
            f"cited={len(hydrated)} grounding_retries={retries}"
        )[:_NODE_INPUT_PREVIEW_CHARS],
    )
    return {"compatibility": assessment, "audit_trail": [audit_event]}


def _abstain(
    reasoning: str,
    *,
    node_input: str,
    model: str | None = None,
    token_usage: TokenUsage | None = None,
) -> dict[str, object]:
    """Return an ``insufficient_information`` assessment plus its audit event."""
    assessment = CompatibilityAssessment(
        verdict=Verdict.INSUFFICIENT_INFORMATION,
        reasoning=reasoning,
        citations=[],
        confidence=_ABSTAIN_CONFIDENCE,
    )
    audit_event = AuditEvent(
        node="compatibility",
        action="assess",
        model=model,
        token_usage=token_usage,
        confidence=_ABSTAIN_CONFIDENCE,
        node_input=node_input[:_NODE_INPUT_PREVIEW_CHARS],
    )
    return {"compatibility": assessment, "audit_trail": [audit_event]}


def _invoke_grounded(
    chain: Runnable[Any, Any],
    messages: list[Any],
    valid_ids: set[str],
    *,
    max_attempts: int = MAX_GROUNDING_ATTEMPTS,
) -> tuple[dict[str, object], CompatibilityOutput | None, list[str], int]:
    """Call ``chain``; on an ungrounded parse, feed the errors back and retry.

    Returns ``(last_result, parsed, errors, retries)``. ``errors`` is empty on
    success; non-empty (with ``parsed`` the last attempt's output, or ``None``
    if it never parsed) when every attempt stayed ungrounded. Transient
    provider failures are handled inside ``_invoke_with_retry`` and propagate.
    """
    conversation = list(messages)
    last_result: dict[str, object] | None = None
    last_parsed: CompatibilityOutput | None = None
    last_errors: list[str] = []
    for attempt in range(1, max_attempts + 1):
        result = _invoke_with_retry(chain, conversation)
        parsed = cast("CompatibilityOutput | None", result.get("parsed"))
        errors = _grounding_errors(parsed, valid_ids)
        if not errors:
            return result, parsed, [], attempt - 1
        last_result, last_parsed, last_errors = result, parsed, errors
        if attempt < max_attempts:
            conversation = [*conversation, HumanMessage(_correction_message(errors))]
    assert last_result is not None  # the loop runs at least once
    return last_result, last_parsed, last_errors, max_attempts - 1


def _grounding_errors(
    parsed: CompatibilityOutput | None, valid_ids: set[str]
) -> list[str]:
    """List the ways ``parsed`` fails the "every assertion cites a clause" rule."""
    if parsed is None:
        return ["the response did not parse into the required schema"]
    errors: list[str] = []
    settles = parsed.verdict != "insufficient_information"
    if settles and not parsed.assertions:
        errors.append(
            f"verdict '{parsed.verdict}' needs at least one assertion, each "
            "citing a retrieved clause id"
        )
    for index, assertion in enumerate(parsed.assertions, start=1):
        unknown = [cid for cid in assertion.clause_ids if cid not in valid_ids]
        if settles and not assertion.clause_ids:
            errors.append(f"assertion {index} cites no clause id")
        if unknown:
            errors.append(
                f"assertion {index} cites clause id(s) that were not retrieved: "
                f"{unknown}"
            )
    return errors


def _correction_message(errors: list[str]) -> str:
    """The follow-up turn that tells the model exactly what to fix."""
    joined = "\n".join(f"- {error}" for error in errors)
    return (
        "Your previous answer is not acceptable:\n"
        f"{joined}\n"
        "Every assertion supporting a compatible or incompatible verdict must "
        "cite at least one clause id copied verbatim from the numbered list. "
        "If you cannot ground the assessment in the retrieved clauses, return "
        "verdict insufficient_information. Answer again."
    )


def _render_reasoning(assertions: list[ReasonedAssertion]) -> str:
    """Render the assertion list into the plain ``reasoning`` string state holds."""
    if not assertions:
        return "Nenhuma afirmação fundamentada foi produzida."
    lines = []
    for index, assertion in enumerate(assertions, start=1):
        cited = ", ".join(assertion.clause_ids) if assertion.clause_ids else "—"
        lines.append(f"{index}. {assertion.statement.strip()} [cláusulas: {cited}]")
    return "\n".join(lines)


def _invoke_with_retry(
    chain: Runnable[Any, Any],
    messages: list[Any],
    *,
    max_attempts: int = DEFAULT_LLM_RETRY_MAX_ATTEMPTS,
    delay_seconds: float = DEFAULT_LLM_RETRY_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Invoke ``chain``, retrying a transient failure before re-raising.

    Same shape as ``nodes.intake._invoke_with_retry`` (kept module-local per the
    [M4-01b] node convention). A grounding failure is not a transient failure --
    that is handled by ``_invoke_grounded``, which feeds the errors back.
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
