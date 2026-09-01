"""The clarification node: one specific question per missing fact ([M4-03]).

When intake leaves ``missing_information`` non-empty and the loop is still
under its cap, this node runs. One LLM call on the fast model, structured
output into ``schemas.ClarificationOutput``, mapped to a
``state.ClarificationQuestion`` per gap and appended to the accumulated
``clarification_questions`` channel. The node also increments
``clarification_rounds`` -- it is the event "a round happened".

Unlike intake, a question generator has a sane fallback: any gap the model
does not address, and every gap when the LLM call fails all its retries, is
filled from ``CLARIFICATION_FALLBACK_TEMPLATES``. The loop can therefore
always make progress, which is what lets [M4-03] guarantee termination.
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
from infrastructure.graph.context import GraphContext
from infrastructure.graph.prompts.clarification import (
    CLARIFICATION_FALLBACK_TEMPLATES,
    build_clarification_prompt,
)
from infrastructure.graph.schemas import ClarificationOutput
from infrastructure.graph.state import (
    AuditEvent,
    ClaimState,
    ClarificationQuestion,
    TokenUsage,
)


def clarification(
    state: ClaimState, runtime: Runtime[GraphContext]
) -> dict[str, object]:
    """Generate one concrete question per open ``missing_information`` gap."""
    context = runtime.context
    entities = state.get("entities")
    missing_information = list(state.get("missing_information") or [])
    prior_questions = list(state.get("clarification_questions") or [])
    rounds = state.get("clarification_rounds", 0)

    structured: Runnable[Any, Any] = context.fast_model.with_structured_output(
        ClarificationOutput, include_raw=True
    )
    messages = [
        SystemMessage(
            build_clarification_prompt(entities, missing_information, prior_questions)
        ),
        HumanMessage(state["raw_claim_text"]),
    ]

    raw_message: object = None
    by_tag: dict[str, str] = {}
    llm_failed = False
    try:
        result = _invoke_with_retry(structured, messages)
        parsed = cast(ClarificationOutput, result["parsed"])
        raw_message = result.get("raw")
        by_tag = {item.field: item.question for item in parsed.questions}
    except Exception:  # noqa: BLE001 - a generator has a template fallback
        llm_failed = True

    new_questions: list[ClarificationQuestion] = []
    templated: list[str] = []
    for tag in missing_information:
        question = by_tag.get(tag) or ""
        if not question.strip():
            question = CLARIFICATION_FALLBACK_TEMPLATES.get(
                tag, f"Poderia esclarecer melhor este ponto: {tag}?"
            )
            templated.append(tag)
        new_questions.append(ClarificationQuestion(field=tag, question=question))

    audit_event = AuditEvent(
        node="clarification",
        action="generate_questions",
        model=None if llm_failed else context.llm_settings.llm_model_fast,
        token_usage=_token_usage(raw_message),
        node_input=_describe_input(missing_information, llm_failed, templated),
    )
    return {
        "clarification_questions": prior_questions + new_questions,
        "clarification_rounds": rounds + 1,
        "audit_trail": [audit_event],
    }


def _describe_input(
    missing_information: list[str], llm_failed: bool, templated: list[str]
) -> str:
    parts = [f"gaps={sorted(missing_information)}"]
    if llm_failed:
        parts.append("llm_failed=all_gaps_templated")
    elif templated:
        parts.append(f"templated={sorted(templated)}")
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

    Same shape as ``nodes.intake._invoke_with_retry`` (kept module-local per
    the [M4-01b] node convention). The caller turns the final re-raise into a
    template fallback -- this node, unlike intake, can proceed without the model.
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
