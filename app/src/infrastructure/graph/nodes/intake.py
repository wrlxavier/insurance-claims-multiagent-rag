"""The intake node: free-text claim narrative -> structured entities ([M4-02]).

One LLM call on the fast model, structured output into
``infrastructure.graph.schemas.IntakeOutput``, mapped onto
``state.ExtractedEntities`` plus the ``missing_information`` channel the
clarification loop ([M4-03]) consumes. The node never invents a value it did
not read: an absent fact stays null and, when it is load-bearing, becomes a
``missing_information`` tag.
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
from infrastructure.graph.prompts.intake import build_intake_prompt
from infrastructure.graph.schemas import IntakeOutput
from infrastructure.graph.state import (
    AuditEvent,
    ClaimState,
    ExtractedEntities,
    TokenUsage,
)

_NODE_INPUT_PREVIEW_CHARS = 200


def intake(state: ClaimState, runtime: Runtime[GraphContext]) -> dict[str, object]:
    """Extract structured entities from ``raw_claim_text`` and flag what is missing."""
    context = runtime.context
    raw_claim_text = state["raw_claim_text"]

    structured: Runnable[Any, Any] = context.fast_model.with_structured_output(
        IntakeOutput, include_raw=True
    )
    messages = [SystemMessage(build_intake_prompt()), HumanMessage(raw_claim_text)]
    result = _invoke_with_retry(structured, messages)
    parsed = cast(IntakeOutput, result["parsed"])

    entities = ExtractedEntities(
        event_type=parsed.event_type,
        event_date=parsed.event_date,
        description=parsed.description,
        estimated_amount=parsed.estimated_amount,
        vehicle_info=parsed.vehicle_info,
        susep_process=parsed.susep_process,
        product_line=parsed.product_line,
    )
    missing_information = list(dict.fromkeys(parsed.missing_information))

    audit_event = AuditEvent(
        node="intake",
        action="extract_entities",
        model=context.llm_settings.llm_model_fast,
        token_usage=_token_usage(result.get("raw")),
        node_input=raw_claim_text[:_NODE_INPUT_PREVIEW_CHARS],
    )
    return {
        "entities": entities,
        "missing_information": missing_information,
        "audit_trail": [audit_event],
    }


def _invoke_with_retry(
    chain: Runnable[Any, Any],
    messages: list[Any],
    *,
    max_attempts: int = DEFAULT_LLM_RETRY_MAX_ATTEMPTS,
    delay_seconds: float = DEFAULT_LLM_RETRY_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Invoke ``chain``, retrying a transient failure before re-raising.

    Mirrors ``scripts.validate_parsing_quality_sample.call_llm_with_retry``:
    structured extraction has no sane fallback value, so the final failure
    propagates rather than being swallowed the way clause classification's does.
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
