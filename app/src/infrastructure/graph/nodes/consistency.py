"""The consistency node: arithmetic vs. judgement; signal, never decide ([M4-06]).

The node runs two legs and merges their output into one ``state.ConsistencyReport``:

1. **Deterministic** -- ``consistency_checks.run_deterministic_checks`` runs
   unconditionally, no model: an ``event_date`` typed in the future, an
   ``estimated_amount`` that is negative or absurd, a field intake contradicted
   in its own extraction, a ``product_line`` whose registered definition rules
   out the described event. Plain Python over parsed values -- "is this date
   after today", "is this number in band", "does this token collide with this
   line" -- because none of that needs a language model's permission.
2. **Semantic** -- one call on the *fast* model (``runtime.context.fast_model``),
   structured into ``schemas.ConsistencyOutput``, for the judgement the first
   leg cannot make: narrative coherence, description vs. stated event type,
   vagueness where a claimant would give detail.

Every signal carries ``source`` (``"deterministic"`` or ``"llm"``) so the two
stay measurable apart. The node returns **signals, never a verdict** -- it flags
for a human and nothing more.

**This is not a fraud detector.** Neither the data (no fraud labels) nor the
method (a few range checks plus an LLM reading for coherence) supports that
claim, and the project does not make it. See ``docs/SCOPE.md`` and
``docs/ARCHITECTURE.md``.

The semantic leg is best-effort: if the model call fails every retry the node
records that in its audit event and returns the deterministic signals alone --
it never raises (a signalling node has a sane partial result, unlike intake's
extraction). An empty report means "nothing flagged", which is the common case.

Two ``AuditEvent`` rows, one per leg, so the deterministic/LLM boundary is
visible in the trail itself: ``action="deterministic_checks"`` (``model`` /
``token_usage`` / ``confidence`` all ``None`` -- the absence is the record that
no model ran) and ``action="semantic_judgement"``.
"""

import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.runtime import Runtime

from application.use_cases.llm_retry_defaults import (
    DEFAULT_LLM_RETRY_DELAY_SECONDS,
    DEFAULT_LLM_RETRY_MAX_ATTEMPTS,
)
from infrastructure.graph.consistency_checks import (
    CHECK_AMOUNT_IMPLAUSIBLY_HIGH,
    CHECK_AMOUNT_IMPLAUSIBLY_LOW,
    CHECK_AMOUNT_NON_POSITIVE,
    CHECK_DATE_IN_FUTURE,
    CHECK_EVENT_DATE_FAR_PAST,
    CHECK_FIELD_CONTRADICTS_MISSING_TAG,
    CHECK_PRODUCT_LINE_CONTRADICTS_EVENT,
    run_deterministic_checks,
)
from infrastructure.graph.context import GraphContext
from infrastructure.graph.prompts.consistency import build_consistency_prompt
from infrastructure.graph.prompts.untrusted_content import wrap_untrusted
from infrastructure.graph.schemas import ConsistencyOutput
from infrastructure.graph.state import (
    AuditEvent,
    ClaimState,
    ConsistencyReport,
    ConsistencySignal,
    ExtractedEntities,
    TokenUsage,
)

_NODE_INPUT_PREVIEW_CHARS = 200

_DATE_CHECKS = {CHECK_DATE_IN_FUTURE, CHECK_EVENT_DATE_FAR_PAST}
_AMOUNT_CHECKS = {
    CHECK_AMOUNT_NON_POSITIVE,
    CHECK_AMOUNT_IMPLAUSIBLY_LOW,
    CHECK_AMOUNT_IMPLAUSIBLY_HIGH,
}


def consistency(state: ClaimState, runtime: Runtime[GraphContext]) -> dict[str, object]:
    """Run the deterministic and semantic consistency legs and merge the signals."""
    context = runtime.context
    entities = state.get("entities")
    missing_information = list(state.get("missing_information") or [])

    deterministic = run_deterministic_checks(
        entities, missing_information, now=datetime.now(UTC)
    )
    llm_signals, raw_message, llm_failed = _semantic_signals(
        context, state["raw_claim_text"], entities
    )

    report = ConsistencyReport(signals=[*deterministic, *llm_signals])
    deterministic_event = AuditEvent(
        node="consistency",
        action="deterministic_checks",
        node_input=_describe_deterministic(deterministic),
    )
    semantic_event = AuditEvent(
        node="consistency",
        action="semantic_judgement",
        model=None if llm_failed else context.llm_settings.llm_model_fast,
        token_usage=_token_usage(raw_message),
        node_input=f"signals={len(llm_signals)} llm_failed={llm_failed}",
    )
    return {
        "consistency": report,
        "audit_trail": [deterministic_event, semantic_event],
    }


def _semantic_signals(
    context: GraphContext,
    raw_claim_text: str,
    entities: ExtractedEntities | None,
) -> tuple[list[ConsistencySignal], object, bool]:
    """Call the fast model for the judgement leg; degrade to ``[]`` on failure.

    Returns ``(signals, raw_message, llm_failed)``. A transient failure is
    retried inside ``_invoke_with_retry``; if it still fails, or the response
    does not parse, the node proceeds with the deterministic signals alone
    rather than raising.
    """
    structured: Runnable[Any, Any] = context.fast_model.with_structured_output(
        ConsistencyOutput, include_raw=True
    )
    messages: list[Any] = [
        SystemMessage(build_consistency_prompt(entities)),
        HumanMessage(wrap_untrusted("claim_narrative", raw_claim_text)),
    ]
    try:
        result = _invoke_with_retry(structured, messages)
    except Exception:  # noqa: BLE001 - a signalling node degrades, it does not fail
        return [], None, True

    parsed = cast("ConsistencyOutput | None", result.get("parsed"))
    if parsed is None:
        return [], result.get("raw"), True
    signals = [
        ConsistencySignal(
            check=item.check,
            severity=item.severity,
            detail=item.detail,
            source="llm",
        )
        for item in parsed.signals
    ]
    return signals, result.get("raw"), False


def _describe_deterministic(signals: list[ConsistencySignal]) -> str:
    """Compact per-check tally for the deterministic leg's audit event."""
    tally = {
        "date": sum(s.check in _DATE_CHECKS for s in signals),
        "amount": sum(s.check in _AMOUNT_CHECKS for s in signals),
        "contradiction": sum(
            s.check == CHECK_FIELD_CONTRADICTS_MISSING_TAG for s in signals
        ),
        "product_line": sum(
            s.check == CHECK_PRODUCT_LINE_CONTRADICTS_EVENT for s in signals
        ),
    }
    parts = " ".join(f"{name}={count}" for name, count in tally.items())
    return f"signals={len(signals)} {parts}"[:_NODE_INPUT_PREVIEW_CHARS]


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
    [M4-01b] node convention). The caller turns the final re-raise into a
    deterministic-only result -- this node, like clarification, can proceed
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
