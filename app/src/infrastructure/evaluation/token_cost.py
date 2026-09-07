"""Per-call token cost for the assessment graph -- [M5-10].

``LlmSettings`` carries the per-1M-token list prices for the fast and reasoning
models (``LLM_{FAST,REASONING}_{INPUT,OUTPUT}_COST_PER_1M_TOKENS_USD``). [M5-07]
already turned them into Langfuse model definitions
(``infrastructure.observability.tracing.model_price_definitions``); [M5-10] needs
the same arithmetic to price the token counts it measures per node.

Pure functions, no I/O, and **no import of ``infrastructure.observability``** --
that would drag ``langfuse`` into the evaluation package for two divisions.
``tests/unit/infrastructure/evaluation/test_token_cost.py`` pins the numbers and
a drift guard keeps ``per_token_rates`` and ``model_price_definitions`` agreeing.

**Reasoning tokens.** A reasoning-model response reports its thinking under
``usage_metadata["output_token_details"]["reasoning"]``; providers bill those at
the completion (output) rate, which is why [M5-07]'s Langfuse pricing tier maps
``output_reasoning`` to the output price. LangChain's ``output_tokens`` is the
*total* completion count and already includes the reasoning tokens, so
``CallCost.total_usd`` is ``input + output`` and ``reasoning_usd`` is a memo: the
share of ``output_usd`` spent on thinking, never added on top.
"""

from __future__ import annotations

from dataclasses import dataclass

from infrastructure.config.settings import LlmSettings

_TOKENS_PER_MILLION = 1_000_000

# The tier a call ran on when the reported model name is not one of the two
# configured ids (a provider alias, a missing name) -- used to pick a rate.
Tier = str  # "fast" | "reasoning"


@dataclass(frozen=True)
class ModelRate:
    """USD per single token, input and output, for one model."""

    input_per_token: float
    output_per_token: float


@dataclass(frozen=True)
class CallCost:
    """The cost of one LLM call, split for the per-node table.

    ``reasoning_usd`` is the portion of ``output_usd`` attributable to reasoning
    tokens -- it is **not** a third addend. ``total_usd`` is ``input + output``.
    """

    input_usd: float
    output_usd: float
    reasoning_usd: float

    @property
    def total_usd(self) -> float:
        """Input plus the full output cost (reasoning tokens already in output)."""
        return self.input_usd + self.output_usd


def per_token_rates(llm: LlmSettings) -> dict[str, ModelRate]:
    """``{model id: ModelRate}`` for the fast and reasoning models.

    Keyed by ``llm.llm_model_fast`` / ``llm.llm_model_reasoning`` so a call's
    reported model name looks its rate up directly. If both settings name the
    same model the reasoning price wins (last write) -- a degenerate config the
    drift-guard test does not exercise.
    """
    return {
        llm.llm_model_fast: ModelRate(
            input_per_token=llm.llm_fast_input_cost_per_1m_tokens_usd
            / _TOKENS_PER_MILLION,
            output_per_token=llm.llm_fast_output_cost_per_1m_tokens_usd
            / _TOKENS_PER_MILLION,
        ),
        llm.llm_model_reasoning: ModelRate(
            input_per_token=llm.llm_reasoning_input_cost_per_1m_tokens_usd
            / _TOKENS_PER_MILLION,
            output_per_token=llm.llm_reasoning_output_cost_per_1m_tokens_usd
            / _TOKENS_PER_MILLION,
        ),
    }


def price_call(
    *,
    model: str | None,
    tier: Tier | None,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int,
    rates: dict[str, ModelRate],
    fast_model: str,
    reasoning_model: str,
) -> CallCost:
    """Price one call. ``output_tokens`` is the total completion count.

    The rate comes from ``model`` when it is one of the two configured ids;
    otherwise from ``tier`` (``"reasoning"`` -> the reasoning rate, anything else
    -> the fast rate) -- a provider that echoes an alias instead of the exact id
    still lands on the right price.
    """
    rate = rates.get(model or "")
    if rate is None:
        rate = rates[reasoning_model if tier == "reasoning" else fast_model]
    capped_reasoning = min(max(reasoning_tokens, 0), max(output_tokens, 0))
    return CallCost(
        input_usd=input_tokens * rate.input_per_token,
        output_usd=output_tokens * rate.output_per_token,
        reasoning_usd=capped_reasoning * rate.output_per_token,
    )
