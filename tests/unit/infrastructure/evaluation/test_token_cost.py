"""Per-call token cost arithmetic [M5-10].

The part that can be wrong without a provider bill to check it against, so it is
pinned exactly here -- and a drift guard keeps it agreeing with the Langfuse
prices [M5-07] registers from the same settings.
"""

import pytest

from infrastructure.config.enums import LlmProvider
from infrastructure.config.settings import LlmSettings
from infrastructure.evaluation.token_cost import (
    ModelRate,
    per_token_rates,
    price_call,
)
from infrastructure.observability.tracing import model_price_definitions

_FAST = "fast-model-x"
_REASONING = "reasoning-model-y"


def _settings() -> LlmSettings:
    return LlmSettings(
        LLM_PROVIDER=LlmProvider.OPENAI,
        LLM_API_KEY="test-key",
        LLM_MODEL_FAST=_FAST,
        LLM_MODEL_REASONING=_REASONING,
        EMBEDDING_MODEL="embed-model",
        RERANKER_MODEL="rerank-model",
        LLM_FAST_INPUT_COST_PER_1M_TOKENS_USD=0.14,
        LLM_FAST_OUTPUT_COST_PER_1M_TOKENS_USD=0.28,
        LLM_REASONING_INPUT_COST_PER_1M_TOKENS_USD=1.1154,
        LLM_REASONING_OUTPUT_COST_PER_1M_TOKENS_USD=3.3462,
        _env_file=None,
    )


@pytest.mark.unit
def test_per_token_rates_divides_the_list_prices_by_a_million() -> None:
    rates = per_token_rates(_settings())
    assert rates[_FAST] == ModelRate(0.14 / 1e6, 0.28 / 1e6)
    assert rates[_REASONING] == ModelRate(1.1154 / 1e6, 3.3462 / 1e6)


@pytest.mark.unit
def test_price_call_pins_a_fast_model_call() -> None:
    rates = per_token_rates(_settings())
    cost = price_call(
        model=_FAST,
        tier="fast",
        input_tokens=1_000_000,
        output_tokens=500_000,
        reasoning_tokens=0,
        rates=rates,
        fast_model=_FAST,
        reasoning_model=_REASONING,
    )
    assert cost.input_usd == pytest.approx(0.14)
    assert cost.output_usd == pytest.approx(0.14)  # 0.5M * 0.28/1M
    assert cost.reasoning_usd == 0.0
    assert cost.total_usd == pytest.approx(0.28)


@pytest.mark.unit
def test_reasoning_tokens_are_a_memo_not_a_third_addend() -> None:
    rates = per_token_rates(_settings())
    # output_tokens is the full completion count; 3_725 of it was thinking.
    cost = price_call(
        model=_REASONING,
        tier="reasoning",
        input_tokens=3_000,
        output_tokens=4_000,
        reasoning_tokens=3_725,
        rates=rates,
        fast_model=_FAST,
        reasoning_model=_REASONING,
    )
    assert cost.total_usd == pytest.approx(3_000 * 1.1154 / 1e6 + 4_000 * 3.3462 / 1e6)
    # reasoning_usd is a slice of output_usd, never added on top
    assert cost.reasoning_usd == pytest.approx(3_725 * 3.3462 / 1e6)
    assert cost.reasoning_usd < cost.output_usd
    assert cost.total_usd == pytest.approx(cost.input_usd + cost.output_usd)


@pytest.mark.unit
def test_reasoning_tokens_are_capped_at_the_output_count() -> None:
    # A provider that reports reasoning tokens separately from (not within)
    # output_tokens must not push reasoning_usd above output_usd.
    rates = per_token_rates(_settings())
    cost = price_call(
        model=_REASONING,
        tier="reasoning",
        input_tokens=0,
        output_tokens=100,
        reasoning_tokens=5_000,
        rates=rates,
        fast_model=_FAST,
        reasoning_model=_REASONING,
    )
    assert cost.reasoning_usd == cost.output_usd


@pytest.mark.unit
def test_unknown_model_falls_back_to_the_tier() -> None:
    rates = per_token_rates(_settings())
    reasoning_alias = price_call(
        model="deepseek/some-provider-alias",
        tier="reasoning",
        input_tokens=1_000_000,
        output_tokens=0,
        reasoning_tokens=0,
        rates=rates,
        fast_model=_FAST,
        reasoning_model=_REASONING,
    )
    assert reasoning_alias.input_usd == pytest.approx(1.1154)

    fast_default = price_call(
        model=None,
        tier=None,
        input_tokens=1_000_000,
        output_tokens=0,
        reasoning_tokens=0,
        rates=rates,
        fast_model=_FAST,
        reasoning_model=_REASONING,
    )
    assert fast_default.input_usd == pytest.approx(0.14)


@pytest.mark.unit
def test_rates_do_not_drift_from_the_langfuse_pricing_definitions() -> None:
    # [M5-07]'s tracing.model_price_definitions and this module's per_token_rates
    # both turn the same LlmSettings fields into per-token prices. One source of
    # truth; this asserts the two consumers stay numerically identical.
    settings = _settings()
    rates = per_token_rates(settings)
    for name, input_per_token, output_per_token in model_price_definitions(settings):
        assert rates[name] == ModelRate(input_per_token, output_per_token)
