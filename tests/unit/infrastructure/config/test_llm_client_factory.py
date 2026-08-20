import pytest
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from infrastructure.config.enums import LlmProvider
from infrastructure.config.llm_client_factory import build_chat_model
from infrastructure.config.settings import LlmSettings

SECRET_VALUE = "super-secret-value"


def build_settings(provider: LlmProvider) -> LlmSettings:
    return LlmSettings(
        LLM_PROVIDER=provider,
        LLM_BASE_URL="https://openrouter.ai/api/v1",
        LLM_API_KEY=SECRET_VALUE,
        LLM_MODEL_FAST="openai/gpt-fast",
        LLM_MODEL_REASONING="openai/gpt-reasoning",
        EMBEDDING_MODEL="embed-model",
        RERANKER_MODEL="rerank-model",
    )


@pytest.mark.unit
def test_build_chat_model_openai_returns_chat_openai() -> None:
    settings = build_settings(LlmProvider.OPENAI)

    model = build_chat_model(settings, settings.llm_model_fast)

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "openai/gpt-fast"
    assert model.openai_api_base == "https://openrouter.ai/api/v1"
    assert isinstance(model.openai_api_key, SecretStr)
    assert model.openai_api_key.get_secret_value() == SECRET_VALUE


@pytest.mark.unit
def test_build_chat_model_openai_uses_the_given_model() -> None:
    settings = build_settings(LlmProvider.OPENAI)

    model = build_chat_model(settings, settings.llm_model_reasoning)

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "openai/gpt-reasoning"


@pytest.mark.unit
def test_build_chat_model_unsupported_provider_raises() -> None:
    settings = build_settings(LlmProvider.ANTHROPIC)

    with pytest.raises(NotImplementedError):
        build_chat_model(settings, settings.llm_model_fast)


@pytest.mark.unit
def test_build_chat_model_pins_provider_order_when_given() -> None:
    settings = build_settings(LlmProvider.OPENAI)

    model = build_chat_model(
        settings, settings.llm_model_fast, provider_order=["baidu"]
    )

    assert isinstance(model, ChatOpenAI)
    assert model.extra_body == {"provider": {"order": ["baidu"]}}


@pytest.mark.unit
def test_build_chat_model_no_provider_order_by_default() -> None:
    settings = build_settings(LlmProvider.OPENAI)

    model = build_chat_model(settings, settings.llm_model_fast)

    assert isinstance(model, ChatOpenAI)
    assert model.extra_body is None


@pytest.mark.unit
def test_build_chat_model_disables_fallbacks_when_given() -> None:
    settings = build_settings(LlmProvider.OPENAI)

    model = build_chat_model(
        settings,
        settings.llm_model_fast,
        provider_order=["baidu/fp8"],
        allow_fallbacks=False,
    )

    assert isinstance(model, ChatOpenAI)
    assert model.extra_body == {
        "provider": {"order": ["baidu/fp8"], "allow_fallbacks": False}
    }


@pytest.mark.unit
def test_build_chat_model_allow_fallbacks_without_provider_order() -> None:
    settings = build_settings(LlmProvider.OPENAI)

    model = build_chat_model(settings, settings.llm_model_fast, allow_fallbacks=True)

    assert isinstance(model, ChatOpenAI)
    assert model.extra_body == {"provider": {"allow_fallbacks": True}}
