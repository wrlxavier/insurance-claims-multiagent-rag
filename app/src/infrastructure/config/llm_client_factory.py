"""Builds a Langchain ``BaseChatModel`` from ``LlmSettings``/``LlmProvider``.

Only OpenAI-compatible construction is wired up: ``langchain-openai`` is the
only Langchain provider integration installed (see ``pyproject.toml``). This
also covers OpenAI-compatible gateways such as OpenRouter, reached via
``ChatOpenAI(base_url=...)`` -- how this repo is actually configured today.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from infrastructure.config.enums import LlmProvider
from infrastructure.config.settings import LlmSettings


def build_chat_model(
    settings: LlmSettings, model: str, *, provider_order: list[str] | None = None
) -> BaseChatModel:
    """Build a chat model for ``model`` from the given provider settings.

    Args:
        settings: Provider credentials/base URL.
        model: The model id to request.
        provider_order: For an OpenRouter-style gateway, an optional
            preferred upstream routing order (OpenRouter's `provider.order`
            request field) -- pins the model to specific backend(s) instead
            of OpenRouter's default routing. Ignored by a direct OpenAI
            endpoint.

    Raises:
        NotImplementedError: if ``settings.llm_provider`` has no installed
            Langchain integration.
    """
    if settings.llm_provider == LlmProvider.OPENAI:
        kwargs: dict[str, object] = {
            "model": model,
            "api_key": settings.llm_api_key,
            "base_url": settings.llm_base_url,
        }
        if provider_order:
            kwargs["extra_body"] = {"provider": {"order": provider_order}}
        return ChatOpenAI(**kwargs)
    raise NotImplementedError(
        f"No BaseChatModel integration installed for provider "
        f"{settings.llm_provider!r} -- only langchain-openai is a "
        "dependency (see pyproject.toml)."
    )
