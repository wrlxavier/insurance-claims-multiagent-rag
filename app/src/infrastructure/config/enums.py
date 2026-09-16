"""This module contains enums used in the application settings."""

from enum import StrEnum


class LlmProvider(StrEnum):
    """Client families ``llm_client_factory.build_chat_model`` can build.

    Only ``OPENAI`` -- any OpenAI-compatible endpoint, including a gateway
    like OpenRouter -- is implemented; ``langchain-openai`` is the only
    Langchain provider integration this project depends on (see
    ``pyproject.toml``). That already covers picking any model from any
    provider (OpenAI, DeepSeek, Google, Anthropic, ...) that an
    OpenAI-compatible gateway exposes -- swapping models is a `.env` change
    (`LLM_MODEL_FAST`/`LLM_MODEL_REASONING`/`LLM_BASE_URL`), not a code
    change. A value here with no branch in the factory would raise
    ``NotImplementedError`` at call time, so this enum only lists what is
    actually wired up.
    """

    OPENAI = "openai"
