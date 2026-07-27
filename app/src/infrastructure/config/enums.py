"""This module contains enums used in the application settings."""

from enum import StrEnum


class LlmProvider(StrEnum):
    """Supported LLM providers for the client factory."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
