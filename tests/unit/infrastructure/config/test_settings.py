import pytest

from infrastructure.config.enums import LlmProvider
from infrastructure.config.settings import (
    DatabaseSettings,
    LlmSettings,
    ObservabilitySettings,
)

SECRET_VALUE = "super-secret-value"


@pytest.mark.unit
def test_database_password_is_redacted_from_repr_and_str() -> None:
    settings = DatabaseSettings(
        DATABASE_HOST="localhost",
        DATABASE_PORT=5432,
        DATABASE_USER="user",
        DATABASE_PASSWORD=SECRET_VALUE,
        DATABASE_NAME="db",
    )

    assert SECRET_VALUE not in repr(settings)
    assert SECRET_VALUE not in str(settings)
    assert settings.database_password is not None
    assert settings.database_password.get_secret_value() == SECRET_VALUE


@pytest.mark.unit
def test_llm_api_key_is_redacted_from_repr_and_str() -> None:
    settings = LlmSettings(
        LLM_PROVIDER=LlmProvider.OPENAI,
        LLM_API_KEY=SECRET_VALUE,
        LLM_MODEL_FAST="gpt-fast",
        LLM_MODEL_REASONING="gpt-reasoning",
        EMBEDDING_MODEL="embed-model",
        RERANKER_MODEL="rerank-model",
    )

    assert SECRET_VALUE not in repr(settings)
    assert SECRET_VALUE not in str(settings)
    assert settings.llm_api_key.get_secret_value() == SECRET_VALUE


@pytest.mark.unit
def test_langfuse_secret_key_is_redacted_from_repr_and_str() -> None:
    settings = ObservabilitySettings(LANGFUSE_SECRET_KEY=SECRET_VALUE)

    assert SECRET_VALUE not in repr(settings)
    assert SECRET_VALUE not in str(settings)
    assert settings.langfuse_secret_key.get_secret_value() == SECRET_VALUE


@pytest.mark.unit
def test_sqlalchemy_database_url_still_encodes_the_real_password() -> None:
    settings = DatabaseSettings(
        DATABASE_HOST="localhost",
        DATABASE_PORT=5432,
        DATABASE_USER="user",
        DATABASE_PASSWORD=SECRET_VALUE,
        DATABASE_NAME="db",
    )

    assert SECRET_VALUE in settings.sqlalchemy_database_url
