from pathlib import Path

import pytest

from infrastructure.config.enums import LlmProvider
from infrastructure.config.settings import (
    DatabaseSettings,
    EmbeddingSettings,
    LlmSettings,
    ObservabilitySettings,
)
from infrastructure.rag.embedding_pipeline import EMBEDDING_BATCH_SIZE

SECRET_VALUE = "super-secret-value"
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _env_keys_in_order(path: Path) -> list[str]:
    return [
        line.split("=", 1)[0].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
    ]


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
def test_llm_model_vision_defaults_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_MODEL_VISION", raising=False)
    settings = LlmSettings(
        LLM_PROVIDER=LlmProvider.OPENAI,
        LLM_API_KEY=SECRET_VALUE,
        LLM_MODEL_FAST="gpt-fast",
        LLM_MODEL_REASONING="gpt-reasoning",
        EMBEDDING_MODEL="embed-model",
        RERANKER_MODEL="rerank-model",
        _env_file=None,
    )

    assert settings.llm_model_vision is None


@pytest.mark.unit
def test_llm_model_vision_can_be_set() -> None:
    settings = LlmSettings(
        LLM_PROVIDER=LlmProvider.OPENAI,
        LLM_API_KEY=SECRET_VALUE,
        LLM_MODEL_FAST="gpt-fast",
        LLM_MODEL_REASONING="gpt-reasoning",
        LLM_MODEL_VISION="google/gemini-3.7-flash",
        EMBEDDING_MODEL="embed-model",
        RERANKER_MODEL="rerank-model",
        _env_file=None,
    )

    assert settings.llm_model_vision == "google/gemini-3.7-flash"


def _llm_settings(**overrides: object) -> LlmSettings:
    fields: dict[str, object] = {
        "LLM_PROVIDER": LlmProvider.OPENAI,
        "LLM_API_KEY": SECRET_VALUE,
        "LLM_MODEL_FAST": "gpt-fast",
        "LLM_MODEL_REASONING": "gpt-reasoning",
        "EMBEDDING_MODEL": "embed-model",
        "RERANKER_MODEL": "rerank-model",
        "_env_file": None,
    }
    fields.update(overrides)
    return LlmSettings(**fields)  # type: ignore[arg-type]


@pytest.mark.unit
def test_llm_reasoning_provider_pin_defaults_to_streamlake_no_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_REASONING_PROVIDER_ORDER", raising=False)
    monkeypatch.delenv("LLM_REASONING_ALLOW_FALLBACKS", raising=False)

    settings = _llm_settings()

    assert settings.llm_reasoning_provider_order == ["streamlake"]
    assert settings.llm_reasoning_allow_fallbacks is False


@pytest.mark.unit
def test_llm_reasoning_provider_order_reads_the_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_REASONING_PROVIDER_ORDER", '["fireworks", "streamlake"]')

    assert _llm_settings().llm_reasoning_provider_order == ["fireworks", "streamlake"]


@pytest.mark.unit
def test_llm_vision_provider_order_defaults_to_the_zone_qualified_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_VISION_PROVIDER_ORDER", raising=False)

    assert _llm_settings().llm_vision_provider_order == ["google-vertex/global"]


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


@pytest.mark.unit
def test_embedding_batch_size_defaults_to_the_module_constant() -> None:
    # `.env`'s EMBEDDING_BATCH_SIZE is the operational knob [M1-09] moved out of
    # code; blank/absent falls back to the embedding_pipeline default.
    settings = EmbeddingSettings(_env_file=None)

    assert settings.embedding_batch_size == EMBEDDING_BATCH_SIZE == 64


@pytest.mark.unit
def test_embedding_batch_size_reads_the_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "16")

    assert EmbeddingSettings(_env_file=None).embedding_batch_size == 16


@pytest.mark.unit
@pytest.mark.skipif(
    not (_REPO_ROOT / ".env").exists(),
    reason="no .env in CI; parity is a local/pre-commit guard ([M1-09] rule)",
)
def test_env_and_env_example_have_identical_keys_in_order() -> None:
    # [M3-02] / [M1-09] DoD: `.env.example` in exact key parity with `.env`.
    assert _env_keys_in_order(_REPO_ROOT / ".env") == _env_keys_in_order(
        _REPO_ROOT / ".env.example"
    )
