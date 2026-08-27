"""This module contains the application settings loaded from environment variables."""

from functools import lru_cache
from typing import Self
from urllib.parse import quote_plus

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)

from .enums import LlmProvider


class ObservabilitySettings(BaseSettings):
    """Settings used by runtime observability and logging concerns."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    app_env: str = Field(alias="APP_ENV", default="development")
    log_level: str = Field(alias="LOG_LEVEL", default="INFO")
    proxy_headers_enabled: bool = Field(alias="PROXY_HEADERS_ENABLED", default=True)
    forwarded_allow_ips: str = Field(
        alias="FORWARDED_ALLOW_IPS",
        default="127.0.0.1,::1",
    )

    langfuse_public_key: str = Field(alias="LANGFUSE_PUBLIC_KEY", default="")
    langfuse_secret_key: SecretStr = Field(
        alias="LANGFUSE_SECRET_KEY", default=SecretStr("")
    )
    langfuse_host: str = Field(
        alias="LANGFUSE_HOST", default="https://api.langfuse.com"
    )

    @property
    def is_development(self) -> bool:
        """Return whether the current runtime should emit dev diagnostics."""
        return self.app_env.lower() in {"dev", "development", "local"}

    @property
    def trusted_proxy_hosts(self) -> tuple[str, ...]:
        """Return the configured trusted proxy IPs or CIDR ranges."""
        values = tuple(
            entry.strip()
            for entry in self.forwarded_allow_ips.split(",")
            if entry.strip()
        )
        return values or ("127.0.0.1", "::1")


def build_sqlalchemy_database_url(
    *,
    database_url: str | None,
    database_host: str | None,
    database_port: int | None,
    database_user: str | None,
    database_password: SecretStr | None,
    database_name: str | None,
) -> str:
    """Build a SQLAlchemy database URL from full or discrete database settings."""
    if database_url:
        return database_url

    missing_fields: list[str] = []

    if not database_host:
        missing_fields.append("DATABASE_HOST")
    if database_port is None:
        missing_fields.append("DATABASE_PORT")
    if not database_user:
        missing_fields.append("DATABASE_USER")
    if database_password is None:
        missing_fields.append("DATABASE_PASSWORD")
    if not database_name:
        missing_fields.append("DATABASE_NAME")

    if missing_fields:
        joined_missing_fields = ", ".join(missing_fields)
        raise ValueError(
            "Provide DATABASE_URL or all discrete database settings. "
            f"Missing: {joined_missing_fields}."
        )

    assert database_host is not None
    assert database_port is not None
    assert database_user is not None
    assert database_password is not None
    assert database_name is not None

    encoded_user = quote_plus(database_user)
    encoded_password = quote_plus(database_password.get_secret_value())
    return (
        "postgresql+psycopg://"
        f"{encoded_user}:{encoded_password}@"
        f"{database_host}:{database_port}/{database_name}"
    )


class DatabaseSettings(BaseSettings):
    """Database-only settings for tooling that should not require app secrets."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    # Database settings
    database_url: str | None = Field(alias="DATABASE_URL", default=None)
    database_host: str | None = Field(alias="DATABASE_HOST", default=None)
    database_port: int | None = Field(alias="DATABASE_PORT", default=None)
    database_user: str | None = Field(alias="DATABASE_USER", default=None)
    database_password: SecretStr | None = Field(alias="DATABASE_PASSWORD", default=None)
    database_name: str | None = Field(alias="DATABASE_NAME", default=None)

    @model_validator(mode="after")
    def validate_database_configuration(self) -> Self:
        """Require either a full database URL or all discrete database fields."""
        build_sqlalchemy_database_url(
            database_url=self.database_url,
            database_host=self.database_host,
            database_port=self.database_port,
            database_user=self.database_user,
            database_password=self.database_password,
            database_name=self.database_name,
        )
        return self

    @property
    def sqlalchemy_database_url(self) -> str:
        """Return the SQLAlchemy-compatible database URL."""
        return build_sqlalchemy_database_url(
            database_url=self.database_url,
            database_host=self.database_host,
            database_port=self.database_port,
            database_user=self.database_user,
            database_password=self.database_password,
            database_name=self.database_name,
        )


class LlmSettings(BaseSettings):
    """Settings used by LLMs and RAG."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    # LLM settings
    llm_provider: LlmProvider = Field(alias="LLM_PROVIDER")
    llm_base_url: str | None = Field(alias="LLM_BASE_URL", default=None)
    llm_api_key: SecretStr = Field(alias="LLM_API_KEY")
    llm_model_fast: str = Field(alias="LLM_MODEL_FAST")
    llm_model_reasoning: str = Field(alias="LLM_MODEL_REASONING")
    llm_model_vision: str | None = Field(alias="LLM_MODEL_VISION", default=None)

    # Matches the vision model's required OpenRouter route -- reused by both
    # scripts/validate_parsing_quality_sample.py and
    # scripts/escalate_vision_boundaries.py rather than reinvented per
    # script. Fallback disabled so a transient provider outage surfaces as
    # an exception (caught and retried by each caller) instead of silently
    # rerouting to a different, unvalidated upstream.
    llm_vision_provider_order: list[str] = Field(
        alias="LLM_VISION_PROVIDER_ORDER", default_factory=lambda: ["google-vertex"]
    )
    llm_vision_allow_fallbacks: bool = Field(
        alias="LLM_VISION_ALLOW_FALLBACKS", default=False
    )

    # Rough, provider-published per-1M-token prices at time of writing --
    # check against the provider's current pricing before treating
    # estimated_cost_usd (scripts/escalate_vision_boundaries.py) as
    # authoritative for a real budgeting decision.
    llm_vision_input_cost_per_1m_tokens_usd: float = Field(
        alias="LLM_VISION_INPUT_COST_PER_1M_TOKENS_USD", default=0.30
    )
    llm_vision_output_cost_per_1m_tokens_usd: float = Field(
        alias="LLM_VISION_OUTPUT_COST_PER_1M_TOKENS_USD", default=2.50
    )

    # Pinned per M1-08b: baidu/fp8 is the required OpenRouter route for
    # deepseek/deepseek-v4-flash-0731 for corpus classification; fallback is
    # disabled so a transient provider outage surfaces as a classifier
    # exception (caught and retried by classify_and_enrich_clauses) instead
    # of silently rerouting to a different, unvalidated upstream.
    llm_classification_provider_order: list[str] = Field(
        alias="LLM_CLASSIFICATION_PROVIDER_ORDER",
        default_factory=lambda: ["baidu/fp8"],
    )
    llm_classification_allow_fallbacks: bool = Field(
        alias="LLM_CLASSIFICATION_ALLOW_FALLBACKS", default=False
    )

    # Empirically: concurrency past ~5-10 workers gave diminishing/negative
    # returns (higher structured-output failure rate under load) -- see the
    # M1-05b PR discussion. Raised to the top of that range for M1-08b now
    # that classify_and_enrich_clauses retries a transient failure (3
    # attempts, 5s apart) instead of eating it silently -- the retry
    # absorbs the risk that kept concurrency capped before.
    llm_classification_max_workers: int = Field(
        alias="LLM_CLASSIFICATION_MAX_WORKERS", default=10
    )

    # RAG settings
    embedding_model: str = Field(alias="EMBEDDING_MODEL")
    reranker_model: str = Field(alias="RERANKER_MODEL")


class ParsingSettings(BaseSettings):
    """Settings used by the text-extraction pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    low_char_page_threshold: int = Field(
        alias="EXTRACTION_LOW_CHAR_PAGE_THRESHOLD", default=40
    )
    ocr_dpi: int = Field(alias="OCR_DPI", default=150)
    clause_tree_orphan_ratio_threshold: float = Field(
        alias="CLAUSE_TREE_ORPHAN_RATIO_THRESHOLD", default=0.15
    )
    clause_max_page_span: int = Field(alias="CLAUSE_TREE_MAX_PAGE_SPAN", default=10)
    clause_max_char_count: int = Field(
        alias="CLAUSE_TREE_MAX_CHAR_COUNT", default=15000
    )


class ChunkingSettings(BaseSettings):
    """Settings used by [M3-01] clause-aware chunking."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    # Real-corpus evidence (4925 clauses / 30 docs, build/parsed_clauses.jsonl):
    # p25=102, p50=290 chars; 1066 clauses sit under 80 chars.
    chunk_min_char_count: int = Field(alias="CHUNK_MIN_CHAR_COUNT", default=150)

    # p75=915, p90=2143 -- a consistent split-piece size between the two.
    chunk_target_char_count: int = Field(alias="CHUNK_TARGET_CHAR_COUNT", default=1800)

    # 332/4925 clauses sit over 3000 chars -- the split rule engages for
    # exactly that evidenced tail.
    chunk_max_char_count: int = Field(alias="CHUNK_MAX_CHAR_COUNT", default=3000)

    # ~15-20% of target_char_count, a conventional RAG overlap ratio; only
    # engaged by the last-resort sliding-window rule (35/4925 clauses over
    # 8000 chars in the real corpus).
    chunk_sliding_window_overlap_chars: int = Field(
        alias="CHUNK_SLIDING_WINDOW_OVERLAP_CHARS", default=300
    )


class Settings(DatabaseSettings, LlmSettings):
    """Application settings loaded from environment variables."""

    redis_url: str = Field(alias="REDIS_URL", default="redis://localhost:6379/0")

    # Logging
    log_level: str = Field(alias="LOG_LEVEL", default="INFO")


@lru_cache
def get_observability_settings() -> ObservabilitySettings:
    """Get cached runtime settings used by logging and diagnostics."""
    return ObservabilitySettings()


@lru_cache
def get_database_settings() -> DatabaseSettings:
    """Get cached database-only settings.

    Separate from ``get_settings`` so database tooling -- Alembic, the
    integration-test path -- does not require the LLM credentials the full
    ``Settings`` object demands.
    """
    return DatabaseSettings()


@lru_cache
def get_llm_settings() -> LlmSettings:
    """Get cached runtime settings used by LLMs and RAG."""
    return LlmSettings()


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


@lru_cache
def get_parsing_settings() -> ParsingSettings:
    """Get cached settings used by the text-extraction pipeline."""
    return ParsingSettings()


@lru_cache
def get_chunking_settings() -> ChunkingSettings:
    """Get cached settings used by clause-aware chunking."""
    return ChunkingSettings()
