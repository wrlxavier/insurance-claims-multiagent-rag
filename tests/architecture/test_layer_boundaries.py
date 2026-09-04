"""Enforce the Clean Architecture dependency rule via static import checks.

domain and application must stay importable with nothing but the standard
library on the path. This is checked here rather than left to code review,
so a violation fails CI instead of surviving until someone notices it in a
PR diff.

The same _import_checker helper is reused for the application-only check
added in M5-02, so extending the forbidden set there does not require
rewriting this scanning logic.
"""

from pathlib import Path

import pytest

from tests.architecture._import_checker import find_forbidden_imports, format_violations

REPO_ROOT = Path(__file__).resolve().parents[2]

DOMAIN_DIR = REPO_ROOT / "app" / "src" / "domain"
APPLICATION_DIR = REPO_ROOT / "app" / "src" / "application"

# Top-level module names, not import paths — e.g. "fastapi" catches both
# `import fastapi` and `from fastapi import APIRouter`.
FORBIDDEN_ROOTS = frozenset(
    {
        "fastapi",
        "sqlalchemy",
        # [M0-08] brought the database in; alembic and psycopg are the two new
        # importable roots it added, and neither belongs above infrastructure.
        "alembic",
        "psycopg",
        "pydantic",
        "langgraph",
        "langchain",
        # [M5-05]: the async queue. `AssessmentQueue` is a port; the RQ/Redis
        # adapter stays in infrastructure.
        "rq",
        "redis",
        # [M5-07]: tracing. `TracePort` is a port on the graph layer; the
        # Langfuse client and its OpenTelemetry exporter stay in infrastructure.
        "langfuse",
        "opentelemetry",
        "fitz",
        "pyarrow",
        "pytesseract",
        "PIL",
    }
)

# [M5-02] adds a stricter, application-only check. The application layer talks
# to infrastructure only through the ports it defines, so `infrastructure` is
# off-limits as an import. `langchain_core` / `langchain_openai` are separate
# top-level roots that `_top_level_module` does not fold into `langchain`, and
# LangGraph re-exports `langchain_core` types -- "the orchestrator hides
# LangGraph entirely" (M5-02 DoD) means barring those too. Scoped here rather
# than added to FORBIDDEN_ROOTS so the domain check is untouched.
APPLICATION_FORBIDDEN_ROOTS = FORBIDDEN_ROOTS | {
    "infrastructure",
    "langchain_core",
    "langchain_openai",
}


@pytest.mark.unit
def test_domain_has_no_forbidden_imports() -> None:
    violations = find_forbidden_imports(DOMAIN_DIR, FORBIDDEN_ROOTS)
    assert not violations, (
        "domain layer must depend on nothing but the standard library, "
        f"found:\n{format_violations(violations)}"
    )


@pytest.mark.unit
def test_application_has_no_forbidden_imports() -> None:
    violations = find_forbidden_imports(APPLICATION_DIR, FORBIDDEN_ROOTS)
    assert not violations, (
        "application layer must depend only on domain, "
        f"found:\n{format_violations(violations)}"
    )


@pytest.mark.unit
def test_application_imports_no_infrastructure_or_llm_sdk() -> None:
    violations = find_forbidden_imports(APPLICATION_DIR, APPLICATION_FORBIDDEN_ROOTS)
    assert not violations, (
        "application layer must not import infrastructure or an LLM SDK -- "
        f"depend on a port instead. Found:\n{format_violations(violations)}"
    )


@pytest.mark.unit
def test_layer_directories_exist() -> None:
    # Guards against the check silently passing because rglob found nothing
    # to scan — an empty layer directory is not the same as a clean layer.
    assert DOMAIN_DIR.is_dir(), f"expected domain layer at {DOMAIN_DIR}"
    assert APPLICATION_DIR.is_dir(), f"expected application layer at {APPLICATION_DIR}"
