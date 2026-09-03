"""Alembic environment for SQLAlchemy metadata-driven migrations."""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from alembic.runtime.environment import NameFilterParentNames, NameFilterType
from sqlalchemy import MetaData, engine_from_config, pool
from sqlalchemy.engine import Connection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "app" / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Tables LangGraph's PostgresSaver creates and migrates itself ([M4-09],
# `make setup-checkpointer`). They live in the same database but not in
# `Base.metadata`, so without this filter `--autogenerate` reads them as tables
# the models dropped and emits a migration that DELETES the checkpointer --
# taking every paused run with it.
#
# Kept as plain literals, like the enum value sets in `20260827_02`, so this
# file imports no app code that could move under it;
# `tests/unit/infrastructure/graph/test_checkpointer.py` ties them back to
# `infrastructure.graph.checkpointer.CHECKPOINTER_TABLES`.
UNMANAGED_TABLES = frozenset(
    {
        "checkpoints",
        "checkpoint_blobs",
        "checkpoint_writes",
        "checkpoint_migrations",
    }
)


def include_name(
    name: str | None,
    type_: NameFilterType,
    parent_names: NameFilterParentNames,
) -> bool:
    """Hide the tables Alembic does not own from autogenerate."""
    if type_ == "table":
        return name not in UNMANAGED_TABLES
    return True


def load_target_metadata() -> MetaData:
    """Load the SQLAlchemy metadata tracked by Alembic."""
    # Imported here, not at module level, because it only resolves once
    # SRC_DIR is on sys.path above.
    from infrastructure.database import Base

    return Base.metadata


target_metadata = load_target_metadata()


def get_database_url() -> str:
    """Resolve the target database URL for Alembic commands.

    Reads the [M0-03] settings loader rather than `alembic.ini`, so the
    project has one source of truth for the database URL. Tests bind a
    different database by setting `config.attributes["database_url"]`.
    """
    configured_url = config.attributes.get("database_url")
    if configured_url is not None:
        return str(configured_url)

    from infrastructure.config.settings import DatabaseSettings

    return DatabaseSettings().sqlalchemy_database_url


def run_migrations_offline() -> None:
    """Run migrations without creating a database connection."""
    context.configure(
        url=get_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        include_name=include_name,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_with_connection(connection: Connection) -> None:
    """Run migrations within an existing SQLAlchemy connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        include_name=include_name,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations using a live SQLAlchemy engine."""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = get_database_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        run_migrations_with_connection(connection)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
