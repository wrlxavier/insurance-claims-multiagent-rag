from collections.abc import Iterator
from os import getenv
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, MetaData
from sqlalchemy.orm import Session, sessionmaker

from infrastructure.database import (
    create_engine_from_database_url,
    create_session_factory,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_alembic_config(database_url: str) -> Config:
    """Build an Alembic config bound to the provided test database URL."""
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    config.attributes["database_url"] = database_url
    return config


def drop_all_tables(engine: Engine) -> None:
    """Drop every reflected table so migrations can rebuild the schema cleanly."""
    metadata = MetaData()
    metadata.reflect(bind=engine)
    metadata.drop_all(bind=engine)


@pytest.fixture(scope="session")
def postgres_database_url() -> str:
    database_url = getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("Set TEST_DATABASE_URL to run PostgreSQL integration tests.")
    return database_url


@pytest.fixture(scope="session")
def postgres_engine(postgres_database_url: str) -> Iterator[Engine]:
    engine = create_engine_from_database_url(postgres_database_url)
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def alembic_config(postgres_database_url: str) -> Config:
    return build_alembic_config(postgres_database_url)


@pytest.fixture
def session_factory(postgres_engine: Engine) -> sessionmaker[Session]:
    return create_session_factory(engine=postgres_engine)


@pytest.fixture
def db_session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    with session_factory() as session:
        yield session
        session.rollback()


@pytest.fixture(autouse=True)
def migrated_database(postgres_engine: Engine, alembic_config: Config) -> None:
    # Downgrade before dropping tables, not after: `drop_all_tables` removes
    # `alembic_version` too, and a downgrade with no version table cannot
    # know there is anything to revert. Running it first means the `vector`
    # extension is genuinely dropped and re-created every time, so
    # `upgrade head` proves the migration installs it rather than inheriting
    # it from an earlier run.
    command.downgrade(alembic_config, "base")
    drop_all_tables(postgres_engine)
    command.upgrade(alembic_config, "head")
