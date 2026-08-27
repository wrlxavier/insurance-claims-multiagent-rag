"""SQLAlchemy declarative base for persistence models."""

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

# Without this, Postgres names constraints itself and Alembic autogenerate
# cannot reliably refer to them: an index or constraint created in one
# migration gets an unpredictable name that a later `op.drop_constraint`
# has to guess. Set here rather than per-table so it holds from the first
# table [M3-02] adds onwards.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
