"""Prove the pgvector extension is usable, not merely installed.

`SELECT * FROM pg_extension` only says a row exists. What [M3-02] and
[M3-04] actually depend on is storing a vector column and ordering by a
distance operator, so that is what this asserts.
"""

import pytest
from sqlalchemy import Engine, text

pytestmark = pytest.mark.integration


def test_vector_extension_is_installed(postgres_engine: Engine) -> None:
    with postgres_engine.connect() as connection:
        version = connection.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).scalar_one()

    major, minor, *_ = (int(part) for part in version.split("."))
    # Iterative index scans -- required by [M3-04]'s filtered-search path --
    # landed in pgvector 0.8.0. See docs/DATABASE.md.
    assert (major, minor) >= (0, 8), f"pgvector {version} is older than 0.8.0"


def test_vector_column_round_trips_and_orders_by_distance(
    postgres_engine: Engine,
) -> None:
    with postgres_engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TEMPORARY TABLE pgvector_probe ("
                "  id integer PRIMARY KEY,"
                "  embedding vector(3)"
                ")"
            )
        )
        connection.execute(
            text("INSERT INTO pgvector_probe (id, embedding) VALUES (:id, :embedding)"),
            [
                {"id": 1, "embedding": "[1,0,0]"},
                {"id": 2, "embedding": "[0,1,0]"},
                {"id": 3, "embedding": "[0.9,0.1,0]"},
            ],
        )

        rows = (
            connection.execute(
                text(
                    "SELECT id FROM pgvector_probe "
                    "ORDER BY embedding <-> '[1,0,0]'::vector"
                )
            )
            .scalars()
            .all()
        )

    # Nearest first: the exact match, then its neighbour, then the orthogonal
    # vector -- an ordering only a working distance operator produces.
    assert rows == [1, 3, 2]
