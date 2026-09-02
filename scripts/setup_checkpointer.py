#!/usr/bin/env python3
"""Run the LangGraph checkpointer's migrations -- [M4-09].

The checkpointer owns its own schema: ``PostgresSaver.setup()`` creates the
``checkpoints`` / ``checkpoint_blobs`` / ``checkpoint_writes`` tables and tracks
its versions in ``checkpoint_migrations``, entirely outside Alembic. So a
database that has had ``make migrate`` run against it is still not ready for the
graph, and this script is the second half of the bring-up:

    make migrate            # the application schema, incl. `audit_event`
    make setup-checkpointer # the checkpointer's own tables

This is the one place ``setup()`` is called, the way ``make migrate`` is the one
place Alembic runs -- every other caller of
``infrastructure.graph.checkpointer.open_claim_checkpointer`` gets
``assert_checkpointer_ready`` instead, which turns a missing schema into a
message naming this command.

Idempotent: re-running applies only the migrations the database has not seen.
Acts on whatever ``DATABASE_URL`` -- or the discrete ``DATABASE_*`` variables --
resolve to, so pointing it at the test database is
``DATABASE_URL=$TEST_DATABASE_URL make setup-checkpointer``.
"""

from infrastructure.config.settings import get_database_settings
from infrastructure.graph.checkpointer import open_claim_checkpointer


def main() -> None:
    """Create or upgrade the checkpointer tables on the configured database."""
    settings = get_database_settings()
    with open_claim_checkpointer(settings.sqlalchemy_database_url, setup=True):
        pass
    # The URL is not echoed: it carries the password.
    print("Checkpointer tables are up to date.")


if __name__ == "__main__":
    main()
