#!/usr/bin/env bash
# Apply migrations to the test database, then run the database-backed
# integration tests.
#
# Migrations run first so the suite never faces a schema the repository has
# already moved past -- see docs/DATABASE.md.
#
# Defaults to tests/integration rather than every `integration`-marked test:
# the ones under tests/unit/infrastructure/parsing/ need the real Tesseract
# binary, not a database, and a missing OCR binary should not surface as a
# database failure. Pass a path to run those instead:
#
#   bash scripts/run_integration_tests.sh tests/unit/infrastructure/parsing
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

# Fall back to .env when TEST_DATABASE_URL is not already exported: it lives
# in .env, which pydantic-settings reads but the shell does not, and the
# README's bring-up sequence stops at `cp .env.example .env`. Read the one key
# rather than sourcing the file, which would execute every other value in it.
if [[ -z "${TEST_DATABASE_URL:-}" && -f .env ]]; then
  TEST_DATABASE_URL="$(grep -m1 -E '^TEST_DATABASE_URL=' .env | cut -d= -f2-)"
fi

if [[ -z "${TEST_DATABASE_URL:-}" ]]; then
  echo "TEST_DATABASE_URL must be set (in the environment or in .env) to run integration tests." >&2
  exit 1
fi

# Exported, not just set: the test fixtures read TEST_DATABASE_URL from the
# environment and skip when it is absent.
export TEST_DATABASE_URL
export PYTHONPATH="app/src"
# Alembic resolves its URL through DatabaseSettings; pointing DATABASE_URL at
# the test database is what aims it there, without a second settings source.
export DATABASE_URL="$TEST_DATABASE_URL"

uv run alembic upgrade head
uv run pytest -m integration "${@:-tests/integration}"
