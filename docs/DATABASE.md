# Database

Postgres with pgvector, Alembic migrations, and the integration-test path.
Landed by [M0-08], which creates the database *capability* — the schema
belongs to the issues that need it: the chunk and vector table to [M3-02],
the checkpointer tables to [M4-09], the domain and audit tables to [M5-03].

The initial migration creates the `vector` extension and nothing else.

## Local bring-up

```bash
cp .env.example .env
docker compose up -d postgres
make migrate
```

`docker-compose.yaml` currently holds one service. [M5-09] adds `api`, `redis`
and `langfuse` to that same file; the Postgres block is written so they are
appended rather than requiring it to be rewritten.

Roll a migration back with `make migrate-down` (one step). Both targets read
the same settings loader the application does, so they act on whatever
`DATABASE_URL` — or the discrete `DATABASE_*` variables — resolve to.

## Minimum pgvector version: 0.8.0 (pinned at 0.8.6)

The image is pinned as `pgvector/pgvector:0.8.6-pg17-bookworm`: Postgres major
version *and* pgvector version are both explicit values. The floating `pg17`
tag would let a pgvector upgrade in silently, and the features below are
version-gated, so "whatever the tag resolves to today" is not a version this
project can reason about.

Verified against the pgvector CHANGELOG and README (checked 2026-08-27), not a
tutorial:

| Feature | Needed by | Available from |
| --- | --- | --- |
| `halfvec` type | [M3-02] (half-precision storage for the chunk vectors) | 0.7.0 |
| **Iterative index scans** | [M3-04] (metadata pre-filter + ANN) | **0.8.0** |

Iterative index scans are what set the floor at **0.8.0**. [M3-04]'s *default*
retrieval path filters by SUSEP process and insurer CNPJ before searching. An
ANN index combined with a restrictive pre-filter can return fewer than *k*
rows, because the index scan exhausts its candidate list on rows the filter
then discards; iterative scans make the index keep scanning until enough
results are found. Without them, that shows up as a recall regression nobody
can explain rather than as a configuration decision.

Indexable dimension limits, which constrain the embedding model [M3-02] may
choose:

- HNSW / IVFFlat over `vector`: up to **2,000** dimensions.
- HNSW / IVFFlat over `halfvec`: up to **4,000** dimensions.
- Storage (unindexed) allows up to 16,000 dimensions for both.

A model producing more than 2,000 dimensions is therefore only indexable via
`halfvec`, which is why the `halfvec` availability floor matters at all.

## `CREATE EXTENSION vector` runs in the Alembic migration

`vector` is **not** a trusted extension, so `CREATE EXTENSION vector` requires
one of:

- a **superuser** role, or
- the extension already installed in the database, in which case
  `CREATE EXTENSION IF NOT EXISTS vector` is a no-op that needs no privilege.

The decision — among "in the migration", "in a container init script", or
"both" — is: **the initial migration owns it, and nothing else creates it.**

- One source of truth. A `docker-entrypoint-initdb.d` script only runs on a
  Compose volume's first boot, so it would cover exactly one of the three
  databases this project uses and leave the other two (CI's service container,
  any managed Postgres) to be set up some other way.
- `alembic upgrade head` on a clean database is then sufficient on its own,
  which is precisely what the CI integration job and `make test-integration`
  rely on.
- Compose and the CI service container both connect as the image's
  `POSTGRES_USER`, which is a superuser, so the privileged step just works
  locally and in CI.

**When the application role is not a superuser** — the normal case on a
managed Postgres — `alembic upgrade head` fails on the first migration with:

```
permission denied to create extension "vector"
HINT:  Must have CREATE privilege on current database to create this extension.
```

The fix is not to grant the application role superuser. A DBA runs
`CREATE EXTENSION vector;` once, out of band, against that database; the
migration's `IF NOT EXISTS` then finds it present and passes as a no-op, and
every later migration runs under the ordinary application role.

`docker/postgres/initdb/01-create-test-database.sql` exists, but it only
*provisions* the `insurance_claims_test` database so `make test-integration`
works right after `docker compose up -d`. It does not touch the extension.

## Sync psycopg3 is the default engine

`app/src/infrastructure/database/session.py` builds a **synchronous**
SQLAlchemy engine over **psycopg 3** (`postgresql+psycopg`). Decided once here
rather than per-caller, because the batch scripts and the future FastAPI
service have different natural answers and only one of them can be the
default:

- It is already what `build_sqlalchemy_database_url()` in
  `app/src/infrastructure/config/settings.py` emits, from [M0-03]. Choosing
  async would have meant either a second URL builder or a driver rewrite.
- Alembic is synchronous, and so is every consumer that lands next: [M3-02]'s
  batch embedding pipeline and the `scripts/` entry points. An async default
  would put an `asyncio.run()` at each of those call sites for no benefit —
  a batch job that saturates one connection gains nothing from concurrency.
- psycopg 3 serves both modes from one driver. [M5]'s FastAPI service can add
  `create_async_engine()` on the **same** `postgresql+psycopg` URL, so the
  async path is an addition rather than a driver migration.

The async engine is [M5]'s to add. This module deliberately does not ship a
second, unused engine ahead of a caller for it.

## Migration file naming

`alembic.ini` sets:

```ini
file_template = %%(year)d%%(month).2d%%(day).2d_%%(rev)s_%%(slug)s
```

Migration files are therefore named `YYYYMMDD_NN_<slug>.py`, with the revision
id set to the matching `YYYYMMDD_NN` — `NN` being a two-digit sequence within
that day. The first one is
`alembic/versions/20260827_01_enable_vector_extension.py`, revision
`20260827_01`. Date-ordered names mean `ls alembic/versions` reads in the same
order the migrations apply, which a hash-based default id does not give.

Create a new migration with `uv run alembic revision -m "<message>"` and set
its revision id by hand to the `YYYYMMDD_NN` matching the generated filename.

## Constraint naming convention

`Base.metadata` in `app/src/infrastructure/database/base.py` carries a
`naming_convention`, so every index, unique constraint, check constraint,
foreign key and primary key gets a deterministic name. Without it Postgres
names them itself, and a later migration that has to drop one is reduced to
guessing what it was called. Set on the metadata rather than per-table so it
holds from the first table [M3-02] adds onwards.

## Integration tests

`make test-integration` runs `scripts/run_integration_tests.sh`, which applies
migrations before executing the suite:

```bash
export DATABASE_URL="$TEST_DATABASE_URL"
uv run alembic upgrade head
uv run pytest -m integration tests/integration
```

Pointing `DATABASE_URL` at the test database is what aims Alembic there —
`alembic/env.py` still resolves its URL through the [M0-03] settings loader,
so there is no second source of truth.

`TEST_DATABASE_URL` is read from the environment, falling back to the key of
that name in `.env` — the shell does not read `.env` the way pydantic-settings
does, and the bring-up sequence above stops at `cp .env.example .env`. With
neither set, the runner exits with a message; running `pytest -m integration`
directly instead skips the tests rather than erroring.

The tests are marked `integration` per [M0-02], so `make test` — which runs
`-m "not integration and not eval"` — never tries to reach a database.

The runner targets `tests/integration` rather than every `integration`-marked
test in the repository. The marker predates this directory and also covers the
[M1-02]/[M1-04] tests under `tests/unit/infrastructure/parsing/`, which need
the real Tesseract binary and no database at all. Mixing them in would mean a
missing OCR binary failing the database job — the same attribution problem the
separate CI job exists to avoid. Run those explicitly with:

```bash
bash scripts/run_integration_tests.sh tests/unit/infrastructure/parsing
```

`tests/integration/conftest.py` downgrades to base, drops every reflected
table and re-applies migrations around each test, so the extension is genuinely
created by the migration on every run rather than inherited from an earlier one.

CI runs this as its own `integration` job, separate from `quality`, against the
same pinned pgvector image — a database failure should read as a database
failure, not as a lint or type error.
