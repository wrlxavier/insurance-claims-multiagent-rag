# Database

Postgres with pgvector, Alembic migrations, and the integration-test path.
Landed by [M0-08], which creates the database *capability* — the schema
belongs to the issues that need it: the `chunk` table to [M3-02] (its
`embedding` vector column follows in the same issue's embedding-pipeline
half), the checkpointer tables **and** the `audit_event` table to [M4-09], the
domain tables to [M5-03].

The audit table moved forward from [M5-03] deliberately: [M4-09]'s DoD requires
the audit trail to be durable *and* separate from graph state, which is the
table. [M5-03] adds the `assessment` / `human_decision` tables, the repository
and unit-of-work adapters behind the [M5-02] ports, and the database-level
append-only enforcement on `audit_event` — see the [M5-03] sections below.

The initial migration creates the `vector` extension and nothing else.

## Local bring-up

```bash
cp .env.example .env
docker compose up -d postgres
make migrate
make setup-checkpointer
```

`make setup-checkpointer` is the second half of the schema, and it is separate
for a reason — see "The checkpointer owns its own schema" below.

`compose.yaml` currently holds one service. [M5-09] adds `api`, `redis`
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
can explain rather than as a configuration decision. [M3-02] measured this
(`docs/EMBEDDINGS.md`, "Filtered search and the fewer-than-`k` question") and
`tests/integration/test_ann_index.py` proves both the shortfall and the
`hnsw.iterative_scan = strict_order` fix.

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

Migrations pass the **bare** constraint name (`name="rule_valid"`, not
`name="ck_chunk_rule_valid"`) — Alembic's `op` context applies the same
convention the model does, and passing the already-expanded name gets it
expanded a second time (`ck_chunk_ck_chunk_rule_valid`).

## The chunk table ([M3-02])

`app/src/infrastructure/database/models.py` — `ChunkRow` — is the persistence
representation of the M3-01 chunk corpus (`build/chunks.jsonl`), indexed in
Postgres for retrieval. It is the first ORM model in the project.

The `embedding halfvec(768)` column landed in migration `20260827_03`
(`ALTER TABLE chunk ADD COLUMN`, plus the `pgvector` Python dependency). Its
width, half-precision storage and cosine metric follow the pinned model
contract — `Alibaba-NLP/gte-multilingual-base` (see `docs/EMBEDDINGS.md` and
`app/src/infrastructure/rag/embedding_config.py`). The `halfvec_cosine_ops`
HNSW index over the column is defined in `app/src/infrastructure/rag/ann_index.py`
— **not** a migration: the benchmark (`docs/EMBEDDINGS.md`, "Does the ANN index
earn its place") measured it at ~4,540 chunks and the chosen default retrieval
path is exact `<=>` over the metadata-filtered partition, so the index does not
belong in the schema. `tests/integration/test_ann_index.py` is the committed
proof of its filtered-search behaviour and the `hnsw.iterative_scan` fix.

- **Columns** mirror `infrastructure.rag.chunk_schema.ChunkRecord` field for
  field (except `text` → `embedded_text`), so the write path is a direct
  mapping and "carry the full [M1-05] provenance" is auditable column by
  column. The enum-valued columns (`rule`, `clause_type`, `type_source`,
  `source`) are `TEXT` + a named `CHECK`, not native PG enums: there is no
  enum precedent in the project, `ALTER TYPE` is transaction-hostile and
  values cannot be removed, and the `ck_` naming convention already exists
  for exactly this. A unit test keeps each `CHECK` value set in step with its
  domain enum.

- **`chunk_id` is the primary key and is deterministic upstream.** [M3-01]
  sets it to the anchor `clause_id` for a one-chunk clause and
  `f"{clause_id}#{index}"` for a split; `clause_id = f"{document_id}:{path}"`
  is [M1-07]'s structural-path id, not a hash. Same input, same id. The write
  path (`infrastructure.database.chunk_repository.upsert_chunks`) is therefore
  `INSERT ... ON CONFLICT (chunk_id) DO UPDATE`, refreshing every non-key
  column *except `embedding`* — a re-run over the same corpus neither
  duplicates rows nor needs a wipe, and refreshing chunk metadata never
  discards vectors already computed. It flushes but never commits; the caller
  owns the transaction.

- **`bundle_section` is a genuinely nullable column** — no server default, no
  sentinel. A strict M3-04 filter `WHERE bundle_section = :x` then silently
  excludes unknown-bundle chunks, and that exclusion is expressible and
  testable in plain SQL (M3-04's cross-note from [M1-06]: ~40% of a
  multi-product document's clauses land in the `None` bucket). Contrast
  `infrastructure/parsing/clause_tree_caching.py`, which collapses `None ↔ ""`
  — but only for the Parquet clause-tree cache, never for a value that
  reaches the table.

- **`embedded_text` vs `display_text`.** `embedded_text` is the exact string
  the embedding model sees — the clause body with its [M3-01] ancestor-path
  breadcrumb prepended. `display_text` is the same clause with only that
  injected breadcrumb removed, keeping its own heading line: the quoted
  excerpt [M4-01]'s citation type needs, which is not the string the model
  should have seen. `char_count` measures `embedded_text`.

- **`embedding` is genuinely nullable, and the embedding pipeline owns it.**
  An un-embedded chunk carries `NULL`; `WHERE embedding IS NULL` is the
  pipeline's resumable cursor
  (`infrastructure.rag.embedding_pipeline.embed_missing_chunks`), which
  batches, retries the shared 3-attempt/5s policy around each embed call, and
  commits per batch so an interrupted run resumes on exactly the remaining
  rows. Query code orders by `ChunkRow.embedding.cosine_distance(...)` — the
  `<=>` operator, matching `embedding_config.DISTANCE_METRIC`. Known gap: a
  chunk whose `embedded_text` changes but whose deterministic `chunk_id` does
  not keeps its stale vector — re-embedding on content change is the deferred
  content-hash-cache item's job.

- **Indexes:** one per field M3-04 filters by (`clause_type`,
  `bundle_section`, `susep_process`, `cnpj`, `product_line`) plus a composite
  `(susep_process, cnpj)` for M3-04's *default* retrieval path. `insurer` is
  not indexed — M3-04 filters insurers by CNPJ, never by name. All of these
  are forward-looking: at ~4,900 chunks Postgres seq-scans in well under a
  millisecond regardless. The `embedding` column has **no** index in the
  migrations — the HNSW definition lives in `infrastructure.rag.ann_index` and
  [M3-02]'s benchmark found the composite `(susep_process, cnpj)` btree +
  exact sort is what the planner picks for the default path anyway (~0.6 ms).

## The checkpointer owns its own schema ([M4-09])

`PostgresSaver` creates and migrates its own tables — `checkpoints`,
`checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations` — through
`setup()`, tracking its versions in a table of its own. Alembic knows nothing
about them, and folding them into a migration would mean either importing
LangGraph into a migration file (which `20260827_02` explicitly rules out for
app code, for the same reason) or hand-copying a schema this project does not
own and would have to re-copy on every upgrade.

So a database with `alembic upgrade head` applied is still not ready for the
graph, and the bring-up has two steps:

```bash
make migrate              # the application schema, incl. audit_event
make setup-checkpointer   # the checkpointer's own tables
```

`make setup-checkpointer` runs `scripts/setup_checkpointer.py`, which is the one
place `setup()` is called — the way `make migrate` is the one place Alembic runs.
It is idempotent, and it reads the same `DATABASE_URL` (or discrete `DATABASE_*`)
settings as everything else, which is what "the checkpointer uses the same
database as the rest of the service" means in practice. Every other caller of
`infrastructure.graph.checkpointer.open_claim_checkpointer` gets
`assert_checkpointer_ready` instead, which names this command rather than letting
a missing table surface as a psycopg `UndefinedTable` from inside a graph run.

**Alembic must be told to ignore them.** Two tools now write to the same
database, and autogenerate compares the database against `Base.metadata`: it
reads four tables it cannot account for as tables the models *dropped*, and
proposes a migration that deletes the checkpointer — and every paused run with
it. `alembic/env.py` therefore passes an `include_name` filter listing those
tables, and `alembic check` is clean with them present. The list is a plain
literal there, like the enum value sets in `20260827_02`, so `env.py` imports no
app code; `tests/unit/infrastructure/graph/test_checkpointer.py` ties it back to
`infrastructure.graph.checkpointer.CHECKPOINTER_TABLES`, and
`tests/integration/test_human_checkpoint.py` runs `alembic check` against a
database that has both schemas.

One consequence for tests: `tests/integration/conftest.py` drops **every**
reflected table between tests, checkpoint tables included, and the Alembic replay
does not bring them back. A checkpointer test creates them itself — which is also
how `tests/integration/test_human_checkpoint.py` proves `setup()` builds the
schema from nothing.

## The audit_event table ([M4-09])

`AuditEventRow` is the persistence representation of
`infrastructure.graph.state.AuditEvent`, the same way `ChunkRow` is of a
`ChunkRecord`: one row per event a graph run produced, written once by the human
checkpoint when the analyst decides.

Primary key `(thread_id, sequence)` rather than an invented event id. The
checkpoint node re-runs from the top whenever its thread is resumed, so the write
has to be repeatable; an event's position in its thread's trail is already
deterministic, which makes `ON CONFLICT DO NOTHING` sufficient. `claim_id` is
indexed because that is what a human searches by; `thread_id` is already covered
by the key. A nullable `payload` JSONB column carries the analyst's whole
`HumanDecision` on the one row that has one, so "what the human decided" is a
`SELECT` rather than a checkpoint deserialization.

No CHECK constraints, unlike `chunk`: none of these columns is a closed enum —
`node` and `action` are free-form strings each node chooses, and constraining
them would turn adding a node into a migration.

The trail is append-only by construction — `infrastructure/database/audit_repository.py`
offers an insert and nothing else — and, since [M5-03], by the database too: see
"Append-only enforcement" below.

Rationale and the interrupt contract that produces these rows:
`docs/HUMAN_CHECKPOINT.md`.

## The assessment and human_decision tables ([M5-03])

`AssessmentRow` / `HumanDecisionRow` (`app/src/infrastructure/database/models.py`)
are the persistence representation of the servable aggregate
`application.assessment_record.AssessmentRecord` and the analyst's
`domain.human_decision.HumanDecision` recorded beside it — the same
one-row-class-per-type pattern as `ChunkRow` and `AuditEventRow`. The
row ↔ aggregate translation lives entirely in
`infrastructure.database.assessment_mapper` (`docs/DOMAIN.md` names this mapper
as [M5-03]'s deliverable). Migration `20260903_01`.

- **`assessment`** — column per aggregate field. `verdict` and `status` are
  `TEXT` + a named `CHECK` against the domain enum (like `chunk`'s enum columns;
  `tests/unit/infrastructure/database/test_models.py` ties each `CHECK` set back
  to the enum). `context_sufficient` is the one nullable column — it is
  genuinely tri-state (`True` / `False` / `None`: retrieval succeeded / the
  [M3-07] gate fired / retrieval never ran). Indexed on `claim_id` (a claim's
  runs, and what a human searches by), `status` (the awaiting-review queue) and
  `created_at` (`AssessmentRepository.list` orders newest-first).

- **`citations` and `consistency_flags` are `JSONB`, not child tables.** Both
  are frozen value-object tuples with no identity of their own, always loaded
  whole with the record and never filtered or joined by field — the same call
  as `audit_event.payload`. A `citation` child table would add a join, an
  ordering column and a second mapper layer for no query it would ever serve.
  `missing_information` is a plain `TEXT[]`, like `chunk.source_clause_ids`.

- **`human_decision`** — one row per *settled* assessment. `assessment_id` is
  both the primary key and a foreign key to `assessment`: "a decision always
  references the assessment it acted on" (M5-01) made structural. The
  `edited_assessment` JSONB holds the analyst's revised `Assessment` and is
  present exactly when `decision = 'edit'` — a `CHECK`
  (`(decision = 'edit') = (edited_assessment IS NOT NULL)`) enforces the
  biconditional, mirroring `HumanDecision`'s own validator. The ORM column uses
  `JSONB(none_as_null=True)` so a Python `None` becomes SQL `NULL` rather than
  the JSON value `null`, which the `CHECK` would read as present.

The clause corpus is **not** a new table — `SqlAlchemyClauseRepository`
(`clause_repository.py`) projects `domain.policy_clause.PolicyClause` from the
existing `chunk` table, reassembling a clause from every row whose
`source_clause_ids` array contains its id (a split clause rejoins its
`display_text` in `chunk_index` order). This is the same reconstruction
`infrastructure.rag.graph_retrieval_adapter.build_clause_index` does, and the
id it matches is the one a `Citation` carries.

## Append-only enforcement ([M5-03])

`audit_event` rejects `UPDATE` and `DELETE` at the database. Migration
`20260903_02` installs a `plpgsql` function that `RAISE EXCEPTION`s and a
`BEFORE UPDATE OR DELETE ON audit_event` trigger that calls it.

A trigger, not an `ON UPDATE ... DO INSTEAD NOTHING` rule: a rule swallows the
write silently, so a bug (or a compromised connection) trying to rewrite history
would look like it succeeded. The trigger fails loudly — which is also what
`tests/integration/test_audit_event_append_only.py` asserts. The insert path is
untouched: `append_audit_events` uses `INSERT ... ON CONFLICT DO NOTHING`, which
never fires an `UPDATE`, so the checkpoint node's idempotent re-write on resume
still works.

The checkpointer's own tables stay outside Alembic (see "The checkpointer owns
its own schema" above); [M5-03]'s DoD line about migrating them is met by
`make setup-checkpointer`, built in [M4-09], not a migration that would
duplicate `PostgresSaver.setup()`.

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
