# insurance-claims-multiagent-rag
Multi-agent assistant that checks insurance claims against real Brazilian policy conditions (SUSEP). LangGraph conditional graph with parallel agents, hybrid RAG with reranking, human-in-the-loop checkpoint and full audit trail. FastAPI + Clean Architecture. Retrieval and end-to-end quality measured on a hand-curated golden set.

The MIT license covers the source code in this repository only.
Documents under `data/policies/raw/` are published by their respective
insurers and remain their property — see NOTICE.md.

This system can say whether a described event is consistent or
inconsistent with the conditions of a registered insurance product — it
cannot say whether a real claim is covered or denied. See
[`docs/SCOPE.md`](docs/SCOPE.md) for the full statement.

## Quickstart

The literal, followable path from a clean clone to a running assessment,
entirely through Docker Compose — [M5-09]. Design rationale for the stack
is in [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

```bash
cp .env.example .env
# fill in LLM_PROVIDER / LLM_BASE_URL / LLM_API_KEY / LLM_MODEL_FAST /
# LLM_MODEL_REASONING -- the rest of .env.example's defaults already match
# the Compose services below.

docker compose up -d postgres redis   # infra first -- api/worker need data in place before they can start
make fetch-corpus-artifacts           # skip the LLM-cost parsing stage (~10MB download)
make build-index                      # chunk -> Postgres -> embeddings against the running stack
docker compose up -d --build          # migrate, api, worker -- each healthchecked
```

`api`/`worker` read `build/chunks.jsonl` (the lexical retriever's BM25 index
and the exclusion-clause graph, `docs/DEPLOYMENT.md`) at startup, which is
why `make build-index` runs before the full `up`: starting `api`/`worker`
before it exists just crash-loops them on a clear "file not found" error.
The second `docker compose up -d --build` runs `migrate` once (schema +
checkpointer setup) then starts `api` (`curl localhost:${API_PORT:-8000}/health`
→ `200`) and `worker`. `make build-index`'s embedding step is a real,
one-time ~41-minute CPU pass the first time it runs — $0 in API cost, since
the embedder runs locally, but real wall-clock time (`docs/EMBEDDINGS.md`).
**Skipping that too:** `make fetch-embedding-cache` (or `make
fetch-demo-artifacts` in place of `fetch-corpus-artifacts` above, for both
fetches in one step) downloads a pre-computed embedding cache so the same
`make build-index` re-fills every vector in seconds instead — once the
maintainer has published that release (see `docs/DEPLOYMENT.md`'s "What
this issue ships, and what it deliberately doesn't (yet)"); until then it's
equivalent to the corpus-only fetch above.

```bash
curl -s -X POST localhost:${API_PORT:-8000}/v1/assessments \
  -H 'Content-Type: application/json' \
  -d '{"raw_text": "Bati o carro na traseira de outro veículo ao tentar estacionar."}'
# -> 202, {"assessment_id": "<id>", "status": "pending"}

curl -s localhost:${API_PORT:-8000}/v1/assessments/<id>
# -> status moves pending -> running -> awaiting_review, with the
# recommendation and its citations once ready
```

Endpoint shapes, the human-decision step and the audit trail are in
[`docs/API.md`](docs/API.md). `docker compose --profile tracing up -d` adds
the self-hosted Langfuse stack alongside the above — see "Tracing" below.

## Fast start: inspect the parsed corpus without running the pipeline

The full M1 parsing pipeline (OCR, LLM clause classification, the vision-LLM
boundary-escalation pass, LLM validation) costs real tokens and real
wall-clock time. If you just want to inspect the already-published result —
the finished 4,925-clause corpus and the LLM caches behind it — run this
instead of `make parse`:

```bash
make fetch-corpus-artifacts
```

This downloads a small (~10MB) release asset and needs no `.env`/LLM
credentials. See [`docs/PARSING.md`](docs/PARSING.md) for the published
accuracy numbers this corpus was measured against. Note: if you later run
`make parse` locally anyway, `git status`/`git diff` on `build/manifest.json`
will show a one-line difference on the `built_at_utc` timestamp field only —
everything else reproduces byte-identical.

## Development Commands

### System dependencies

Two policy PDFs have no extractable text layer and are OCR'd via
[Tesseract](https://github.com/tesseract-ocr/tesseract) (see [M1-02]).
Install it locally before running `make extract-text`:

```bash
sudo apt update && sudo apt install tesseract-ocr tesseract-ocr-por
```

Verify with `tesseract --version`, and confirm the Portuguese language pack
is present with `tesseract --list-langs`.

### Environment variables

Copy the example file and fill in the values for your environment:

```bash
cp .env.example .env
```

### Database (Postgres + pgvector) and Redis

This is the bare-metal dev loop -- infra in containers, `make serve` /
`make worker` on the host for hot reload. (For the fully containerized stack,
including `api` / `worker` themselves, see the Quickstart above; don't run
both at once against the same ports.) Postgres is required by the retrieval
index, the LangGraph checkpointer and the audit trail; Redis backs the
asynchronous assessment queue ([M5-05]). Bring both up locally, in this
order, on a clean clone:

```bash
cp .env.example .env
docker compose up -d postgres redis   # infra only -- not migrate/api/worker
make migrate
make setup-checkpointer
```

The `.env.example` defaults match the Compose services, so the sequence works
as written without editing anything first. `make migrate` applies the Alembic
migrations (the first one enables the `vector` extension); `make migrate-down`
rolls the latest one back. `make setup-checkpointer` is the second half of the
schema: the LangGraph checkpointer creates and migrates its own tables outside
Alembic, so a database with `make migrate` applied is still not ready to run the
graph. It is idempotent and reads the same `DATABASE_URL`.

Run the database-backed tests with:

```bash
make test-integration
```

This applies migrations to the `insurance_claims_test` database — created by
the Compose service on first boot — before running `pytest -m integration`.

See [`docs/DATABASE.md`](docs/DATABASE.md) for the pinned pgvector version and
why, how `CREATE EXTENSION vector` is executed and what privilege it needs,
the sync-vs-async engine decision, and why the checkpointer's schema is a
separate step. The human checkpoint that uses it is
[`docs/HUMAN_CHECKPOINT.md`](docs/HUMAN_CHECKPOINT.md).

### Index the corpus

One command rebuilds the whole searchable index from `data/policies/raw/`:

```bash
make build-index     # parse -> chunk -> Postgres -> embeddings
```

It needs the same environment as `make parse` (the `LLM_*` keys in `.env`, plus
Tesseract) and a running Postgres. When `build/parsed_clauses.jsonl` is already
present — from an earlier `make parse` or `make fetch-corpus-artifacts` — the
parse stage is skipped and the rest runs from cache. The manual breakdown of the
last two steps:

```bash
make load-chunks     # upsert the chunk corpus into Postgres (idempotent)
make embed-chunks    # embed the chunks; installs the optional `embed` group on first run
```

The embedding model (`Alibaba-NLP/gte-multilingual-base`) runs locally, so the
dollar cost is **$0.00** — no API key needed. A cold pass over the ~4,540-chunk
corpus is ~41 min of CPU time on an AMD Ryzen 5 5600H (a few minutes on a GPU);
re-runs are served from an on-disk cache and do zero inference. See
[`docs/EMBEDDINGS.md`](docs/EMBEDDINGS.md).

Score every retrieval configuration on the golden set with
`make eval-retrieval-matrix` — the committed comparison table and verdict are in
[`docs/RETRIEVAL_BENCHMARK.md`](docs/RETRIEVAL_BENCHMARK.md).

### Run the assessment API and the worker

Once the schema, the checkpointer and the index are in place (bare-metal dev
loop -- see the note above; skip this if you're already running the
containerized `api` / `worker` from the Quickstart):

```bash
make serve           # the API: uvicorn presentation.app:app on :8000
make worker          # the queue workers (a second shell)
```

`POST /v1/assessments` submits a claim (202, `status: "pending"`); a worker runs
the graph in the background. `GET /v1/assessments/{id}` reads its lifecycle state
(`pending` → `running` → `awaiting_review`, or `failed`) and, once ready, the
recommendation. `POST /v1/assessments/{id}/decision` submits the human decision
and resumes the run; `GET /v1/assessments/{id}/audit` returns the audit trail.
Endpoint shapes and error codes are in [`docs/API.md`](docs/API.md); the queue,
the retry/back-off policy and the dead-letter path are in
[`docs/ASYNC_PROCESSING.md`](docs/ASYNC_PROCESSING.md). `api` and `worker` also
run as Compose services from the same Dockerfile — see the Quickstart above
and [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) [M5-09].

### Tracing (optional)

A run is a tree of nodes, LLM calls and one retrieval step; when a verdict is
wrong, the question is which of them was. [M5-07] traces that into a
self-hosted Langfuse, which rides along in the Compose stack behind a profile:

```bash
# .env: set the three secrets first -- openssl rand -hex 32 each
#   LANGFUSE_NEXTAUTH_SECRET, LANGFUSE_SALT, LANGFUSE_ENCRYPTION_KEY
docker compose --profile tracing up -d    # + langfuse-web, langfuse-worker, clickhouse, minio
open http://localhost:3000                # sign in with LANGFUSE_INIT_USER_*
```

This also brings up `migrate` / `api` / `worker` alongside the tracing
services (they have no profile, so any `up` starts them) — if you're running
the bare-metal `make serve` / `make worker` instead, stop them first to avoid
a port clash. The project is seeded on first boot with the
`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` already in your `.env`, so
there is no key-copying step. Tracing is off unless both keys are set and
`TRACING_ENABLED` is not `false`, and the plain `docker compose up -d` from
the Quickstart runs identically with none of this. Reading a trace, and a
worked example of diagnosing a wrong verdict, are in
[`docs/OBSERVABILITY.md`](docs/OBSERVABILITY.md).

### Prompt-injection classifier (optional)

Every prompt in this system carries text this project does not control: a
retrieved clause excerpt or a claimant's own narrative
([`docs/PROMPT_INJECTION.md`](docs/PROMPT_INJECTION.md)). Delimiters, schema
rejection and metadata-only document trust are the actual guard and are
always on. The M5-08 issue's Appendix additionally spikes a runtime
classifier as an optional, advisory-only defense-in-depth layer — off by
default:

```bash
uv sync --group embed                          # transformers + torch
PROMPT_INJECTION_CLASSIFIER_ENABLED=true make serve   # and/or `make worker`
```

It never blocks a node or changes a verdict — a flagged span only adds one
entry to the audit trail. **On this project's own Portuguese corpus it is
not recommended**: the pinned model
(`protectai/deberta-v3-base-prompt-injection-v2`) is trained only on English
text and flags 70% of real, non-adversarial policy clauses in the measured
benchmark. That is a property of this model on this domain, not of the
approach — the code is kept as a working, tested reference for the pattern
on a domain closer to the model's training language. Method, the real
numbers and the reasoning behind the `false` default are in
[`docs/PROMPT_INJECTION_CLASSIFIER.md`](docs/PROMPT_INJECTION_CLASSIFIER.md).

### Pre-commit hooks

1. Install the dev dependency (skip if already in the lockfile — run `uv sync` instead):

```bash
uv add --dev pre-commit
```

2. Enable the hooks in your local clone:

```bash
uv run pre-commit install
```

3. Run the hooks against all files once, to check the existing codebase:

```bash
uv run pre-commit run --all-files
```

### Jupyter kernel for notebooks

Notebooks under `notebooks/` should run against this project's virtual
environment rather than a globally installed kernel. Register a
project-bound kernel with:

```bash
./scripts/setup_dev_kernel.sh
```

This adds `ipykernel` as a dev dependency (via `uv`) and registers a Jupyter
kernel named "Insurance Claims (uv)". Select it in Jupyter/VS Code, or launch
directly with:

```bash
uv run jupyter lab
```
