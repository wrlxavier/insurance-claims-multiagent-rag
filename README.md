# insurance-claims-multiagent-rag
Multi-agent assistant that checks insurance claims against real Brazilian policy conditions (SUSEP). LangGraph conditional graph with parallel agents, hybrid RAG with reranking, human-in-the-loop checkpoint and full audit trail. FastAPI + Clean Architecture. Retrieval and end-to-end quality measured on a hand-curated golden set.

The MIT license covers the source code in this repository only.
Documents under `data/policies/raw/` are published by their respective
insurers and remain their property — see NOTICE.md.

This system can say whether a described event is consistent or
inconsistent with the conditions of a registered insurance product — it
cannot say whether a real claim is covered or denied. See
[`docs/SCOPE.md`](docs/SCOPE.md) for the full statement.

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

### Database (Postgres + pgvector)

Postgres is required by the retrieval index, the LangGraph checkpointer and
the audit trail. Bring it up locally, in this order, on a clean clone:

```bash
cp .env.example .env
docker compose up -d postgres
make migrate
make setup-checkpointer
```

The `.env.example` defaults match the Compose service, so the sequence works
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

### Run the assessment API

Once the schema, the checkpointer and the index are in place:

```bash
make serve           # uvicorn presentation.app:app on :8000
```

`POST /v1/assessments` submits a claim, `GET /v1/assessments/{id}` reads its
state and recommendation, `POST /v1/assessments/{id}/decision` submits the human
decision and resumes the run, `GET /v1/assessments/{id}/audit` returns the audit
trail. Endpoint shapes, the error codes and the design notes are in
[`docs/API.md`](docs/API.md). The Docker Compose stack is [M5-09].

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
