# Deployment: the Docker Compose stack

How the `api` / `worker` / `migrate` Compose services are built and wired —
[M5-09]. Step-by-step run instructions live in README's Quickstart; this
document is the "why", the same split `docs/ASYNC_PROCESSING.md` and
`docs/OBSERVABILITY.md` use for their own services.

## One image, two containers

`api` and `worker` are built from the same `Dockerfile` and share the same
composition root (`infrastructure.bootstrap.build_core_components` — the same
one `presentation/app.py`'s lifespan and `scripts/run_assessment_worker.py`
both call). They need the same dependency set: the `embed` uv group
(`sentence-transformers` / `transformers`, which pull in `torch`) is required
at runtime because the retrieval stack loads the embedder and cross-encoder
in-process on both the request path and the queue-worker path
(`docs/EMBEDDINGS.md`, `docs/RERANKING.md`) — there is no "thin" variant of
either process. `worker` overrides `command:` to run
`python -m scripts.run_assessment_worker` instead of uvicorn; everything else
about the image is identical.

Tesseract and the OCR/PDF-parsing dependencies (`pytesseract`, `pymupdf`)
install as Python packages (they're in `pyproject.toml`'s base dependencies)
but the Tesseract *binary* is deliberately not installed in the image: OCR
only runs inside `make parse`, which reads raw PDFs from
`data/policies/raw/` and never runs inside these containers.
`.dockerignore` excludes `eval/`, `data/policies/`, `data/cache/`, and the
dev/VCS caches from the build context for the same reason — none of it is
needed at runtime.

**The retrieval stack is not purely a Postgres client, though.**
`retriever_factory.load_retriever_components()` builds the BM25 lexical leg
and the exclusion-clause graph in-process, straight from
`build/chunks.jsonl` and `build/parsed_clauses.jsonl`
(`docs/LEXICAL_RETRIEVAL.md`, `docs/EXCLUSION_CO_RETRIEVAL.md`) — only the
dense leg queries Postgres. Both files are gitignored, regenerated
artifacts, so rather than bake a snapshot into the image, `api` and `worker`
bind-mount the host's `./build` directory read-only. This also means a
re-`make build-index` on the host (a new corpus, a re-chunk) is picked up by
a container restart, not an image rebuild. One further runtime dependency
lives under `data/`: `data/rag/lexical_stemming_exceptions.csv`, a small,
committed CSV the lexical analyzer's stemmer reads — copied into the image
directly (it's static source, unlike `build/`'s regenerated artifacts), so
`.dockerignore` excludes the rest of `data/` but not `data/rag/`.
`data/cache/embeddings/` and `data/cache/reranker/` are *not* needed at
startup — both caches degrade gracefully to empty when their file is
missing.

**This drives the Quickstart's step order.** `load_retriever_components()`
raises at startup if `build/chunks.jsonl` is missing — by design, the same
fail-fast-with-a-fix shape as the checkpointer probe below — so `api` /
`worker` crash-loop until at least `make build-chunks` (fast: pure local
computation over an already-parsed corpus, no DB, no LLM) has run on the
host. That's why the Quickstart brings up `postgres`/`redis` alone first,
runs `make build-index` against them (which both creates `build/chunks.jsonl`
and loads/embeds the `chunk` table), and only then starts `api`/`worker` —
starting the whole stack in one `up -d --build` before any of that exists
would leave `api`/`worker` crash-looping on a missing file, not merely
degraded.

## Why a one-shot `migrate` service

`build_core_components()` probes the LangGraph checkpointer at API startup
and raises an actionable error if its schema is missing — by design, so a
missing `make setup-checkpointer` reads as a startup failure with a fix, not
a mysterious 500 on the first request (see `presentation/app.py`'s module
docstring). That means `api` / `worker` cannot simply
`depends_on: postgres: condition: service_healthy` — the schema has to exist
first. Compose's `condition: service_completed_successfully` is exactly the
primitive for a one-off init step, so `migrate` runs
`alembic upgrade head && python -m scripts.setup_checkpointer` once
(`restart: "no"`) and `api` / `worker` both depend on it completing before
they start. All three services share one image and one environment shape via
the `&app-build` / `&app-environment` YAML anchors, the same reuse pattern
`compose.yaml` already uses for `langfuse-worker` / `langfuse-web`
(`&langfuse-service` / `&langfuse-env`).

## Environment: `.env` plus three network-address overrides

`api` / `worker` / `migrate` all use `env_file: .env` rather than
re-declaring every `LLM_*` / `EMBEDDING_*` / `ASSESSMENT_*` key in
`compose.yaml` (the way `postgres` does, for example) — the app's own
`Settings` classes already read the exact variable names `.env` uses, so
there is nothing to rename. The one wrinkle: three values mean something
different from *inside* the compose network than from the host, because a
container reaches `postgres` / `redis` / `langfuse-web` by service name, not
by `localhost` or a published port. `compose.yaml`'s `environment:` block
overrides exactly those three keys (`DATABASE_HOST`, `REDIS_URL`,
`LANGFUSE_HOST`) on top of whatever `.env` sets for host-side tools
(`make migrate`, `make build-index`, pytest) to use. `LANGFUSE_HOST` is inert
unless the `tracing` profile is also started and `TRACING_ENABLED=true`.

One caveat this override doesn't cover: if `.env` sets `DATABASE_URL`
directly (its default is empty — `DATABASE_HOST` and friends are the
documented local path), `DatabaseSettings` prefers the full URL and the
`DATABASE_HOST: postgres` override has no effect. That combination — a
custom `DATABASE_URL` together with the bundled Compose `postgres` — isn't a
configuration this stack is meant to support; anyone pointing at an external
database should also point `api` / `worker` there directly.

## Healthchecks

`api`'s healthcheck curls `GET /health` (liveness — "the process is up"),
deliberately not `GET /ready` (readiness — Postgres/Redis/vector-index,
[M5-06]). Compose has no notion of "healthy but degraded": a demo stack with
an empty `chunk` table would leave `/ready` returning 503 indefinitely, and
Compose would report `api` as permanently unhealthy even though the process
is fine and the API is answering requests. `/ready` stays available for
external monitoring and for the reader diagnosing "why is my assessment
coming back `insufficient_information`" — it's just not what gates the
container's own health state.

`worker` has no HTTP surface, so its healthcheck is a plain
`pgrep -f scripts.run_assessment_worker` — "the worker pool process exists".
`procps` (for `pgrep`) and `curl` (for the `api` check) are the only two
packages the runtime image adds over `python:3.12-slim`.

## The `hf_cache` volume

`api` and `worker` both mount a named `hf_cache` volume at
`/root/.cache/huggingface`. The embedder and reranker's combined weights are
about 1.2GB (`docs/EMBEDDINGS.md`) and download on first use; without a
persistent volume, every `docker compose restart` (or a `down`/`up` without
`-v`) would re-download them. This is the same "download once, cache
forever" shape `data/cache/embeddings/` already gives the host-side
`make embed-chunks` path — just a Docker volume instead of a bind mount,
since the container's HF cache and the host's embedding cache are different
things (model weights vs. computed vectors).

## Demo mode: skipping the embedding cost, not inventing a new mechanism

`make build-index`'s `embed-chunks` step is a real cold ~41-minute CPU pass
over the full corpus the first time it runs (`docs/EMBEDDINGS.md`'s "Corpus
embedding cost") — $0 in API charges (the embedder runs in-process, not
behind an API), but a real wait. `scripts/fetch_corpus_artifacts.py` already
solves the identical problem one pipeline stage earlier — a GitHub Release
tarball + checksum that lets a reviewer skip the LLM-cost parsing stage
entirely. `scripts/fetch_embedding_cache.py` / `package_embedding_cache.py`
are the exact same shape of script, one release tag later
(`m3-embedding-cache-v1`), bundling `data/cache/embeddings/cache.jsonl` — the
content-addressed cache `CachingEmbedder` already reads before ever loading
the model. With that cache in place, `embed-chunks` re-fills every vector
from disk in ~2.6 seconds instead of running the model at all. `make
fetch-demo-artifacts` runs both fetches (corpus + embedding cache) in one
step; README's Quickstart uses it.

**What this issue ships, and what it deliberately doesn't (yet).** The
fetch/package scripts and Makefile targets exist and are ready to use; the
`m3-embedding-cache-v1` release itself is not published as part of this
change — publishing it means actually paying the ~41-minute embedding pass
once and running `gh release create` under a real GitHub identity, which
stays a deliberate manual step for whoever maintains the repo, not something
this change does on its own initiative. Until that release exists,
`fetch-embedding-cache` (and therefore `fetch-demo-artifacts`) fails with a
clear "has this release been published?" error, and `make build-index` still
works correctly — it just falls through to the real cold embed.

## Known limitation: shares a Redis queue name with the integration tests

`tests/integration/test_assessment_queue.py` and the Compose `worker` both
enqueue onto the same RQ queue name (`"assessments"`) and, by default,
`TEST_REDIS_URL`/`REDIS_URL` point at the same `redis://localhost:6379/0` —
so a live `worker` container racing to claim a test's fake job (which it
cannot run — the fake job path lives under `tests/`, not in the image) makes
`make test-integration` fail with jobs stuck at `pending` instead of
reaching `failed`/`succeeded`. Confirmed live: `docker compose stop worker`
before running the suite, three tests that failed under a running worker
passed clean. `make test-integration`'s own fixtures use a separate
`insurance_claims_test` *database*, but not a separate Redis *db index* —
stop the Compose `worker` (`docker compose stop worker`) before running it,
or point `TEST_REDIS_URL` at a different db index.

## Known limitation: image size

The built `api`/`worker` image is large (~9GB, measured 2026-09-05) — almost
entirely `torch`'s default CUDA wheels, pulled in transitively by
`sentence-transformers` even though the retrieval stack only ever runs on
CPU (`docs/EMBEDDINGS.md`: "CPU is the conservative headline since a
reproducer is not assumed to have a GPU"). This is not new to the Docker
image — `uv sync --group embed` already resolves the same CUDA wheels for
every host-side `make embed-chunks` / `eval-*` run; the image just makes an
existing project-wide dependency footprint visible in a new place. Trimming
it means pointing `torch` at PyPI's CPU-only wheel index
(`[tool.uv.sources]` in `pyproject.toml`, then re-locking `uv.lock`), which
changes dependency resolution for the whole project, not just the image —
left as a follow-up rather than folded into this issue.

## Enforcement

`docker build .` (also `ci.yml`'s `docker-image` job) proves the image
builds. `docker compose config` proves `compose.yaml` stays valid YAML with
every anchor resolving. There is no automated test that exercises the full
containerised stack end-to-end (that would need Docker-in-CI plus real LLM
credentials); this issue's own DoD asks for a hand-checked run against a
clean clone instead, which is how the two gaps above (`build/chunks.jsonl`,
`data/rag/lexical_stemming_exceptions.csv`) surfaced: both are read directly
off the local filesystem by code that predates this issue, so nothing in a
Postgres-only mental model of the retrieval stack would have caught them.
The live run that found them also completed one, for real: `docker compose
up -d --build` against a Postgres already holding the full 4,540-chunk
embedded corpus, `POST /v1/assessments` on a CASCO collision narrative, and
`GET /v1/assessments/{id}` returning `awaiting_review` with a `compatible`
verdict, four real citations and `is_grounded: true` — the graph's own
happy path, not a synthetic check of the plumbing around it.
