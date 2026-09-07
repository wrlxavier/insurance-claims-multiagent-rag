# Getting started

Everything needed to clone this repository and reproduce its behaviour on a
clean machine — Linux, macOS or Windows. It covers prerequisites, `.env`
configuration, running the stack through Docker Compose (and the bare-metal dev
loop), managing the containers, and a full worked session against the API.

The design rationale for the Compose stack is separate, in
[`DEPLOYMENT.md`](DEPLOYMENT.md); the endpoint reference is [`API.md`](API.md).
This document is the *how*.

---

## 1. What you get

Two application processes — an **API** (`FastAPI`/`uvicorn`) and a **worker** —
sharing one image, backed by **Postgres + pgvector** (the dense retrieval index,
the LangGraph checkpointer, the audit trail) and **Redis** (the assessment
queue). You submit a claim narrative to the API; a worker runs the LangGraph
pipeline (intake → clarification loop → hybrid retrieval → parallel
compatibility/consistency assessment → recommendation → a human checkpoint); you
poll for the result, submit a decision, and read the audit trail.

---

## 2. Prerequisites

**All platforms**

- **Docker Engine 24+ with Compose v2** (`docker compose`, not the legacy
  `docker-compose`). Docker Desktop includes both.
- **git** and **GNU `make`**.
- An **LLM API key** for an OpenAI-compatible endpoint (see §3). Nothing runs
  without one.

**Linux** — Docker Engine from your distro or Docker's apt/yum repo; `make` from
the distro package (`build-essential` / `make`).

**macOS** — Docker Desktop (Apple Silicon and Intel both work; the image base is
multi-arch). `make` ships with the Xcode Command Line Tools (`xcode-select
--install`). The built image is large (~9 GB, mostly PyTorch CUDA wheels that go
unused on CPU) — make sure Docker Desktop's disk allocation has room.

**Windows** — Docker Desktop with the **WSL2 backend**, and run every command
from a **WSL2 shell** (e.g. Ubuntu), where `make` is available
(`sudo apt install make`). Git Bash on its own is not enough: it has no `make`,
and bind-mount path translation is unreliable. Clone the repo *inside* the WSL2
filesystem, not under `/mnt/c`, or the `./build` bind mount will be slow.

**Only for `make parse` or the bare-metal dev loop** (not needed for the Compose
path): [`uv`](https://docs.astral.sh/uv/), Python 3.12, and — for the two
policy PDFs with no text layer — `tesseract-ocr` plus `tesseract-ocr-por`.

---

## 3. Configure `.env`

```bash
cp .env.example .env
```

`.env.example` is annotated in full. The defaults already match the bundled
Compose services, so the only values you must supply are the LLM ones.

### Must set

| Key | What it is |
| --- | --- |
| `LLM_PROVIDER` | the client family — `openai` for any OpenAI-compatible gateway (OpenRouter, a local proxy, …) |
| `LLM_BASE_URL` | the gateway base URL, e.g. `https://openrouter.ai/api/v1` |
| `LLM_API_KEY` | your key for that gateway |
| `LLM_MODEL_FAST` | the fast model — intake, clarification, consistency, the recommendation prose |
| `LLM_MODEL_REASONING` | the reasoning model — the compatibility verdict |

If your gateway is **OpenRouter**, also pin the provider route for each model —
the built-in defaults (`["baidu/fp8"]` for fast, `["streamlake"]` for reasoning)
can return HTTP 404 for a model that route does not serve:

| Key | Example |
| --- | --- |
| `LLM_FAST_PROVIDER_ORDER` | `["streamlake/fp8"]` |
| `LLM_REASONING_PROVIDER_ORDER` | `["novita/fp8"]` |

A concrete, known-good configuration (the one behind
[`PERFORMANCE.md`](PERFORMANCE.md)):

```dotenv
LLM_PROVIDER=openai
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-...
LLM_MODEL_FAST=deepseek/deepseek-v4-flash-0731
LLM_MODEL_REASONING=deepseek/deepseek-v4-pro-0813
LLM_FAST_PROVIDER_ORDER=["streamlake/fp8"]
LLM_REASONING_PROVIDER_ORDER=["novita/fp8"]
```

`LLM_MODEL_VISION` (e.g. `google/gemini-3.8-flash`) is **optional** — it is only
used by `make escalate-vision-boundaries` and
`make validate-parsing-quality-sample`, neither of which is on the assessment
path.

### Leave as-is

`DATABASE_HOST` / `DATABASE_PORT` / `DATABASE_USER` / `DATABASE_PASSWORD` /
`DATABASE_NAME`, `REDIS_URL`, `TEST_DATABASE_URL` / `TEST_REDIS_URL`,
`API_PORT` (`8000`), `LOG_FORMAT` (`json`; set `text` for readable local lines),
`EMBEDDING_MODEL`, `RERANKER_MODEL`, and the `ASSESSMENT_*` queue knobs all have
working defaults. Inside the Compose network the app reaches Postgres, Redis and
Langfuse by service name, so `compose.yaml` overrides `DATABASE_HOST`,
`REDIS_URL` and `LANGFUSE_HOST` for the containers automatically —
[`DEPLOYMENT.md`](DEPLOYMENT.md) explains the split.

> **Do not set `DATABASE_URL`.** It is blank by default on purpose. If it is
> set, `DatabaseSettings` prefers it and the Compose `DATABASE_HOST: postgres`
> override stops working — the containers then try to reach Postgres at
> `localhost` and fail. Use the discrete `DATABASE_*` keys.

---

## 4. Run the stack — Docker Compose (recommended)

From a clean clone, with `.env` filled in:

```bash
# 1. Infrastructure first. api/worker read data files at startup and will
#    crash-loop if they start before the index exists.
docker compose up -d postgres redis

# 2. Get the pre-processed corpus (skips the LLM parsing cost — ~10 MB download,
#    no .env needed).
make fetch-corpus-artifacts

# 3. Build the searchable index against the running Postgres:
#    chunk -> load into Postgres -> embed.
make build-index

# 4. Build the image, run migrations once, start api + worker.
docker compose up -d --build
```

**About step 3.** The embedding pass runs a local model (no API cost) but a cold
run over the full corpus is ~41 minutes of CPU time. To skip it, run
`make fetch-embedding-cache` (or `make fetch-demo-artifacts`, which does step 2
and the cache in one) *before* `make build-index` — with the cache in place the
embed step re-fills every vector from disk in seconds. That download depends on
a published `m3-embedding-cache-v1` release; until the maintainer publishes it,
the fetch fails with a clear message and `make build-index` falls through to the
real cold embed ([`DEPLOYMENT.md`](DEPLOYMENT.md), "Demo mode").

**About step 4.** `docker compose up -d --build` builds
`insurance-claims-assessment:local`, runs the one-shot `migrate` service
(`alembic upgrade head` + the LangGraph checkpointer's own schema) to
completion, then starts `api` and `worker`. Both wait on `migrate` succeeding
and on Postgres/Redis being healthy.

### Verify

```bash
docker compose ps                       # migrate = exited (0); api, worker = healthy
curl -s localhost:8000/health           # {"status":"ok"}
curl -s localhost:8000/ready            # 200 once the chunk table has embedded rows
```

`GET /ready` returns **503** with `"vector_index": {"status": "error", ...}`
while the `chunk` table is empty — that is expected before `make build-index`
has loaded and embedded the corpus, and it is why the container healthcheck uses
`/health`, not `/ready`.

---

## 5. Manage the containers

| Task | Command |
| --- | --- |
| Status | `docker compose ps` |
| Follow logs | `docker compose logs -f api` · `docker compose logs -f worker` |
| Restart one service | `docker compose restart api` |
| Pick up a re-`make build-index` (new corpus) | `docker compose restart api worker` — `./build` is bind-mounted, no rebuild needed |
| Rebuild after a code change | `docker compose up -d --build` |
| Stop everything (keep data) | `docker compose stop` |
| Tear down (keep volumes) | `docker compose down` |
| Tear down **and wipe** DB + model cache | `docker compose down -v` — next start re-downloads ~1.2 GB of model weights |
| Add self-hosted tracing | `docker compose --profile tracing up -d` (see §8) |

**Before `make test-integration`, stop the worker.** The Compose `worker` and
the integration tests share the Redis queue name `assessments`; a running worker
races the tests for their fake jobs and they hang. `docker compose stop worker`
first ([`DEPLOYMENT.md`](DEPLOYMENT.md), "Known limitation").

The ~9 GB image is expected — it is PyTorch's default CUDA wheels, pulled in
transitively even though retrieval runs on CPU.

---

## 6. Run the stack — bare-metal dev loop

For hot-reload development, run only the infrastructure in containers and the
app on the host. Do **not** run this and the containerised `api`/`worker` at the
same time — they bind the same port.

```bash
docker compose up -d postgres redis
make migrate                # Alembic: assessment / decision / audit_event tables
make setup-checkpointer     # the LangGraph checkpointer's own tables (outside Alembic)
make build-index            # same index build as §4 step 3
make serve                  # uvicorn presentation.app:app --reload --port 8000
make worker                 # the queue workers — in a second shell
```

`make serve` and `make worker` run under the optional `embed` uv group
(`sentence-transformers` + `torch`); the first run installs it. Both fail fast
with an actionable message if the checkpointer schema or the embedded chunk
corpus is missing.

---

## 7. Use the API

Full endpoint reference, the error envelope and the `code` table are in
[`API.md`](API.md). A complete session:

### Submit a claim

```bash
curl -s -i -X POST localhost:8000/v1/assessments \
  -H 'Content-Type: application/json' \
  -d '{
        "raw_text": "Ao dar ré na garagem bati a traseira do meu carro no portão. Quero acionar o seguro para o conserto.",
        "policy_ref": "15414.610650/2024-59",
        "claim_id": "CLAIM-2026-0007"
      }'
```

- `raw_text` — **required**, the claimant's narrative.
- `policy_ref` — optional SUSEP process number (canonical `NNNNN.NNNNNN/NNNN-NN`
  or 17 digits). It is stored on the job and prepended to the narrative as
  `[Apólice registrada: processo SUSEP …]` so intake extracts it and retrieval
  pre-filters on the right product.
- `claim_id` — optional external id; one is minted otherwise.

Response — **202 Accepted**, `Location: /v1/assessments/<id>`:

```json
{ "assessment_id": "a1b2c3d4-...", "status": "pending" }
```

### Poll for the result

```bash
curl -s localhost:8000/v1/assessments/a1b2c3d4-...
```

`status` moves `pending → running → awaiting_review` (or `failed`). While it is
not ready:

```json
{ "assessment_id": "a1b2c3d4-...", "claim_id": "CLAIM-2026-0007",
  "status": "running", "error": null, "verdict": null, "citations": [],
  "is_grounded": false, "created_at": "2026-09-07T12:00:00Z", "decision": null }
```

Once `status` is `awaiting_review` it is the full aggregate:

```json
{ "assessment_id": "a1b2c3d4-...", "claim_id": "CLAIM-2026-0007",
  "status": "awaiting_review", "error": null,
  "verdict": "compatible",
  "reasoning": "O evento descrito — colisão da traseira do veículo do segurado ...",
  "recommended_action": "Encaminhar para análise de cobertura de danos ao veículo.",
  "confidence": 0.72, "is_grounded": true,
  "citations": [
    { "clause_id": "casco-...-3.1", "document_id": "15414610650202459",
      "susep_process": "15414.610650/2024-59", "clause_type": "coverage",
      "excerpt": "Garante ao Segurado ... colisão, abalroamento ...",
      "relevance_score": 0.83 }
  ],
  "consistency_flags": [],
  "context_sufficient": true, "clarification_exhausted": false,
  "missing_information": [],
  "created_at": "2026-09-07T12:00:00Z", "decision": null }
```

`status` is one of `pending`, `running`, `awaiting_review`, `decided`, `failed`.
A `failed` job carries the cause in `error`.

### Submit the human decision

```bash
curl -s -X POST localhost:8000/v1/assessments/a1b2c3d4-.../decision \
  -H 'Content-Type: application/json' \
  -d '{ "decision": "approve", "notes": "Conferido contra as condições gerais." }'
```

`decision` is `approve`, `edit` or `reject`. For `edit`, add an `edited` object
carrying `verdict` / `reasoning` / `recommended_action` / `confidence` /
`citations` (every cited clause is validated against the corpus before the graph
resumes). The system's own verdict, prose and citations are **never
overwritten** — an edit lives in `decision.edited_assessment`. Returns
**200 OK** with the settled aggregate; a second decision returns **409**.

### Read the audit trail

```bash
curl -s localhost:8000/v1/assessments/a1b2c3d4-.../audit
```

`{"entries": []}` until a decision is submitted — the durable trail is written
once, at the human checkpoint. After that it is the per-node record ending in
`human_review` / `human_decision:approve` with the decision in `payload`.

### List completed assessments

```bash
curl -s 'localhost:8000/v1/assessments?status=awaiting_review&limit=20'
```

Lists `awaiting_review` / `decided` only (not queued or failed), newest first;
accepts `claim_id`, `status`, `limit`, `offset`.

### Correlation ids

Send `X-Correlation-ID: <anything>` and it is echoed on the response and tagged
on every log line the request and its worker run emit — `docker compose logs
worker | grep <id>` follows one claim end to end. See
[`OBSERVABILITY.md`](OBSERVABILITY.md).

---

## 8. Optional components

**Tracing (self-hosted Langfuse).** Set the three secrets in `.env`
(`openssl rand -hex 32` each): `LANGFUSE_NEXTAUTH_SECRET`, `LANGFUSE_SALT`,
`LANGFUSE_ENCRYPTION_KEY`. Then `docker compose --profile tracing up -d` adds
`langfuse-web` / `langfuse-worker` / `clickhouse` / `minio` and opens the UI on
`http://localhost:3000`. The project is seeded with the `LANGFUSE_*_KEY` values
already in `.env`, so there is no key-copying step. Reading a trace, and a worked
example of diagnosing a wrong verdict, are in [`OBSERVABILITY.md`](OBSERVABILITY.md).

**Prompt-injection classifier.** An advisory-only runtime classifier, **off by
default**. On this project's Portuguese corpus it is *not* recommended — the
pinned model is English-only and flags ~70% of real policy clauses. Kept as a
working reference for the pattern. Method and numbers:
[`PROMPT_INJECTION_CLASSIFIER.md`](PROMPT_INJECTION_CLASSIFIER.md).

**Pre-commit hooks.**

```bash
uv run pre-commit install
uv run pre-commit run --all-files   # check the current tree once
```

Runs lint, format and type checks and strips notebook output. See
[`../CONTRIBUTING.md`](../CONTRIBUTING.md).

**Jupyter kernel.** `./scripts/setup_dev_kernel.sh` registers a project-bound
kernel ("Insurance Claims (uv)") for the notebooks under `notebooks/`.

---

## 9. Inspect the parsed corpus without running the pipeline

To read the published parsing result — the 4,925-clause corpus and the LLM
caches behind it — without spending tokens or time:

```bash
make fetch-corpus-artifacts
```

A ~10 MB release asset, no `.env` needed. [`PARSING.md`](PARSING.md) has the
accuracy numbers this corpus was measured against. A later local `make parse`
reproduces it byte-identically except for `build/manifest.json`'s `built_at_utc`
timestamp.

---

## 10. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `api` / `worker` restart continuously right after `up` | The index files are missing. Run `make build-index` on the host (it creates `build/chunks.jsonl`), then `docker compose up -d`. |
| `GET /ready` → 503, `vector_index` error | The `chunk` table has no embedded rows. Finish `make build-index` (the `embed-chunks` stage). |
| Assessments always return `insufficient_information` | Check `/ready` first. If the index is fine, the retrieval pre-filter may be selecting nothing — see [`RETRIEVAL_NODE.md`](RETRIEVAL_NODE.md). |
| Many claims `failed` with a provider `429` | The pinned provider is rate-limiting a sustained run. Use a quieter route, or raise `ASSESSMENT_RETRY_BACKOFF_SECONDS` / `ASSESSMENT_MAX_RETRIES` in `.env` ([`END_TO_END_EVALUATION.md`](END_TO_END_EVALUATION.md), arm 3). |
| ~10% of a batch `failed` at intake | The fast model returned empty structured output on that route — a known robustness gap ([`PERFORMANCE.md`](PERFORMANCE.md), finding 6). A different fast route usually recovers them. |
| `make test-integration` hangs at `pending` | A Compose `worker` is draining the test queue. `docker compose stop worker` first. |
| `DATABASE_HOST: postgres` seems ignored | `DATABASE_URL` is set in `.env`. Clear it. |

---

## 11. Verify your setup

```bash
make check                                    # lint + format-check + typecheck + unit tests
docker compose stop worker && make test-integration   # DB-backed tests against Postgres + Redis
```

Retrieval quality is reproducible offline once the index is built:

```bash
make eval-retrieval-matrix    # regenerates the docs/RETRIEVAL_BENCHMARK.md table
```
