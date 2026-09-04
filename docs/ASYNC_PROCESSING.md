# Asynchronous processing — [M5-05]

A 207-page policy filing is not an HTTP request. Before M5-05, `POST
/v1/assessments` returned **202** but ran the whole LangGraph assessment —
minutes, many LLM calls — synchronously in the request handler. This milestone
puts a Redis-backed queue behind that 202: the request persists a job and
returns; a worker pool runs the graph.

```
POST /v1/assessments ─▶ SubmitClaim ─▶ persist AssessmentJob(PENDING) ─▶ RQ enqueue ─▶ 202 {id, status:"pending"}
                                                                              │
  make worker ── run_assessment_job(id) ──▶ RunAssessment                     ▼
     job ─▶ RUNNING (attempts += 1)
     orchestrator.start(assessment_id=id, claim=<rebuilt from the job row>)
       ├─ paused at the human checkpoint ─▶ one txn: assessments.add(record, AWAITING_REVIEW) + jobs.update(SUCCEEDED)
       ├─ transient provider fault ───────▶ jobs.update(PENDING, failure{transient});  raise TransientAssessmentError ─▶ RQ retries (backoff)
       │                                     …budget exhausted ─▶ jobs.update(FAILED); ─▶ RQ FailedJobRegistry
       └─ real error / contract breach ───▶ jobs.update(FAILED, failure{permanent});  raise PermanentAssessmentError
                                             ─▶ worker zeroes the retry budget ─▶ RQ FailedJobRegistry (traceback preserved)
```

The `202` + `Location` contract does not change. `GET /v1/assessments/{id}`
resolves for the whole lifecycle now — see `docs/API.md`.

## Code

- `app/src/application/assessment_job.py` — `AssessmentJob` / `JobStatus` /
  `JobFailure`. A separate aggregate from `AssessmentRecord`: the record is
  invariant-laden (a non-empty verdict, prose, a ≥1-citation projection) and
  cannot represent a claim that has not been assessed yet.
- `app/src/application/use_cases/submit_claim.py` — persists the `PENDING` job,
  then enqueues. Committed **before** enqueued, so a poller always finds the row.
- `app/src/application/use_cases/run_assessment.py` — `RunAssessment`, the
  worker's half: idempotent (a redelivered `SUCCEEDED`/`FAILED` job is a no-op; a
  job stuck at `RUNNING` re-runs and LangGraph resumes from its last checkpoint).
- `app/src/application/use_cases/get_assessment.py` — returns an
  `AssessmentReadModel` spanning the lifecycle (record if it exists, else the
  job's `pending`/`running`/`failed` state).
- `app/src/infrastructure/llm_errors.py` — `is_transient_llm_error`: the single
  transient-vs-real decision, over the `openai` SDK exception hierarchy, walked
  through the whole `__cause__` chain.
- `app/src/infrastructure/queue/` — `rq_queue.py` (`RqAssessmentQueue`, the port
  adapter), `tasks.py` (`run_assessment_job`, the function RQ runs), `worker.py`
  (`run_worker` + the `stop_retry_on_permanent` exception handler).
- `app/src/infrastructure/bootstrap.py` — `build_core_components`, the heavy
  singletons (DB engine, chat models, retrieval stack, orchestrator) shared by
  the API `lifespan` and the worker.
- `app/src/infrastructure/database/{models,assessment_job_mapper,assessment_job_repository}.py`
  + `alembic/versions/20260904_01_create_assessment_job_table.py` — the
  `assessment_job` table.
- `scripts/run_assessment_worker.py` — `make worker`.

## Bring-up

```bash
docker compose up -d          # postgres + redis
make migrate                  # …now includes assessment_job
make setup-checkpointer
make build-index
make serve                    # the API   (one shell)
make worker                   # the queue (another shell)
```

`make worker` runs under the `embed` uv group like `make serve` — each worker
loads the sentence-transformers embedder and the cross-encoder reranker once.

## Bounded parallelism

`ASSESSMENT_WORKER_CONCURRENCY` (default **2**) is the number of RQ workers. Each
worker runs one assessment at a time, so it is also the count of concurrent graph
runs hitting the LLM provider — the DoD's configurable parallelism bound, the
direct analog of `LLM_CLASSIFICATION_MAX_WORKERS`. Raise it to use more of the
provider's throughput; lower it if the provider rate-limits.

**On the 4 GB-VRAM dev box, set it to 1.** One eval process holding the embedder
+ cross-encoder uses ~3.1 GB of VRAM; two workers on the GPU give `CUDA out of
memory`. `SimpleWorker` (no fork per job) keeps each worker's models loaded for
the life of the process, so a single worker is not reloading them per job.

## Retry, back-off, dead-letter

| failure | classified | what happens |
| --- | --- | --- |
| `openai.RateLimitError` (429), `APITimeoutError`, `APIConnectionError`, `InternalServerError`, any `APIStatusError` 429/5xx | **transient** | job → `PENDING`, `TransientAssessmentError` raised → RQ reschedules after the next `ASSESSMENT_RETRY_BACKOFF_SECONDS` interval (`[30, 120, 300]` by default) |
| the same, on the last of `ASSESSMENT_MAX_RETRIES` (default 3) attempts | transient, exhausted | job → `FAILED`; RQ moves it to `FailedJobRegistry` |
| a 4xx that is not 429, a schema-validation error, a malformed claim, a graph contract breach, a bug | **real** | job → `FAILED` immediately, `PermanentAssessmentError` raised; the worker's `stop_retry_on_permanent` handler zeroes the RQ retry budget so it dead-letters now instead of burning retries |

RQ's `Retry` retries on *any* exception, so the transient/real split is enforced
by the exception type `RunAssessment` raises plus that one exception handler.

**The dead-letter has two faces**, both preserving the cause: the
`assessment_job` row (`status = failed`, `failure.kind` / `failure.message`,
readable through `GET /v1/assessments/{id}` as `error`), and RQ's
`FailedJobRegistry` (the full traceback, for an operator with `rq info` / the
RQ dashboard).

The node-level retry helpers in `infrastructure/graph/nodes/*` are **unchanged**:
they still absorb a mid-run network blip (3× 5 s). A *sustained* 429 now bubbles
past them to the job, which backs off for minutes — which is what
`docs/END_TO_END_EVALUATION.md` observed a rate limit actually needs.

## Deviations on record

- **The list endpoint is record-only.** `GET /v1/assessments` still lists
  completed `AssessmentRecord`s; `pending` / `failed` jobs are not in the
  collection. The per-id `GET` covers "status tracking per assessment id".
- **The commit-then-enqueue window.** If the process dies between the job commit
  and the RQ `enqueue`, the job sits at `PENDING` forever. A reconciler that
  re-enqueues stale `PENDING` jobs is left to operations.
- **A connection pool for the checkpointer** ("the M5 shape", `docs/DATABASE.md`)
  is still deferred — each `RunAssessment` opens its own.
- **`api` / `worker` Compose services + the Dockerfile + the CI image** are
  [M5-09]. M5-05 adds only the `redis` service to `compose.yaml`.

## Results

`tests/integration/test_assessment_queue.py` runs the whole thing against real
Postgres + real Redis, with an in-process burst `SimpleWorker` and the fake graph
context (canned model, one-clause retriever):

- **submission → completion**: `POST` → `202 pending` → `GET` `pending` → drain
  the queue → `GET` `awaiting_review` + `compatible` + a structured citation →
  `POST …/decision {approve}` → `200 decided` → the audit trail ends in
  `human_decision:approve`.
- **transient → recovery**: the first attempt hits a wrapped 429, RQ retries, the
  second attempt completes; the job is *not* in `FailedJobRegistry`.
- **real failure → dead-letter**: a `ValueError` from the run → `GET` `failed`
  with the cause in `error`, and the id present in `FailedJobRegistry`.

`tests/unit/application/use_cases/test_run_assessment.py` covers the state
machine directly (transient-with-budget, transient-exhausted, permanent,
contract breach, idempotent redelivery, transactional rollback);
`tests/unit/infrastructure/test_llm_errors.py` is the classifier truth table.
