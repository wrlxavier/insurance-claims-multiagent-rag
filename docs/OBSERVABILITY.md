# Tracing the assessment graph — [M5-07]

[M5-06] made a run loggable: JSON lines to stdout, one correlation id carried
from the HTTP request through Redis into every node. That answers *what
happened*. It does not answer *which node was wrong*, because that question
needs a node's input beside its output beside the clauses it was reasoning over
— a tree, not a sequence of lines. M5-07 adds a self-hosted **Langfuse** to the
Compose stack and traces the graph into it.

```
POST /v1/assessments (X-Correlation-ID: abc)
  └─ trace "claim-assessment"          session_id = assessment id, tags = [abc]
     ├─ intake                         chain  · state in → state delta out
     │   └─ ChatOpenAI                 GEN    · prompt, completion, tokens, cost
     ├─ retrieval                      chain
     │   └─ retrieval                  SPAN   · query, filter, k, candidates+scores, gate
     │       └─ retrieval.rerank       SPAN   · hybrid rank vs cross-encoder rank
     ├─ compatibility                  chain
     │   └─ ChatOpenAI                 GEN    · the reasoning model
     ├─ consistency                    chain
     │   └─ ChatOpenAI                 GEN
     ├─ recommendation                 chain
     │   └─ ChatOpenAI                 GEN
     └─ human_review                   chain  · ends on interrupt()
```

Everything but the two `SPAN` rows is produced by installing one LangChain
callback handler on the graph's run config. `build.py`, the eight node modules
and the `(state, runtime) -> dict` convention are untouched by this issue.

## Code

- `app/src/infrastructure/observability/tracing.py` — the whole application
  side: `build_tracer` (the switch), `NullTracer` (off), `LangfuseTracer`
  (`callbacks()` / `assessment_run()` / `span()` / `shutdown()`), and
  `register_model_prices`.
- `app/src/infrastructure/graph/context.py` — `TracePort`, the capability a node
  depends on, and `NO_TRACING`, the null object that is `GraphContext.tracer`'s
  default.
- `app/src/infrastructure/graph/orchestrator.py` — the one wiring point:
  `config["callbacks"]`, the root span, the flush.
- `app/src/infrastructure/graph/nodes/retrieval.py` — the explicit retrieval
  span.
- `app/src/infrastructure/rag/graph_retrieval_adapter.py` — the nested
  `retrieval.rerank` span (`SpanRecorder`, the port restated so `rag` keeps not
  importing `graph`).
- `app/src/infrastructure/bootstrap.py` — one client per process, shut down with
  the engine.
- `compose.yaml` — the `tracing` profile: `langfuse-web`, `langfuse-worker`,
  `clickhouse`, `minio`.
- `docker/postgres/initdb/02-create-langfuse-database.sql` — Langfuse's database
  on the existing Postgres.

## Bring-up

```bash
cp .env.example .env                       # then fill the five values listed below
docker compose --profile tracing up -d     # postgres + redis + the 4 langfuse services
make migrate && make setup-checkpointer
make serve                                 # one shell
make worker                                # another
```

Open <http://localhost:3000> and sign in with `LANGFUSE_INIT_USER_EMAIL` /
`LANGFUSE_INIT_USER_PASSWORD`.

**Five `.env` values have no safe default**, and the `tracing` profile refuses to
start without them — each with an error naming the variable, so you find out at
`up`, not at the first missing trace:

```bash
openssl rand -hex 32   # LANGFUSE_NEXTAUTH_SECRET
openssl rand -hex 32   # LANGFUSE_SALT
openssl rand -hex 32   # LANGFUSE_ENCRYPTION_KEY
# plus any non-empty pair, `pk-lf-…` / `sk-lf-…` by convention:
#   LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
```

The key pair is **seeded into the server**, not read from it: compose passes it
as `LANGFUSE_INIT_PROJECT_PUBLIC_KEY` / `_SECRET_KEY`, so on a fresh volume the
project is created holding exactly the keys the application is already
configured with. There is no key-copying step and no click-through.

Leaving the pair empty is the supported "no tracing" state, and it is what
`.env.example` ships: `docker compose up -d` (no profile) then brings up
postgres and redis exactly as before, and the service runs untraced.

**On an existing Postgres volume**, `initdb` will not re-run, so create the
database by hand once:

```bash
docker compose exec postgres createdb -U postgres langfuse
```

## What is traced, and by whom

**The callback handler does the bulk.** `LangGraph` nodes and the
`chain.invoke(messages)` calls inside them are LangChain runnables, so a handler
on the run's config sees all of them. Each node becomes a span carrying the
state it read and the delta it returned; each LLM call becomes a *generation*
carrying the prompt, the completion, the model id and `usage_details`.

Attaching it in `orchestrator._invoke` rather than to the model objects is
deliberate: the nodes pass no `config` to their own `chain.invoke`, relying on
LangChain's ambient child config, so a handler bound to a model would be
bypassed while one bound to the run is not.

**Two spans are added by hand**, because no LLM call passes through them:

- **`retrieval`** (the node) — the query built from the entities, the metadata
  pre-filter, `k`, then every returned clause with its id, document, type and
  score, and the whole gate result. Three of the gate's fields (`threshold`,
  `missing_category`, `closest_clause_ids`) are computed on every run and
  recorded nowhere else: state keeps only the boolean, and the audit event keeps
  only the trigger name.
- **`retrieval.rerank`** (the adapter) — each candidate's hybrid rank beside its
  cross-encoder rank and score, plus anything exclusion co-retrieval injected.
  `RERANK_CANDIDATE_DEPTH` is 10, the same as the node's `k`, so the reranker
  reorders rather than prunes; the question this span answers is not "what was
  dropped" but "did the hybrid legs surface the right clause at all".

**Cost** needs one more thing. Langfuse prices a generation by matching the
reported model name against the model definitions in the project, and it has
never heard of an OpenRouter id like `deepseek/deepseek-v4-flash-0731`. So
`register_model_prices` upserts one definition per model at startup from
`LLM_{FAST,REASONING}_{INPUT,OUTPUT}_COST_PER_1M_TOKENS_USD`. Without it every
generation costs `0.00`.

Those prices are **per pinned provider route**, not per model: the same model is
served at $0.05/1M input on one OpenRouter route and $0.35 on another. Re-pin
`LLM_FAST_PROVIDER_ORDER` and the price goes stale. They are list prices, not a
measurement — [M5-10] owns measured cost per assessment.

## Reaching a trace from a log line, and back

Both directions work, which is the point of tagging:

- **log → trace.** Every run logs one `trace.started` line carrying `trace_id`
  and a ready-made `trace_url`, beside the `correlation_id`. Open the URL.
- **trace → log.** The trace carries the correlation id twice: as a **tag** and
  in `metadata.correlation_id`. Paste it into Langfuse's search to find the
  trace for a correlation id you already have from a log line.
- `session_id` is the assessment id, so the run up to the human checkpoint and
  the run resumed after the decision group as one session.

The trace URL is resolved **once, at startup**, not per run: the SDK's
`get_trace_url()` fetches the project id over HTTP, and a blocking round trip
per assessment — one that fails slowly exactly when Langfuse is down — is not
something observability is allowed to cost.

## Turning it off

```bash
TRACING_ENABLED=false      # or leave LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY empty
```

`ObservabilitySettings.tracing_active` is the flag **and** both keys.
Inactive, `build_tracer` returns `NullTracer` and no Langfuse client is
constructed at all — no exporter thread, no network. The graph runs the
identical code path; `GraphContext.tracer` is a null object, so nodes do not
branch on whether tracing is on.

Tracing is also optional at the infrastructure layer: the langfuse services sit
behind a Compose **profile**, so plain `docker compose up -d` is still exactly
postgres + redis.

Two consequences worth stating plainly:

- **A tracing failure is never a run failure.** Every call into the SDK is
  guarded; a broken or unreachable Langfuse logs a warning and the assessment
  proceeds untraced. `test_a_broken_tracer_does_not_break_the_body_it_wraps`
  holds this.
- **`GET /ready` does not check Langfuse.** Readiness means "can serve an
  assessment", and an assessment does not need a tracer. Adding it would make an
  optional dependency able to take the service out of rotation.

## Reading a trace

Open the trace and read the span tree top-down; the first place the run went
wrong is usually visible before any prompt is read.

1. **Did retrieval see the right document?** The `retrieval` span's `filter`
   shows what constrained the search — a stated SUSEP process, else the product
   line, else nothing. `candidates[].document_id` shows what came back.
2. **Did it see the right *clause*?** `candidates[].clause_id` with scores. The
   right document with the wrong clauses is a different bug from the wrong
   document, and this is the only place the distinction is visible.
3. **What did the gate decide, and why?** `gate.sufficient` with `top_score`
   against `threshold`. A gate that says *sufficient* over clauses that do not
   answer the question is the setup for a confident wrong verdict.
4. **What did the reasoning model actually see?** The `compatibility`
   generation's prompt contains the clause text that was passed in. If step 2
   showed the right clause missing, this is where you confirm the model reasoned
   correctly over the wrong evidence.
5. **Latency and cost** sit on each span and total on the trace.

## Worked example: a confident wrong verdict

Run on 2026-09-04 through `make serve` + `make worker` against the compose
`tracing` profile, on `compatible-007` from `data/synthetic_claims/claims.jsonl`
— a claimant who swerved an e-scooter to avoid a pedestrian and hit a parked
car, filed against SUSEP process `15414.614145/2021-31` (document 18). The
golden label is **compatible**. The system said **incompatible**, confidence
0.85, and this run reproduces the M4-10 result exactly
(`docs/END_TO_END_EVALUATION.md` classifies it `retrieval_miss`).

The trace, `claim-assessment`, 234.6 s wall clock:

| span | type | latency | in | out | cost |
|---|---|---:|---:|---:|---:|
| `intake` | chain | 61.26 s | | | |
| ⟶ `ChatOpenAI` | generation | 61.25 s | 1069 | 4 (+1773 reasoning) | $0.000086 |
| `retrieval` | chain | 2.39 s | | | |
| ⟶ `retrieval` | **span** | 2.39 s | | | |
| ⟶ `retrieval.rerank` | **span** | 2.39 s | | | |
| `compatibility` | chain | 51.38 s | | | |
| ⟶ `ChatOpenAI` | generation | 51.36 s | 3022 | 271 (+3725 reasoning) | $0.004303 |
| `consistency` | chain | 79.32 s | | | |
| ⟶ `ChatOpenAI` | generation | 79.30 s | 846 | 225 | $0.000108 |
| `recommendation` | chain | 91.55 s | | | |
| ⟶ `ChatOpenAI` | generation | 91.54 s | 1387 | 264 | $0.000158 |
| `human_review` | chain | 0.002 s | | | |

**The cost column is what this run recorded, and it is an undercount** — the run
predates the `output_reasoning` pricing tier described in Finding 4, so it prices
`input` and `output` only. Corrected, the compatibility call is ~$0.0168 rather
than $0.0043 and the run totals ~$0.0176 rather than $0.0047. The latencies and
token counts are as measured. Route matters too: this run pinned the fast model
to `deepinfra/fp8` ($0.08/$0.18 per 1M) and the reasoning model to `alibaba`
($1.122/$3.366), with `LLM_*_COST_PER_1M_TOKENS_USD` set to match — the repo
defaults price the default pins instead.

Reading it in the order of "Reading a trace" above:

**1. The right document was searched.** The `retrieval` span's input:

```json
{"query": "RCF-A O segurado, andando de patinete elétrico, desviou para evitar
           um pedestre e colidiu com um carro estacionado...",
 "k": 10,
 "filter": {"susep_process": "15414.614145/2021-31", "product_line": null}}
```

The stated process filtered the search to document 18 — the [M4-10] filter
behaving exactly as intended. Every candidate that came back is from document
18. So this is **not** a wrong-document bug, and one glance at this span rules
that whole class out.

**2. The right clause was never retrieved.** The span's `candidates`:

| rank | clause | type | score |
|---:|---|---|---:|
| 1 | `18:3/3.11` | definition | 0.5074 |
| 2 | `18:9/12.3` | **exclusion** | 0.4878 |
| 3 | `18:3/3.39` | definition | 0.4580 |
| 4–10 | `18:3`, `18:3/3.6`, `3.7`, `3.9`, `3.10`, `3.21`, `3.23` | definition | 0.2681 (all seven identical) |

The claim's reference clause is **`18:11`**, and it is not there. Nine of the ten
retrieved clauses are *definitions*; the tenth is an *exclusion*. **Not one
coverage clause was retrieved.** The seven-way tie at exactly 0.2681 is the
cross-encoder saying it cannot tell those apart at all.

`retrieval.rerank` shows the reranker did not cause this — it moved
`18:9/12.3` from hybrid rank 3 to rank 2 and `18:3/3.39` from 2 to 3, and
changed nothing else. The clause was never in the hybrid candidate set to begin
with, so the fix belongs in retrieval, not reranking.

**3. The gate passed anyway, by 0.047.** The span's `gate`:

```json
{"sufficient": true, "trigger": null, "top_score": 0.5074, "threshold": 0.46}
```

This is the finding the trace makes unmissable. The [M3-07] gate is a threshold
on the top reranked score, and 0.5074 clears 0.46 — on a *definition* clause.
The gate measures whether the top hit looks relevant, not whether the retrieved
set can answer the question, and a set of nine definitions plus one exclusion
cannot. `top_score` beside `threshold` in one object is what turns "the gate
said sufficient" into "the gate said sufficient by 0.047, on a definition".

**4. The model reasoned correctly over the wrong evidence.** The
`compatibility` generation's completion:

```json
{"assertions": [{"clause_ids": ["18:3/3.39"],
                 "statement": "A cláusula 3.39 define expressamente 'patinetes'
                               como equipamento de mobilidade individual..."}]}
```

and the final recommendation cites `18:9/12.3` and `18:3/3.39` — the exclusion
and the definition that feeds it. Given only an exclusion and definitions, an
`incompatible` verdict is the *correct* reading of that context. The reasoning
node is not the bug.

**The diagnosis, in one line:** retrieval returned no coverage clause, the gate
could not tell that from a good result, and the assessment then faithfully
applied the only rule it was shown — an exclusion. The fix is in the gate (a
composition signal beside the score threshold: does the retrieved set contain a
coverage clause at all?) and in retrieval recall for `18:11`, and **not** in the
compatibility prompt, which is where a reader without the trace would have
started, because the wrong output came out of that node.

Two numbers the trace also settles in passing: the parallel branch saved 51.4 s
of the run's 234.6 s (`compatibility` 51.38 s ran entirely inside `consistency`'s
79.32 s), and the single reasoning call is **92% of the run's cost** — the other
three LLM calls together came to $0.00035 against its $0.0043. That ratio is a
sharper argument for the [M4-07] parallel branch than the wall-clock saving is:
the expensive call is the one on the critical path either way.

## Findings

1. **The callback handler covers "trace every node" with no graph change.**
   Every one of the six nodes on the executed path appears as a span with its
   input, output and latency, and every LLM call as a generation with prompt,
   completion and token usage — from one line in `orchestrator._invoke`.
   `build.py`, the node modules and the [M4-01b] node convention are untouched.
2. **Retrieval needed a hand-written span, and it is the one that pays.** No LLM
   call passes through the retrieval node, so callbacks see a black box. Every
   step of the diagnosis above came from the two hand-written spans; the four
   generations only confirmed the conclusion.
3. **Three gate fields existed but were unrecorded.** `threshold`,
   `missing_category` and `closest_clause_ids` are computed on every run and
   were reaching neither state nor the audit trail. `top_score: 0.5074` against
   `threshold: 0.46` is the single most useful number in this trace, and before
   M5-07 it was discarded microseconds after being computed.
4. **Cost needed model definitions, and reasoning tokens needed a pricing
   tier.** Langfuse prices from a project model definition and knows no
   OpenRouter model id, so every generation costs `0.00` without
   `register_model_prices`. Worse, the obvious flat `input_price`/`output_price`
   pair prices only the `input` and `output` usage keys — and this reasoning
   model reported **3725 of ~4000 completion tokens** under `output_reasoning`.
   Priced flat, the compatibility call reads $0.0043; with `output_reasoning`
   priced at the completion rate it is ~$0.017, a **4x** understatement. The
   registration sends a pricing tier for that reason.
5. **The trace-URL lookup was a hidden per-run HTTP call.** `get_trace_url()`
   fetches the project id over the network. Called per assessment it would add a
   blocking round trip to every run — one that fails *slowly* exactly when
   Langfuse is down. It is resolved once at startup and formatted locally after.
6. **Idempotent price registration needed a lookup, not just a stable id.**
   Model *names* are unique per project, so an upsert under our own id is
   rejected with "already exists in project" whenever a definition was created
   under a Langfuse-assigned id. Registration lists the project's definitions
   (paged — a fresh project ships ~180) and updates whatever id the name
   actually has.
7. **Langfuse v4 self-hosting is four containers, not one.** `langfuse-web`,
   `langfuse-worker`, ClickHouse and MinIO, because the v4 SDK exports over
   OpenTelemetry to a v4 server. It reuses this stack's Postgres and Redis, and
   the whole set sits behind a Compose profile so the default stack is unchanged.
8. **The v2 SDK was not an option.** `langfuse<3`'s LangChain integration
   hard-imports the `langchain` meta-package *and* `langchain.schema.*`, removed
   in langchain 1.x; the version that has it pins `langchain-core <1`, which this
   project is past. The v4 handler imports from `langchain_core` — but still
   does `import langchain` to read `__version__`, which is why the meta-package
   is now a direct dependency.

## What this leaves for later

- **Measured latency and cost** are [M5-10]. M5-07 makes the numbers visible per
  node and per run; it does not report a p50/p95 or a cost per assessment, and
  the prices registered here are list prices for the pinned route.
- **No integration test needs a Langfuse server.** The span assertions in
  `tests/unit/infrastructure/observability/test_tracing.py` run the real graph
  against a real Langfuse client whose exporter writes to memory, so CI proves
  "every node appears as a span" with no server and no credentials. Keeping the
  `tracing` profile out of CI is the point of it being a profile.
- **The langfuse services have no resource limits** and share this stack's
  Postgres and Redis (its own database, and Redis db index 1 against the queue's
  db 0). Fine for a development and demo stack, stated rather than hidden.
- **Sampling is not configured.** Every run is traced. The SDK supports a sample
  rate; at this project's volume there is nothing to sample.
