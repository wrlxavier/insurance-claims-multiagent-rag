# Performance: latency and cost per assessment — [M5-10]

Two numbers every reviewer asks first, and most portfolio projects cannot
answer: how long one claim assessment takes, and what it costs in tokens. Every
M4 document measured *quality*; this one measures time and money, over the same
51 synthetic claims, by running the **whole compiled graph** per claim and
watching it from the outside.

M5 says "latency and cost stop being unknowns and become measured numbers"; M5's
exit criterion names "p95 latency and mean cost per assessment measured and
recorded". [M5-07] already made both visible *per run* in a Langfuse trace
(`docs/OBSERVABILITY.md`) — it does not aggregate a p50/p95 or a
cost-per-assessment, and the prices it registers are OpenRouter list prices, not
a measurement. This document is the measurement.

Code:

- `scripts/eval_performance.py` — the runner (`make eval-performance`). Runs
  `build_claim_graph()` over every synthetic claim (policy-header arm, the
  checkpoint resumed with an automatic `approve`, exactly as
  `scripts/eval_end_to_end.py`).
- `app/src/infrastructure/evaluation/latency_stats.py` — the nearest-rank
  percentile / `summarise_latency`, shared with the retrieval benchmark scripts.
- `app/src/infrastructure/evaluation/token_cost.py` — the per-call pricing, kept
  numerically in step with [M5-07]'s `tracing.model_price_definitions` by a
  drift-guard test.
- Raw regenerable run: `eval/runs/performance.{md,json}` +
  `eval/runs/performance_per_claim.jsonl`. Committed snapshot:
  `eval/performance.json`.

**Scope.** This issue measures; it does not tune. It also does not change
`TokenUsage`, the node `_token_usage` helpers or the `audit_event` table — the
measurement is taken from a callback handler outside the graph. Two audit-trail
fidelity gaps it *found* are filed as follow-ups — see *Handler vs the audit
trail*, below.

---

## Method

- **The claim set.** All 51 synthetic claims —
  `data/synthetic_claims/claims.jsonl` (40) +
  `data/synthetic_claims/product_claim_mismatch.jsonl` (11). Shuffled (seed 0)
  before any `--limit`, so a smoke run spans cohorts. The **policy-header arm**
  only (`build_claim_text(..., policy_header=True)`): the no-header arm is a
  verdict-accuracy concern ([M4-10]) and mixing the two would blur the
  percentiles.
- **One assessment** = one full graph run, `START → intake → … → recommendation
  → human_review`, *including* any clarification rounds. `human_review`
  interrupts; the runner supplies an `InMemorySaver` and resumes with a canned
  `approve`. **End-to-end latency excludes the human's deliberation time** —
  there is no human — and is `perf_counter` around the two `.invoke` calls (to
  the interrupt, then the resume).
- **Per-node latency** is the [M5-06] node logger's `duration_ms`.
  `build.py::_instrumented` already brackets every node run with a
  `perf_counter`-measured `node.completed` / `node.failed` line carrying the
  run's correlation id; a `logging.Handler` captures those, a unique correlation
  id per claim demultiplexes them. Each node in the [M4-07] parallel superstep
  times *itself*, so `compatibility` / `consistency` / `injection_scan` are real
  per-node numbers, not an additive fiction.
- **Per-node token cost** is a LangChain callback handler on the graph's run
  config. It keys LLM calls by `run_id` (the concurrent superstep is
  unambiguous), attributes each to its node via `metadata["langgraph_node"]`,
  and reads the **full** `usage_metadata` — including
  `output_token_details["reasoning"]`. `token_cost.price_call` prices the counts
  from `LLM_{FAST,REASONING}_{INPUT,OUTPUT}_COST_PER_1M_TOKENS_USD`.
- **Non-determinism.** Intake's product-line classification varies between runs,
  grounding-retry counts vary, and provider latency is noisy. Every number here
  is one sample unless `--repeats N` was passed; the run metadata records which.

---

## Prediction, pre-registered

Written into the approved implementation plan **before any run** and reproduced
here unedited. Deviations are reported in **Findings**, not absorbed.

1. **End-to-end latency**: p50 in the 80–130 s range, p95 in the 200–320 s range
   (the tail is clarification-loop claims — two extra intake + two clarification
   calls — and compatibility grounding retries).
2. **The compatibility (reasoning) node dominates both**: ≥ 50 % of per-node
   latency and ≥ 85 % of per-assessment token cost. The other four LLM calls
   together cost under $0.001.
3. **Mean token cost per assessment**: $0.015–0.04; p95 $0.04–0.09.
4. **Reasoning tokens are the majority of the compatibility call's output** —
   ≥ 70 %. Whether the persisted `AuditEvent.TokenUsage` *total* already includes
   them (it captures LangChain's `output_tokens`, which for the fast model on
   `baidu/fp8` is observed to include reasoning) is settled by the handler-vs-
   audit-trail table — the prediction is that they match, and the only thing the
   audit trail lacks is the reasoning *breakdown*.
5. **Parallel branch vs sequential**: token cost identical; latency saving ≈ one
   consistency call (fast model, ~15–25 s), i.e. 15–30 % of the assessment stage
   and 8–20 % of end-to-end latency. Consistent with `docs/PARALLEL_ASSESSMENT.md`
   ([M4-07], 25.9 % of the assessment stage at `--limit 6`).
6. **Retrieval latency** is a few seconds (the cross-encoder reranker runs on the
   RTX 3050, not the CPU path `docs/RERANKING.md` measures at ~9.7 s/query).

---

## Run — 2026-09-07

- **Hardware**: AMD Ryzen 5 5600H (6 cores / 12 threads), 16 GB RAM, NVIDIA
  RTX 3050 Laptop GPU (4 GB). Linux 7.0. The cross-encoder reranker ran on the
  GPU; every LLM call went to OpenRouter.
- **Models / routes**: reasoning `deepseek/deepseek-v4-pro-0813` on `alibaba`
  ($0.5808 / $1.7424 per 1M in/out); fast `deepseek/deepseek-v4-flash-0731` on
  `baidu/fp8` ($0.14 / $0.28). The repo's default fast price already matches
  `baidu/fp8`; the reasoning price was set to `alibaba`'s current list price,
  because the settings default (`streamlake`, $1.1154 / $3.3462) is stale and
  `streamlake` was 404-ing for this model on the run day. No provider fallbacks.
- **Scored: 46 of 51** — 5 claims errored (see finding 6). One repeat. Wall
  clock ≈ 1 h 40 m (75.7 min of it inside the graph, the rest model load,
  per-claim session setup and the errored claims).

Regenerate with `make eval-performance`; the raw run is `eval/runs/performance.md`
and the committed snapshot is `eval/performance.json`.

## End-to-end latency

| metric | seconds |
| --- | ---: |
| p50 | **83.2** |
| p95 | **250.5** |
| mean | 98.7 |
| min | 8.2 |
| max | 404.4 |

The checkpoint resume adds nothing measurable (mean 0.002 s) — an assessment is
~99 s to the human checkpoint, then the analyst's own time, then a near-instant
resume. Human deliberation is not counted.

## Per-node latency

Per-invocation wall time from the [M5-06] node logger, over the 46 scored
assessments. `calls/asmt` is invocations ÷ 46 (a node the retrieval gate skips,
or a clarification round that did not happen, contributes 0); `ms/asmt` is the
node's mean total contribution to one assessment.

| node | invocations | calls/asmt | p50 ms | p95 ms | mean ms | ms/asmt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| intake | 56 | 1.22 | 13 195 | 30 208 | 15 788 | 19 220 |
| clarification | 10 | 0.22 | 4 504 | 11 221 | 4 981 | 1 083 |
| clarification_exhausted | 5 | 0.11 | 0 | 0 | 0 | 0 |
| retrieval | 41 | 0.89 | 622 | 2 296 | 769 | 686 |
| **compatibility** | 33 | 0.72 | **66 593** | **256 377** | **98 389** | **70 583** |
| consistency | 33 | 0.72 | 6 343 | 56 913 | 12 821 | 9 198 |
| injection_scan | 33 | 0.72 | 0 | 0 | 0 | 0 |
| recommendation | 46 | 1.00 | 5 120 | 13 989 | 5 367 | 5 367 |
| human_review | 46 | 1.00 | 0.1 | 0.1 | 0 | 0 |

`compatibility`, `consistency` and `injection_scan` run in one [M4-07] superstep;
each times itself, so these rows are real, not additive. `injection_scan` is a
no-op here (the optional classifier is off by default).

**The compatibility node is 72 % of a mean assessment's wall time** (70.6 s of
98.7 s) and its p95 (256 s) is the whole end-to-end p95. It is one call on the
reasoning model, which spends most of its output budget thinking (finding 3).
`intake` is second at ~16 s per call — the *fast* model also reasons heavily on
this route (~1 700 reasoning tokens per intake call). Retrieval, on the GPU
reranker, is sub-second at the median.

## Token cost per assessment

**Mean $0.0125 / assessment, p95 $0.0371.** The whole 46-assessment run cost
**$0.58**, over 438 k input + 416 k output tokens — **377 k (91 %) of the output
was reasoning tokens**.

| node | calls | in tok/call | out tok/call | reasoning/call | mean $/asmt | p95 $/asmt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| intake | 61 | 1 997 | 1 881 | 1 718 | $0.001069 | $0.003229 |
| clarification | 11 | 1 326 | 288 | 182 | $0.000064 | $0.000581 |
| **compatibility** | 45 | 3 575 | 4 981 | 4 515 | **$0.010521** | **$0.035384** |
| consistency | 36 | 1 537 | 1 297 | 1 286 | $0.000453 | $0.001747 |
| recommendation | 36 | 2 386 | 755 | 587 | $0.000427 | $0.000788 |

`calls` exceeds one-per-assessment where a node retried: `compatibility` re-grounds
an ungrounded answer (up to 3 calls — 6 of 33 assessed claims hit the ceiling),
`intake` retries a transient failure. Reasoning tokens are priced at the output
rate and are already inside `out tok/call` — the column is a memo, not an addend.

**The compatibility call is 84 % of the mean per-assessment cost** ($0.0105 of
$0.0125); the other four LLM calls together are $0.002. That ratio is the sharper
argument for [M4-07]'s parallel branch than the wall-clock saving: the expensive
call is on the critical path regardless.

## Parallel branch vs sequential

- **Token cost is identical** either way — $0.505 of assessment-branch LLM calls
  across the 33 claims that reached the fan-out, whether the compatibility,
  consistency and injection-scan calls run concurrently or back to back.
  Parallelism changes scheduling, not the call set.
- **Latency saving**: mean **10.4 s** (p50 6.3 s, p95 46.1 s) — **9.3 % of the
  assessment stage, 11.3 % of end-to-end latency**. Sequential mean 111.2 s →
  parallel mean 100.8 s.

The saving is bounded by `min(t_compat, t_consist, t_injection)`, in practice the
consistency call (the consistency node uses the fast model deliberately —
`docs/CONSISTENCY_NODE.md`). `docs/PARALLEL_ASSESSMENT.md` ([M4-07]) measured
25.9 % of the assessment stage at `--limit 6`, where the compatibility call
averaged ~63 s; here it averages ~98 s, so the same ~10 s consistency call hides
under a larger number and the *percentage* falls — the direction [M4-07]'s doc
predicted. The **absolute** saving still ≈ one consistency call, as [M4-07]
claimed structurally.

---

## One-off corpus indexing cost

Reported **separately** and never folded into the per-assessment number: it is
paid once to build the retrieval index (`make build-index`), not per claim.

| stage | dollar cost | wall clock | source |
| --- | ---: | ---: | --- |
| Text extraction + OCR | $0.00 | minutes | local Tesseract; `docs/PARSING.md` |
| LLM clause classification | **~$0.5–3** (estimate) | ~1–2 h at 10 workers | this document — see below |
| Vision boundary escalation | **~$2.85** (real, across 3 runs) | ~3 h 04 m | `docs/PARSING.md` "Known limitations"; the committed `eval/boundary_escalation_cost_report.json` reads $0.00 only because the final run was fully cache-served |
| Chunk embedding | **$0.00** | **~41.2 min** cold, CPU | `docs/EMBEDDINGS.md` §"Corpus embedding cost (2026-08-28, AMD Ryzen 5 5600H)" + `eval/runs/embedding_cost_report.{md,json}` — local `gte-multilingual-base`, no API |
| `CREATE INDEX` (pgvector) | $0.00 | < 1 s | `docs/EMBEDDINGS.md` §ANN |

**Total: ~$4–6, dominated by the LLM passes (classification + vision escalation);
embeddings are free but cost ~41 min of CPU.** A reviewer running the demo mode
(`make fetch-demo-artifacts`) pays none of it — the parsed corpus and the
embedding cache are downloaded. Either way this is a **one-off**: at $0.0125 per
assessment it is ~400 assessments' worth of cost, paid once.

### The classification-cost estimate

`scripts/build_corpus.py` runs `LangchainClauseClassifier`
(`deepseek/deepseek-v4-flash-0731`, the fast model, `baidu/fp8` route) over every
clause the deterministic rules ([M1-05]) leave as `other`. It is not separately
metered — the cache (`data/cache/llm_classification/cache.jsonl`,
content-addressed) records only `{clause_type, confidence}`, no token counts —
so this is an estimate, not a bill:

- **Calls**: ~5,100 (one per distinct clause; `build/manifest.json` records 4,925
  clauses, the cache holds a few hundred more from earlier corpus versions).
- **Tokens per call**: a ~60-token system prompt + the structured-output schema +
  one clause's title and full text in (most clauses a few hundred tokens, a
  handful far longer); a two-field `{clause_type, confidence}` out. Take ~700
  input per call. Output is the unknown: this measurement observed the same fast
  model emitting ~1 700 *reasoning* tokens per intake call, but
  `docs/INTAKE_EXTRACTION.md`'s classifier-era run reads ~4 300 tokens/call
  total, which leaves no room for that — so the corpus classification either
  predates the reasoning behaviour or ran a non-reasoning configuration. Both
  bounds:
  - **no reasoning** (~30 out/call): `5 100 × (700 × $0.14 + 30 × $0.28) / 1e6 ≈
    $0.54`.
  - **~1 700 reasoning out/call**: `5 100 × (700 × $0.14 + 1 730 × $0.28) / 1e6 ≈
    $3.0`.
- So **$0.5–3**, most likely near the low end (the classifier prompt is short and
  the task trivial next to intake).

If a metered figure is ever wanted, `scripts/build_corpus.py` would get a
`--cost-report` writing `eval/…_cost_report.json` the way
`scripts/escalate_vision_boundaries.py` does.

---

## Handler vs the audit trail

`scripts/eval_performance.py` reads token usage from a callback handler on the
graph run; the persisted `audit_trail` reads it from each node's one
`AuditEvent.TokenUsage`. Per node, over the 46 scored assessments:

| node | handler tokens | audit-trail tokens | gap |
| --- | ---: | ---: | ---: |
| intake | 236 561 | 236 561 | 0 |
| clarification | 17 751 | 17 751 | 0 |
| **compatibility** | **384 996** | **279 399** | **−27 %** |
| consistency | 102 007 | 102 007 | 0 |
| recommendation | 113 064 | 113 064 | 0 |

Four of five nodes match exactly — for `deepseek-v4` on these routes LangChain's
`output_tokens` (and so the audit trail's `total_tokens`) already includes the
reasoning tokens, so the audit trail's *totals* are complete; only the reasoning
*breakdown* is not stored. **The compatibility node is the exception**, and not
because of reasoning tokens: `_invoke_grounded` makes up to 3 billable calls when
it re-grounds an ungrounded answer, but the node writes **one** `AuditEvent`
carrying only the **last** call's usage. On the 6 claims that hit the retry
ceiling the audit trail sees roughly a third of the real compatibility cost. See
finding 5.

`docs/OBSERVABILITY.md` finding 4 is a separate, Langfuse-SDK artifact (the v4
handler splits `output_reasoning` out of `output`, so a project without the
`output_reasoning` pricing tier under-prices — which is why [M5-07] added that
tier).

Two well-scoped follow-ups fall out of this table, both tracked separately from
this measurement issue: (a) record every grounding-retry call in the audit trail,
not just the last; (b) add a `reasoning_tokens` field to `TokenUsage` so the
breakdown is queryable in SQL.

---

## Findings

**Method note.** One sample, 46 of 51 claims scored. Percentile bins are coarse
at n = 46 (p95 is the 43rd-smallest value); read the tail figures as directional.
Provider latency is noisy and the reasoning model's verbosity varies call to
call. The pre-registered predictions are checked below; deviations are published,
not absorbed.

### 1. p50 83 s, p95 250 s — the compatibility node is essentially the whole story

Prediction 1 held (p50 80–130 s, p95 200–320 s). Prediction 2's latency half held:
the compatibility node is **72 % of a mean assessment** (70.6 s of 98.7 s) and its
p95, 256 s, *is* the end-to-end p95 of 250 s. It is a single reasoning-model call.
Everything else on the critical path is small: intake ~16 s/call, consistency
~13 s (hidden under compatibility), retrieval sub-second on the GPU reranker
(prediction 6 held), recommendation ~5 s, the checkpoint resume ~0.

### 2. $0.0125 mean / $0.0371 p95 per assessment — 84 % of it is one call

Prediction 3 was slightly optimistic on the low side: **mean $0.0125** (predicted
$0.015–0.04), **p95 $0.0371** (predicted $0.04–0.09, so also a touch low). The
whole 46-assessment run cost **$0.58**. Prediction 2's cost half: the
compatibility call is **84 %** of the mean (predicted ≥ 85 % — a hair under).
OpenRouter served 0.8–1.8 k cached input tokens on many fast-model calls
(`cache_read_tokens` in the per-claim file), so the real input cost sits a little
below the list-price estimate.

### 3. 91 % of all output tokens are reasoning

Prediction 4 held strongly: of 416 k output tokens across the run, **377 k (91 %)
were reasoning**. Both `deepseek-v4` variants think far more than they answer —
the reasoning model averages 4 515 reasoning tokens per compatibility call
(against ~470 of actual answer), and the *fast* model averages ~1 700 per intake
call. This is why intake costs ~16 s and compatibility ~66–256 s: the wall time
is thinking, not generation of the structured result.

### 4. The parallel branch saves 9 % of the assessment stage, not the 25 % of the [M4-07] doc

The [M4-07] fan-out saves a mean **10.4 s — 9.3 % of the assessment stage, 11.3 %
of end-to-end** (prediction 5 said 15–30 % of the stage — a **miss, low** — and
8–20 % of end-to-end — held). The absolute saving (≈ one 10 s consistency call)
is exactly [M4-07]'s structural claim; the stage *percentage* fell because this
run's compatibility call (~98 s mean) is half again as long as the `--limit 6`
run `docs/PARALLEL_ASSESSMENT.md` reports (~63 s), so the same consistency call is
a smaller slice — the direction that doc already predicted. The token-cost half is
identical by construction — $0.505 of assessment-branch calls either way.

### 5. The audit trail under-reports the compatibility node's cost by 27 %

Prediction 4 anticipated "the audit trail reads low on that node" but for the
wrong reason. The reasoning breakdown *is* inside `output_tokens` for these
routes (every other node's handler total matches the audit trail exactly). The
compatibility gap is **grounding-retry loss**: `nodes/compatibility._invoke_grounded`
issues up to 3 billable calls, the node persists one `AuditEvent` with only the
last call's `token_usage`, and 6 of 33 assessed claims hit the 3-call ceiling.
A compliance reader pricing an assessment from the `audit_event` table would
undercount those claims ~3×. Follow-up (a) above.

### 6. 5 of 51 claims (10 %) failed at intake — the fast model returned no output

`compatible-013`, `incompatible-005`, `incompatible-007`,
`insufficient_information-003`, `mismatch-006` all raised `SchemaValidationError`:
`deepseek-v4-flash` on `baidu/fp8` returned `content=''` (only ~150 reasoning
tokens, no structured output) for the intake prompt. It is deterministic per
claim — `compatible-013` failed the same way in an earlier smoke run. The intake
node raises on an unparseable response by [M4-02]'s design (no template
fallback, unlike clarification), so the claim is lost. This is a fast-model
structured-output robustness problem on this route, not a harness bug; a
different route or a retry-with-reprompt in the node would recover them. The
scored figures above are over the 46 that completed.

### What this means downstream

- **The README results table** ([M6-01]) should carry p95 latency ≈ 250 s and
  mean cost ≈ $0.013 with the "one sample, n = 46" caveat, and note that latency
  is dominated by one reasoning-model call and is therefore route-sensitive.
- **The compatibility node is the single lever** for both latency and cost. A
  faster or less verbose reasoning model, or a cap on its reasoning budget, moves
  every headline number here.
- Two audit-trail fidelity follow-ups (grounding-retry accounting; reasoning-token
  field) are filed off finding 5.
