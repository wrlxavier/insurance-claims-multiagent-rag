# The assessment API — [M5-04]

The HTTP surface over the [M5-02] use cases. Five endpoints, one uniform error
envelope, citations as structured objects. The graph stays entirely behind
`ClaimAssessmentOrchestrator` — no route, schema or mapper imports LangGraph.

```
POST   /v1/assessments                     submit a claim            -> 202 + id
GET    /v1/assessments                      list, newest first       -> 200
GET    /v1/assessments/{id}                 state + recommendation   -> 200
POST   /v1/assessments/{id}/decision        approve / edit / reject  -> 200
GET    /v1/assessments/{id}/audit           durable audit trail      -> 200
GET    /health                              liveness                 -> 200
```

Code:

- `app/src/presentation/app.py` — `create_app()` and the `lifespan` composition
  root (engine, session factory, the two chat models, the retrieval components,
  the LangGraph orchestrator, `SystemClock`, the id minter).
- `app/src/presentation/dependencies.py` — `Depends` providers; one use case per
  request, a fresh read session for the `GET` paths.
- `app/src/presentation/{schemas,mappers}.py` — Pydantic models, pure
  dataclass↔schema conversions.
- `app/src/presentation/errors.py` — the single error→HTTP edge.
- `app/src/presentation/routes/{assessments,health}.py` — the routers.
- `app/src/infrastructure/graph/orchestrator.py` — `LangGraphClaimAssessmentOrchestrator`
  and its `_CapturingAuditSink`.
- `app/src/infrastructure/graph/state_mapper.py` — the `state.py` ↔ domain mapper.
- `app/src/infrastructure/graph/verdict_readout.py` — the verdict read (shared
  with `scripts/eval_end_to_end.py`).
- `app/src/infrastructure/clock.py` — `SystemClock`.
- `app/src/infrastructure/rag/retriever_factory.py` — the shared retrieval-stack
  builder.
- `app/src/infrastructure/database/audit_trail_{reader,writer}.py` — the audit
  read model and the transactional writer.

## Bring-up

```bash
cp .env.example .env          # fill LLM_PROVIDER / LLM_API_KEY / DATABASE_URL
make migrate                  # assessment / decision / audit_event tables
make setup-checkpointer       # the LangGraph checkpointer's own tables
make build-index              # raw PDFs -> parsed -> chunks -> Postgres -> embeddings
make serve                    # uvicorn presentation.app:app --reload --port 8000
```

`make serve` runs under the `embed` uv group — the retriever loads the local
sentence-transformers embedder and cross-encoder reranker once at startup and
fails fast with an actionable message if the group or the embedded chunk corpus
is missing. The checkpointer schema is probed at startup too.

## The endpoints

### `POST /v1/assessments`

```json
{ "raw_text": "Bati o carro na traseira de outro veículo...",
  "policy_ref": "15414.610650/2024-59",   // optional; SUSEP process, canonical or 17 digits
  "claim_id": "optional-external-id" }
```

`202 Accepted`, `Location: /v1/assessments/<id>`, body `{"assessment_id": "<id>",
"status": "awaiting_review"}`. The graph runs synchronously to the human
checkpoint before the response returns (see *Findings*). `policy_ref` is
prepended to the narrative as `[Apólice registrada: processo SUSEP …]` so intake
extracts it and retrieval pre-filters on it.

### `GET /v1/assessments/{id}` and `GET /v1/assessments`

The full servable aggregate:

```json
{ "assessment_id": "...", "claim_id": "...", "status": "awaiting_review",
  "verdict": "compatible", "reasoning": "...", "recommended_action": "...",
  "confidence": 0.72, "is_grounded": true,
  "citations": [ { "clause_id": "...", "document_id": "...",
                   "susep_process": "15414.610650/2024-59",
                   "clause_type": "coverage", "excerpt": "...",
                   "relevance_score": 0.83 } ],
  "consistency_flags": [ { "check": "...", "severity": "attention",
                           "detail": "...", "source": "deterministic" } ],
  "context_sufficient": true, "clarification_exhausted": false,
  "missing_information": [], "created_at": "2026-09-03T12:00:00Z",
  "decision": null }
```

The collection endpoint takes `claim_id`, `status`, `limit` (>0, default 50) and
`offset` (>=0), newest first.

### `POST /v1/assessments/{id}/decision`

```json
{ "decision": "approve" | "edit" | "reject",
  "notes": "conferido",
  "edited": {                         // required iff decision == "edit"
    "verdict": "incompatible", "reasoning": "...", "recommended_action": "...",
    "confidence": 0.6,
    "citations": [ { "clause_id": "...", "document_id": "...",
                     "susep_process": "...", "clause_type": "exclusion",
                     "excerpt": "..." } ] } }
```

`200 OK` with the settled aggregate. The system's verdict/prose/citations are
**unchanged** — an `edit` lives entirely in `decision.edited_assessment`. Every
cited clause on an `edit` is validated against the corpus before the graph is
resumed.

### `GET /v1/assessments/{id}/audit`

```json
{ "assessment_id": "...",
  "entries": [ { "sequence": 0, "timestamp": "...", "node": "retrieval",
                 "action": "retrieve_clauses", "model": null, "confidence": null,
                 "node_input": "...", "payload": null },
               ... ,
               { "sequence": 6, "node": "human_review",
                 "action": "human_decision:approve",
                 "payload": { "decision": "approve", "notes": "conferido", ... } } ] }
```

Empty (`"entries": []`) until a decision is submitted — the durable trail is
written once, at the human checkpoint.

## Errors

Uniform envelope; `code` is stable, branch on it:

```json
{ "error": { "code": "assessment_not_found", "message": "...",
             "details": { "assessment_id": "..." } } }
```

| code | HTTP | when |
| --- | --- | --- |
| `assessment_not_found` | 404 | `GET` / decision / audit on an unknown id |
| `assessment_already_decided` | 409 | a decision on a settled assessment |
| `unknown_clause` | 422 | an `edit` cites a clause absent from the corpus (`details.clause_ids`) |
| `citation_required` | 422 | an `edit` cites nothing |
| `invalid_susep_process` / `invalid_value_object` / `verdict_not_permitted` | 422 | a malformed `policy_ref` or `edited` payload |
| `invalid_request` | 422 | an `edit` without a payload, a non-`edit` carrying one, bad paging |
| `request_validation_error` | 422 | the body failed schema validation (`details.errors`) |
| `orchestrator_contract_error` | 502 | `start` did not pause / `resume` did not finish |
| `internal_error` | 500 | anything unexpected — logged, no traceback in the body |

## Results

`tests/integration/test_assessment_api.py` runs the whole flow against real
Postgres (the assessment/decision/audit tables and the LangGraph checkpointer)
with a canned-output model and a one-clause stub retriever: `POST` → 202 + id →
`GET` shows `awaiting_review` + a `compatible` recommendation + a structured
citation → `GET …/audit` is empty → `POST …/decision {approve}` → 200 `decided`
→ `GET` shows the system verdict unchanged beside the decision → `GET …/audit`
shows the trail ending in `human_review` / `human_decision:approve` with the
decision in `payload`. A second decision returns 409. The `reject` and `edit`
paths, the unknown-clause 422, the unknown-id 404, and the transactional fold
(a `SELECT count(*) FROM audit_event` proving the trail committed with the
`DECIDED` record) are covered too.

## Findings

1. **`POST` is synchronous behind its 202.** The handler blocks for the whole
   graph run (minutes, many LLM calls). This is the interim: [M5-05] owns the
   Redis queue and the `PENDING/RUNNING/FAILED` run states that make it truly
   non-blocking. The 202 + `Location` contract does not change.
2. **The audit trail is empty before a decision.** By [M4-09] design the durable
   trail is written once, in `human_review`, after the analyst decides. `GET
   …/audit` on an `AWAITING_REVIEW` assessment is `200 {"entries": []}`, not a
   404.
3. **`policy_ref` is a text header, not a graph change.** It is byte-identical to
   the measured headline arm of `scripts/eval_end_to_end.py` — no new code path,
   no eval re-run.
4. **The transactional fold uses a capturing sink.** On `resume` the graph's
   trail is captured in `OrchestratorResult.audit_records` and written by the
   use case in the same transaction as the settled record, through
   `UnitOfWork.audit`. One `commit()`, both or neither. The append is idempotent,
   so a decision retry after a mid-write crash is a no-op.
5. **An `edit` drops consistency flags.** `EditedAssessmentInput` → domain
   `Assessment` carries none. Flags are attention points kept beside the verdict,
   not part of the decision.

## What this leaves for later

- **[M5-05]** — Redis-backed queue, bounded concurrency, retry/back-off,
  dead-letter, and the `PENDING/RUNNING/FAILED` run states behind a non-blocking
  202.
- **[M5-06]** — `GET /ready` with per-check detail, JSON logs to stdout, and
  correlation-id propagation from the request into every node and LLM call.
- **[M5-09]** — the Compose `api` service, the Dockerfile, the CI image build,
  and the proxy-header / trusted-host middleware from `ObservabilitySettings`.
- A pooled checkpointer connection (`docs/DATABASE.md` "the M5 shape") — the
  adapter opens one per `start` / `resume` today.
