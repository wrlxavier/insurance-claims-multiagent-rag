# The Postgres checkpointer and the human checkpoint — [M4-09]

The graph now stops before its opinion counts as anything. `recommendation`
([M4-08]) still produces the assessment; a new terminal node, `human_review`,
surfaces it, pauses on LangGraph's `interrupt()`, and does not continue until a
person has approved, edited or rejected it. The decision is recorded **beside**
the system's recommendation — never over it — and the run's audit trail is
written to a table a compliance reader can query without LangGraph in the loop.

Two things make that possible, and both are new here:

- a **Postgres checkpointer** on the same database as the rest of the service, so
  a paused run survives the process that started it;
- an **`audit_event` table**, so the trail is durable *and* separate from graph
  state.

Topology after this issue:

```
… → recommendation → human_review → END
```

`human_review` is the only node with an edge to `END`. The checkpoint is
unconditional — it is the product behaviour, not a mode, and there is no flag
that removes it.

Code:

- `app/src/infrastructure/graph/checkpointer.py` — `open_claim_checkpointer`
  (the context manager every caller uses), `checkpointer_conn_string` (the URL
  conversion), `build_checkpoint_serializer` (the allowlist),
  `assert_checkpointer_ready` (the actionable failure).
- `app/src/infrastructure/graph/nodes/human_review.py` — the node, plus
  `_review_payload` (what a reviewer sees), `_await_decision` (the
  interrupt/validate cycle), `_persist_audit_trail` (the one side effect).
- `app/src/infrastructure/graph/state.py` — `HumanDecision` (declared back in
  [M4-01], first written here) and `AuditRecord`.
- `app/src/infrastructure/graph/context.py` — the `AuditTrailSink` port and
  `GraphContext.audit_sink`.
- `app/src/infrastructure/database/{models,audit_repository,graph_audit_sink}.py`
  — the table, the insert-only write path, the one database↔graph adapter.
- `alembic/versions/20260902_04_create_audit_event_table.py`,
  `scripts/setup_checkpointer.py` (`make setup-checkpointer`).

## Bring-up

The schema comes from two places, because the checkpointer owns its own:

```bash
docker compose up -d postgres
make migrate              # the application schema, incl. audit_event
make setup-checkpointer   # checkpoints, checkpoint_blobs, checkpoint_writes,
                          # checkpoint_migrations
```

`make setup-checkpointer` is idempotent and acts on the same `DATABASE_URL` (or
the discrete `DATABASE_*` variables) as `make migrate` — the DoD's "configure the
checkpointer with the same database as the rest of the service". Point it at the
test database with `DATABASE_URL=$TEST_DATABASE_URL make setup-checkpointer`.
It is the single place `PostgresSaver.setup()` runs, the way `make migrate` is
the single place Alembic runs; every other caller gets `assert_checkpointer_ready`
instead, which turns a missing schema into a message naming this command rather
than a psycopg `UndefinedTable` from inside a superstep.

Compiling and running:

```python
with open_claim_checkpointer() as checkpointer:
    graph = build_claim_graph().compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": claim_id}}

    paused = graph.invoke(initial_state, config=config, context=context)
    payload = paused["__interrupt__"][0].value  # the review packet

    final = graph.invoke(Command(resume=decision), config=config, context=context)
```

Sync `PostgresSaver`, not `AsyncPostgresSaver`: the whole project is sync
psycopg3, a project-level decision recorded in `docs/DATABASE.md`.

## The review contract

**Out**, as the `__interrupt__` value — plain JSON, so nothing downstream needs
this project's Pydantic models to read it (this crosses a process boundary today
and an HTTP one in [M5-04]):

```json
{
  "schema_version": "v1",
  "claim_id": "claim-001",
  "recommendation": { "recommended_action": "...", "justification": "...",
                      "citations": [...], "consistency_flags": [...],
                      "confidence": 0.62 },
  "context_sufficient": true,
  "clarification_exhausted": false,
  "missing_information": [],
  "decision_options": ["approve", "edit", "reject"]
}
```

The *full* recommendation, not a summary of it — the DoD's "surfacing the full
recommendation for review".

**In**, as the `Command(resume=…)` value — anything `HumanDecision` validates: a
mapping (from an API) or the model itself (in process).

```python
{"decision": "approve", "notes": "conferido"}
{"decision": "reject", "notes": "fora de vigência"}
{"decision": "edit", "notes": "...", "edited_recommendation": {...}}
```

`HumanDecision`'s own validator enforces the one rule that matters: `edit`
carries a revision, and nothing else may. The revision lands in
`human_decision.edited_recommendation`; `state["recommendation"]` is never
written by this node, so the machine's original opinion survives an edit
untouched and the two sit side by side for the record.

## Four things about `interrupt()`, verified against langgraph 1.2.11

Reproduced against the installed version rather than taken from a tutorial. Each
one changed a design decision.

**1. The code above the pause runs twice.** Once on the invocation that reaches
the interrupt, once again on the invocation that resumes — LangGraph re-runs an
interrupted node from the top, it does not resume mid-function. Code below the
pause runs once. So `human_review` is deliberately pure until `interrupt()`
(read state, build a payload) and does its one side effect, the durable audit
write, strictly afterwards.
`test_the_durable_write_happens_once_and_only_after_the_pause` pins it: zero
writes at the pause, exactly one after the resume.

**2. Raising after the pause bricks the thread — permanently.** The resume value
is stored in the checkpoint's pending writes. If the node then raises, that same
value is *replayed* on every later `Command(resume=…)`: `state.next` becomes
empty, and resuming with a corrected value re-raises the original error forever.
The decision would be unrecoverable. This is why nothing below the interrupt in
this node is allowed to raise.

**3. Re-interrupting is the recoverable answer.** Calling `interrupt()` again
re-opens the checkpoint; the next resume supplies a fresh value while earlier
ones replay by index. So a malformed decision does not raise — the node loops,
attaching the validation error to the payload, and asks again:

```json
{ "...": "...", "error": "1 validation error for HumanDecision\n  Value error, decision 'edit' requires edited_recommendation" }
```

This is not a graph loop and cannot spin: every iteration hands control back to
the caller, and the interrupt sequence stays deterministic — call *n* always
returns resume value *n* — which is the condition the LangGraph documentation
puts on repeated interrupts.

**4. The serializer degrades unknown types to `dict`, silently.** LangGraph's
msgpack serializer will not rebuild a type it was not told about, and it does not
raise when it meets one: it returns a plain mapping. A `ClaimState` restored from
a checkpoint would then hand nodes dicts where they expect models, several
supersteps from the cause. `build_checkpoint_serializer` passes an explicit
allowlist, and derives it by enumerating the Pydantic models `state.py` defines
(plus the two `domain` enums that reach state through them) rather than listing
them by hand — so a sub-model added by a later issue is covered without anyone
remembering to come back. Allowlist entries are `(module, class name)` pairs; a
module-only entry matches nothing.
`test_a_full_claim_state_survives_the_checkpoint_round_trip` round-trips a fully
populated state and checks every value's type;
`test_an_incomplete_allowlist_degrades_silently` pins the failure mode it guards.

A fifth trap, this one about the schema rather than the API: **autogenerate
wants to drop the checkpointer.** Its four tables sit in the same database as the
application's but not in `Base.metadata`, so `alembic revision --autogenerate`
reads them as tables the models dropped and writes a migration deleting them —
along with every paused run. `alembic/env.py` filters them out by name; see
`docs/DATABASE.md`.

A sixth, smaller one: **compiling without a checkpointer is not an error.**
`.invoke()` returns early with `__interrupt__` set and no exception, so a
forgotten checkpointer shows up as a silently truncated run rather than a crash.
Every call site passes one — `open_claim_checkpointer` in the service,
`InMemorySaver` in unit tests and `scripts/eval_clarification.py`.

## The durable audit trail

`audit_event`, one row per `state.AuditEvent` the run produced:

| column | source |
| --- | --- |
| `thread_id`, `sequence` | composite primary key — the run, and the event's position in it |
| `claim_id` | what a human searches by (indexed) |
| `timestamp`, `node`, `action`, `model`, `model_version`, `confidence`, `node_input` | `AuditEvent`, field for field |
| `input_tokens`, `output_tokens`, `total_tokens` | `AuditEvent.token_usage`, flattened |
| `payload` (JSONB) | the human-review row only: the whole `HumanDecision` |

**Why the key is `(thread_id, sequence)`.** The checkpoint node re-runs from the
top every time its thread is resumed, and a run that dies between the write and
the checkpoint commit resumes into exactly the same write. An event's position in
its thread's trail is deterministic, so the insert is
`ON CONFLICT DO NOTHING` and repeating it is a no-op — no invented event id
needed. `thread_id` rather than `claim_id` because a claim re-submitted for a
second assessment is a second thread with a trail of its own; keying on the claim
would make the two collide.

**Why `payload`.** `AuditEvent` is flat and has no field for the analyst's notes
or a revised recommendation. Those live in graph state, which the checkpointer
makes durable — but only LangGraph can read that back. The DoD asks for a record
of "what the human actually decided"; a JSONB column makes it a `SELECT`.

**Why one write, at the checkpoint, rather than per node.** It is the single
point every path reaches; it is the only place a side effect is safe from the
interrupt's re-execution; and the human decision — the part a compliance reader
is actually after — does not exist until then. The cost is real and worth stating:
**a run abandoned before the checkpoint leaves no `audit_event` rows.** Its trail
sits in the checkpoint until the thread is resumed, and resuming writes the whole
thing. That is the trade the checkpointer buys.

Failure of the sink is not allowed to cost the decision: it is logged, recorded
as an `AuditEvent(action="persist_audit_trail_failed")` in the returned trail,
and the run completes. Per finding 2, raising there would strand the decision for
good.

**Scope.** `docs/DATABASE.md` originally assigned the audit table to [M5-03]
along with the domain tables. Only the audit table moved forward, to the issue
whose DoD requires it. [M5-03] still owns the `Assessment` / `HumanDecision`
tables and the database-level append-only enforcement (a rule or trigger
rejecting `UPDATE`/`DELETE`); until then the property holds by construction,
because `audit_repository` offers no update path.

## Evidence

Unit (`pytest -m unit`, no database, no network):

- `tests/unit/infrastructure/graph/test_human_review.py` — the pause and its
  payload, the side-effect-free-before-the-pause guarantee, approve/edit/reject,
  the original recommendation surviving an edit, the re-ask on a malformed
  decision, and a failing sink not costing the decision.
- `tests/unit/infrastructure/graph/test_checkpointer.py` — the URL conversion,
  the readiness probe, and the serializer allowlist.
- `tests/unit/infrastructure/graph/test_claim_graph.py` — nothing reaches `END`
  without passing the checkpoint.
- `tests/unit/infrastructure/database/test_models.py` — `audit_event` mirrors
  `AuditEvent`.

Integration (`make test-integration`, real Postgres):

- `tests/integration/test_human_checkpoint.py` +
  `tests/integration/checkpoint_restart_worker.py` — the DoD's restart check. Two
  **separate OS processes**, launched with `subprocess.run`, sharing nothing but
  `TEST_DATABASE_URL`: the first drives a claim to the checkpoint and exits, the
  second resumes it and finishes the run. The recommendation comes back
  byte-identical (and as a model, which is where a serializer gap would surface),
  the decision is recorded alongside it, and `audit_event` holds the whole trail
  exactly once. Two savers inside one interpreter would have proved only that a
  connection can be reopened.
- Also there, because they need the real tables: `setup()` building the
  checkpointer schema from nothing, its idempotency, and the audit write's.

## What this leaves for later

- **[M4-10]** measures end-to-end verdict accuracy. It runs the graph, so it
  supplies a checkpointer and resumes past the checkpoint like any other caller.
- **[M5-02]/[M5-04]** put a use case and an endpoint in front of this: the
  payload above is what `GetAssessment` returns and `SubmitHumanDecision`
  resumes. Validating the decision payload belongs there too — the node
  re-validates defensively, but a 422 to the client is better than a re-ask.
- **[M5-03]** takes the connection story further. `open_claim_checkpointer` opens
  a single connection, which is right for a script or a test and not for a
  service handling concurrent requests; a `ConnectionPool` is the M5 shape. It
  also owns the append-only enforcement and the domain tables.
