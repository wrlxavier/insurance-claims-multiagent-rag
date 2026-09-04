# Architecture

Cross-cutting design decisions that shape the pipeline and are not owned by any
one stage's document. Each entry states the decision, why it is a deliberate
choice rather than an incidental one, and where the evidence lives.

Stage-local rationale stays in the stage's own document — `docs/PARSING.md`,
`docs/EMBEDDINGS.md`, `docs/LEXICAL_RETRIEVAL.md`, `docs/HYBRID_RETRIEVAL.md`,
`docs/RERANKING.md`, `docs/EXCLUSION_CO_RETRIEVAL.md`. The project scope
statement is `docs/SCOPE.md`.

---

## Exclusion co-retrieval is a domain rule, not a retrieval heuristic — [M3-06]

**Decision.** After the retrieval pipeline ranks, for every retrieved `coverage`
clause the system pulls the `exclusion` clauses structurally linked to it — a
sibling or nested exclusion in the same section, an exclusion in an adjacent
section of a flat document, or one named by an in-text `cláusula N` cross-
reference — and reserves a slot of the final context for the best-linked
exclusion the ranking missed. `ExclusionCoRetrievalRetriever` and `ClauseGraph`
in `app/src/infrastructure/rag/exclusion_co_retrieval.py`; the tuned constants
and the fingerprint in `exclusion_co_retrieval_config.py`.

**Why it is a rule and not a heuristic.** A coverage clause tells you an event
*is* covered; the exclusion three paragraphs down tells you it *is not*, under
the circumstances that actually apply. An assessment built on the first without
the second is fluent, cites a real clause id, and reaches the wrong verdict —
the single failure mode M3 was scoped around ("a retriever that returns the
first without the second is worse than one that returns nothing"). The linked
exclusion is therefore not "a nice extra result" whose value a relevance score
should decide; it is **as load-bearing as the coverage clause it modifies**, and
the pipeline is not permitted to return one without the other when a structural
link exists. Reranker scores order passages by topical similarity to the
question — a question phrased as a coverage question will score the exclusion
lower — so leaving this to the ranker is leaving it to the wrong signal. The
step runs as deterministic Python over the M1 clause tree, in the same spirit as
M4-06's deterministic consistency checks: a structural fact the system should
not need a model's permission to act on.

**What it costs.** The step fires on most questions that retrieve a coverage
clause, not only exclusion questions, because insurance filings pair coverage
and exclusion sections as siblings almost everywhere. With one reserved slot and
an eviction rule that only ever drops a *supporting* clause (never a coverage or
an exclusion clause), this is measurably free on `golden-set-v1`: exclusion-
clause recall 92.6% → 100%, `coverage_with_exclusion` Recall@10 78.9% → 84.2%,
every other question type unchanged, overall Recall@10 91.5% → 92.3%. The slot
count, the adjacent-section page window and the "inject what the base missed"
rule were all set from the sweep in `docs/EXCLUSION_CO_RETRIEVAL.md`, which also
records the one design iteration and the residual limitations (a
structurally-isolated coverage/exclusion pair is a blind spot; the remaining
subset gap is now entirely coverage-side).

**Deviation on record.** The DoD asks for in-text cross-references to be parsed
during M1. M1's corpus schema is frozen and a re-parse is disproportionate for a
pure function of text the artifact already carries, so `extract_cross_references`
runs at graph-build time — the production formalisation of the helper
`scripts/find_candidate_clauses.py` ([M2-08]) already uses for golden-set
curation. Full rationale in `docs/EXCLUSION_CO_RETRIEVAL.md`.

---

## Graph nodes are plain functions with dependencies injected through the runtime — [M4-01b]

**Decision.** A graph node is a module-level function
`def <name>(state: ClaimState, runtime: Runtime[GraphContext]) -> dict[str, object]`,
one node per file under `app/src/infrastructure/graph/nodes/`. It reads `state`,
never mutates it, and returns only the keys it changed — a partial `ClaimState`
update — always including at least one `AuditEvent` appended to `audit_trail`.
No class-based nodes, no `__call__` objects, no module-level model or retriever
singletons. Run-scoped dependencies — the fast and reasoning chat models, the
retrieval port, the model config — are fields of `GraphContext`
(`app/src/infrastructure/graph/context.py`), a frozen dataclass registered with
`StateGraph(ClaimState, context_schema=GraphContext)` and read inside a node as
`runtime.context`. Each LLM node's structured-output schema is a frozen Pydantic
`<Node>Output` in `app/src/infrastructure/graph/schemas.py`, kept distinct from
the `state.py` sub-models. Each node's prompt is built by
`build_<node>_prompt(...) -> str` in
`app/src/infrastructure/graph/prompts/<node>.py`, wrapped in
`with_scope_preamble(...)`; prompt text never appears inside a node function.

**Why a function and not a class.** The node is then testable by calling it with
a literal state dict and a `GraphContext` of fakes — no graph, no `compile()`,
no framework in the test. It is LangGraph's documented v1 style, and it is the
shape `tests/unit/infrastructure/graph/test_state_merges.py` already uses for
its fan-in fixtures. A class holding its dependencies in `__init__` reintroduces
the construction-order and shared-mutable-state problems `context_schema` exists
to remove, and invites five M4 issues to settle on five different constructors.

**Why dependencies arrive through `Runtime[GraphContext]`.** Nodes are
registered as bare functions — there is no constructor to pass anything to. A
module global couples every node to one process-wide client and makes the test
suite fight over monkeypatches. `context_schema` is the single seam LangGraph
provides, and `GraphContext` is the one object the composition root builds and
hands to `graph.invoke(state, context=...)`. The API moved at v1: LangGraph
1.2.11 uses `context_schema=` / `Runtime` / `get_runtime`, not the deprecated
`config_schema=` / `config["configurable"]` of pre-1.0 tutorials. The return
type is `dict[str, object]`, not `ClaimState` — a partial return typed
`ClaimState` fails on its required `claim_id` / `raw_claim_text` keys.

**How to add a node.**

1. `schemas.py`: add `class <Node>Output(BaseModel)` (frozen) — the exact shape
   passed to `.with_structured_output(...)`. Map it onto the `state.py`
   sub-model inside the node. A deterministic node makes no LLM call, so it adds
   no schema and no prompt file — the retrieval node ([M4-04]) is the first of
   those.
2. `prompts/<node>.py`: add `build_<node>_prompt(...) -> str` returning
   `with_scope_preamble(body)`.
3. `nodes/<node>.py`: the function. Take `fast_model` / `reasoning_model` /
   `retriever` / `llm_settings` off `runtime.context`; read optional state with
   `state.get(...)`. Return the changed keys plus an `AuditEvent` — set `model`
   and `token_usage` for an LLM call, leave them `None` for a deterministic one.
   Any other function in the module is a private `_`-prefixed helper.
4. `tests/unit/infrastructure/graph/test_<node>.py`, `@pytest.mark.unit`: a fake
   `BaseChatModel` and a stub retriever, no network. Verdict accuracy against
   the synthetic claims is a separate `eval`-marked test ([M4-10]).
5. Edges, the parallel fan-in and the checkpointer are not the node's concern —
   see [M4-07] and [M4-09]. Note that since [M4-09] the compiled graph needs a
   checkpointer and a `thread_id`; a single-node test graph for one node in
   isolation does not, because it has no `interrupt()` in it.

**Enforcement.** `tests/architecture/test_graph_node_conventions.py`
(unit-marked, runs in CI) fails on any class under `nodes/` that defines
`__call__` or subclasses a `Runnable*`, and on any exported node function whose
first two parameters are not `(state, runtime)`.
`tests/architecture/test_scope_vocabulary.py` already scans the same tree for
verdict-vocabulary drift.

**Instrumentation — [M5-06].** `build.py` registers every node through a
`_instrumented(node)` wrapper: each run brackets itself with a correlation-tagged
`node.start` / `node.completed` log line on the `infrastructure.graph.node`
logger (`node.failed` on a real exception; LangGraph control-flow exceptions such
as `human_review`'s `interrupt()` bubble silently), plus a `duration_ms`. It sits
in `build.py`, not the node files, so the `(state, runtime) -> dict` convention
and its enforcement test are untouched. The correlation id is
`GraphContext.correlation_id`, set by `LangGraphClaimAssessmentOrchestrator` from
the ambient request/worker id; the orchestrator also puts it in the graph
`config` metadata, which LangGraph copies onto every node's child
`RunnableConfig` so LLM calls carry it. See `docs/API.md` "Structured logging &
correlation IDs".

[M5-07] adds spans beside those log lines, and adds them **without touching this
wrapper or any node** — the Langfuse callback handler goes on the run's config in
the orchestrator, where it sees the nodes *and* the LLM calls inside them. See
"Tracing is one callback handler on the run" below and
`docs/OBSERVABILITY.md`.

---

## The clarification loop is self-capping in the router, not in the framework — [M4-03]

**Decision.** When intake leaves `missing_information` non-empty, a conditional
edge routes the claim to a `clarification` node (one specific question per gap)
and back to `intake`. `route_after_intake` in
`app/src/infrastructure/graph/build.py` — the first real graph assembly, which
every later M4 node issue extends — enforces the cap: once `clarification_rounds`
reaches `MAX_CLARIFICATION_ROUNDS` (a module constant, `2`) with gaps still open,
it routes to a deterministic `clarification_exhausted` node that sets
`clarification_exhausted = True`, and the graph ends. Loop termination is a
property of the router plus the counter; the graph never depends on LangGraph's
`recursion_limit` / `GraphRecursionError`.

**Why the cap is a code constant, not an `.env` setting.** The clarification cap
is product behaviour — "ask, then follow up once, then escalate to a human" —
defined and tested in code, exactly like the retry constants in
`application/use_cases/llm_retry_defaults.py`. The `.env` LLM knobs
(`LLM_*_PROVIDER_ORDER`, `LLM_CLASSIFICATION_MAX_WORKERS`) are deployment-
environment concerns; the number of times a claims bot re-asks is not.

**Why `clarification_exhausted` is its own state channel.** It means "the
claimant never supplied enough to proceed", which [M4-08] maps to
`INSUFFICIENT_INFORMATION` and [M4-10] catalogues as a distinct failure mode. It
is deliberately *not* folded into `context_sufficient` (owned by [M4-04] and the
[M3-07] retrieval gate — "retrieval did not return enough"), nor written as a
citation-free `CompatibilityAssessment` into the `compatibility` channel (whose
"every assertion cites a clause" contract is [M4-05]'s). The open gaps stay
listed in `missing_information`.

**Why intake merges on re-entry.** The DoD requires accumulated context to
survive an iteration. The `intake` node reads the previous `entities` and the
questions already asked, feeds the questions into its prompt, and null-coalesces
its fresh extraction over the prior one (fresh non-null wins; prior non-null
fills a fresh gap). Extraction re-runs, but an already-known fact is never lost.
`missing_information` is recomputed fresh so an answered gap drops off.

Full method and the committed measurement: `docs/CLARIFICATION_LOOP.md`.

---

## The retrieval node wraps the M3 pipeline behind a widened port — [M4-04]

**Decision.** The retrieval node (`app/src/infrastructure/graph/nodes/retrieval.py`)
depends only on `RetrievalPort` (`infrastructure/graph/context.py`). It builds the
query from the extracted *entities* (`_build_query` — `event_type` + `description`
+ `vehicle_info`, never the raw claim text), builds the metadata pre-filter from
intake's classification (`_build_filter` — SUSEP process when the claim stated one,
plus the product line; `None` when neither is known, the [M3-04] unconstrained
degradation path), calls the port, maps every returned clause to a `state.Citation`,
assembles a [M3-07] `GateSignals` from the reranker scores and runs `evaluate_gate`
to set `context_sufficient`. The router `route_after_retrieval` in `build.py` acts
on that flag; the sufficient-context path enters the assessment stage
([M4-05]/[M4-06], fanned out as fixed parallel branches by [M4-07]), and
`"insufficient"` routes straight to the recommendation node ([M4-08]).

**Why the port returns `RetrievedClause`, not clause-id strings.** A `Citation`
needs the clause's document, SUSEP process, clause type and a quoted excerpt, and
the gate needs the rank-1 reranker score — none of which a `list[str]` carries.
The port now returns `list[RetrievedClause]`
(`infrastructure/rag/retrieved_clause.py`): a frozen row with the provenance and
the score, ranked best-first. It lives in `infrastructure.rag`, referenced by the
graph port under `TYPE_CHECKING` exactly as `RetrievalFilter` already is — no new
import edge in either direction. The node does a trivial 1:1 field map; the seam
stays where "a retrieval hit" becomes "a clause an assertion cites".

**Deviation on record.** [M4-01b] wrote that the port "returns ranked clause-id
strings only — the retrieval node hydrates `Citation` objects itself", anticipating
an LLM retrieval node that emits ids. [M4-04] is deterministic instead: the widened
port hands back hydrated, scored rows and the node maps them. `context.py` reserved
this change ("[M4-04] owns any change to this shape"); the [M4-01b] entry's step-1
line is updated to match.

**Why a single bridge module.** `GraphRetrievalAdapter`
(`infrastructure/rag/graph_retrieval_adapter.py`) is the one place that composes
the M3 retrievers (`HybridRetriever` → cross-encoder rerank → optional
`ExclusionCoRetrievalRetriever`) and hydrates from the chunk corpus. It reranks by
hand rather than through `RerankingRetriever`, which discards the scores the gate
needs — the same reason `scripts/eval_insufficient_context_gate.py` re-implements
that step. A co-retrieved exclusion clause carries `score = 0.0`: co-retrieval
never re-scores, so rank-1 — the gate's only score signal — is unchanged, and the
gate's calibration (hybrid + rerank, no co-retrieval) still holds.

**Why `context_sufficient` is the node's, not the router's.** The gate decision is
retrieval's concern; the routing is topology. The node records the boolean and the
`GateTrigger` in its `AuditEvent`; `route_after_retrieval` only reads the boolean.
This keeps `context_sufficient` ("retrieval did not return enough") distinct from
`clarification_exhausted` ("the claimant never supplied enough"), as the [M4-03]
entry requires.

**Enforcement.** `tests/architecture/test_graph_node_conventions.py` (the node is a
plain `(state, runtime)` function with `_`-prefixed helpers, no LLM schema);
`tests/unit/infrastructure/graph/test_retrieval.py` and
`tests/unit/infrastructure/rag/test_graph_retrieval_adapter.py` (unit, fakes only);
`tests/eval/test_retrieval_node_baseline.py` (eval-marked, skips without the stack).

Full method and the committed measurement: `docs/RETRIEVAL_NODE.md`.

**A stated SUSEP process outranks the narrative-derived product line — [M4-10].**
`_build_filter` originally ANDed the two whenever intake produced both. That is
wrong, and [M4-10]'s end-to-end run is what exposed it. The two fields do not
describe the same thing: a SUSEP process names the **registered product the
claim was filed against**, while `product_line` is intake's classification of
the **event the claimant described**. On a product/claim mismatch — the case
`data/synthetic_claims/product_claim_mismatch.jsonl` exists to test — those two
disagree *by construction*, so their conjunction selects nothing, the [M3-07]
gate fires on an empty result, and a knowable `incompatible` degrades to
`insufficient_information`.

The mechanism is not a matter of model behaviour, and the evidence for it is
LLM-free: counting chunks in the database under each filter shows **11 of 11**
mismatch claims have an empty search space under the conjunction (their process
alone selects 19–117 chunks), and **9 of the 13** documents the main claim set
targets empty out too if intake reads the event as CASCO — which is the reading
the corpus invites, since 3,244 of its 4,540 chunks are CASCO. The defect was
never confined to the cohort that revealed it.

So when a process is stated it wins alone; the product line constrains only the
fallback path, where no process was read from the claim. This restores what
`infrastructure/rag/retrieval_filter.py`'s own docstring already asserted — "a
claims analyst works a case for a known policy" — and it narrows rather than
widens the search: a process selects one document, a product line selects a
whole segment of the corpus. Measured both ways in
`docs/END_TO_END_EVALUATION.md`, under a prediction registered before the run.

---

## The compatibility node grounds every assertion in a retrieved clause — [M4-05]

**Decision.** The compatibility node
(`app/src/infrastructure/graph/nodes/compatibility.py`) is the node that answers
the question. It reads `state.citations` and `state.entities`, calls the
**reasoning** model (`GraphContext.reasoning_model`, pinned by
`LlmSettings.llm_reasoning_provider_order` — [M4-05] is its first consumer),
structured output into `schemas.CompatibilityOutput`, and writes a
`state.CompatibilityAssessment` (verdict / reasoning / citations / confidence).
`route_after_retrieval` enters it on the sufficient-context path; [M4-07] made
that a fixed parallel fan-out to this node and the consistency node, converging
on `END` (see the [M4-07] section).

**Why the reasoning is a list of `(statement, clause_ids)` pairs, not prose.**
The DoD requires that *every assertion in the reasoning reference at least one
clause id*, and that a citation-free assertion be "a malformed output, rejected
and retried, not post-edited". Modelling `CompatibilityOutput.assertions` as a
list of `ReasonedAssertion` makes that a field check
(`_grounding_errors`): for a `compatible` / `incompatible` verdict every
assertion must carry ≥1 `clause_id`, and every id must be one retrieval actually
returned. On a violation the node appends a corrective turn naming the offending
assertions and re-invokes — up to `MAX_GROUNDING_ATTEMPTS` (3, a code constant
like `RETRIEVAL_K` / `MAX_CLARIFICATION_ROUNDS`). The node then renders the list
into the plain `reasoning: str` the state model holds and hydrates a
`state.Citation` per cited id from `state.citations`.

**Why an ungroundable answer degrades to `insufficient_information`.** After the
retries are spent the node does not raise and does not emit an ungrounded
`compatible` / `incompatible` verdict — it returns `insufficient_information`
with confidence `0.0` and a reasoning line recording the failure. An assessment
that cannot be tied to a clause is exactly the M0-06 "use
`insufficient_information` rather than guessing" case, and the graph stays
runnable. A *transient* (network) failure still propagates, via the same
`_invoke_with_retry` helper intake uses. When retrieval returned nothing at all,
the node skips the model call entirely and abstains — there is nothing to reason
over.

**Why the exclusion-weighing rule lives in the prompt.** The DoD asks the node
to "weigh retrieved exclusions against retrieved coverage explicitly, and say so
in the reasoning". Exclusion co-retrieval ([M3-06]) already guarantees the
linked exclusions are in `state.citations`; the prompt instructs the model to
compare them against the coverage side in an explicit assertion citing both.
This is a prompt requirement, not a second structural check — the structural
guarantee is the grounding one above.

**Enforcement.** `tests/architecture/test_graph_node_conventions.py` (plain
`(state, runtime)` function, `_`-prefixed helpers) and
`test_scope_vocabulary.py` (no `covered` / `denied` / `coverage decision` in the
new prompt); `tests/unit/infrastructure/graph/test_compatibility.py` (unit,
fake reasoning model — the grounding-retry loop, the degrade path, the
no-context guard, the audit event); `tests/eval/test_compatibility_baseline.py`
(eval-marked, skips without a reasoning model + the retrieval stack).

Full method and the committed measurement: `docs/COMPATIBILITY_ASSESSMENT.md`.

---

## The consistency node splits arithmetic from judgement, and only signals — [M4-06]

**Decision.** The consistency node
(`app/src/infrastructure/graph/nodes/consistency.py`) reads a claim for
*internal* consistency and runs two legs. The **deterministic** leg
(`app/src/infrastructure/graph/consistency_checks.py`) is plain Python over the
entities intake extracted, executed **unconditionally**, with no model: a stated
`event_date` in the future or long past, an `estimated_amount` outside crude BRL
bands, a field intake populated while also tagging it missing, a `product_line`
whose registered definition is contradicted by a token in the event text. The
**semantic** leg is one call on the *fast* model
(`GraphContext.fast_model`), structured into `schemas.ConsistencyOutput`, for
the judgement the first leg cannot make: narrative coherence, description vs.
stated event type, vagueness where a claimant would give detail. The two legs'
signals merge into one `state.ConsistencyReport`, each `ConsistencySignal`
tagged `source` (`"deterministic"` / `"llm"`). The node emits **two**
`AuditEvent` rows — `action="deterministic_checks"` (`model` / `token_usage` /
`confidence` all `None`, the absence being the record that no model ran) and
`action="semantic_judgement"` — so the boundary is visible in the trail itself.
[M4-07] wires this node alongside the compatibility node; `build.py` is untouched
here.

**Why the split falls where it does.** The rule is: *string / number / set
equality is Python; "do these two prose fields disagree in meaning" is the
model's.* Whether a date is after today, whether a number is in band, whether an
event token collides with a product line's own definition — none of that needs a
language model's permission, and routing it through one turns a system that is
right into one that is usually right. This is the same principle as exclusion
co-retrieval ([M3-06] above): *a structural fact the system should not need a
model's permission to act on.* The product-line/event-type check is deliberately
a closed contradiction table, not a re-run of intake's classifier: it fires only
on a positive, un-negated token collision, never proposes an alternative line,
and stays silent when `product_line` is `None` — a hit means intake's own
classification disagrees with the event text intake itself extracted. The table
holds only the two lines with an absolute definitional constraint (GAR.EST — no
external cause; ASSIST — a service, not indemnification); CASCO vs. RCF-A turns
on *who* was damaged, which a terse extracted description renders unreliably, so
those lines were dropped after the first eval run flagged coherent claims. Full
account: `docs/CONSISTENCY_NODE.md`.

**Why `fast_model`, not the reasoning model.** The semantic leg is a scan for
inconsistencies, not legal reasoning — it matches the fast model's weight and
the `clarification` node's precedent (lightweight structured output, graceful
degradation). The failure path is non-fatal: a model call that fails every retry
degrades to the deterministic signals alone and the node records that, it never
raises. And once [M4-07] runs this node and the compatibility node in the same
superstep, using the fast model here keeps the parallel branch from being two
reasoning-model calls on the critical path.

**Why signals, not a verdict.** `ConsistencyOutput` and `ConsistencySignal`
carry no verdict field, and the prompt says so explicitly (the scope preamble
names the verdict vocabulary the *other* nodes use, which is a tension worth
removing in words). The node flags for human attention; [M4-08] presents those
flags as attention points kept separate from the compatibility verdict, and
[M4-10] treats a signal as triage input, never a verdict.

**Why this is not a fraud detector.** Stated in the code (the module docstrings
of both `consistency_checks.py` and `nodes/consistency.py`), the prompt (which
forbids speculation about intent), and here. The data carries no fraud labels
and the method — four range checks plus an LLM reading for coherence — is not
one. `docs/SCOPE.md` is the canonical statement; this entry and the code must
stay consistent with it. Whether the event date falls inside the policy period —
the issue's motivating example — is **not** attempted: the corpus is registered
product conditions, not contracts, so it carries no vigência to compare against;
intake's `data_evento_vigencia` tag records that gap.

**LangGraph v1.** The node introduces no new idiom — it copies the
`compatibility` node's shape (`Runtime[GraphContext]` / `runtime.context`,
`with_structured_output(..., include_raw=True)` → `{"parsed", "raw"}`, a
module-local `_invoke_with_retry`). No deprecated `config_schema` /
`config["configurable"]`.

**Enforcement.** `tests/architecture/test_graph_node_conventions.py` (plain
`(state, runtime)` function, `_`-prefixed helpers) and
`test_scope_vocabulary.py` (whole-package rglob — no forbidden verdict
vocabulary in the new module or prompt);
`tests/unit/infrastructure/graph/test_consistency_checks.py` (every deterministic
check in isolation, no LLM import — the DoD's "unit-test every deterministic
check in isolation");
`tests/unit/infrastructure/graph/test_consistency.py` (unit, fake fast model —
the leg merge, the degrade path, the two audit events, the no-verdict guards);
`tests/eval/test_consistency_baseline.py` (eval-marked, skips without
`LLM_PROVIDER`).

Full method and the committed measurement: `docs/CONSISTENCY_NODE.md`.

---

## The two assessment nodes run as fixed parallel branches — [M4-07]

**Decision.** After the retrieval node, `route_after_retrieval`
(`app/src/infrastructure/graph/build.py`) fans the sufficient-context path out to
**both** the compatibility ([M4-05]) and consistency ([M4-06]) nodes — one
superstep, both nodes — and the two converge on `END`. This is LangGraph's
*fixed parallel branches* primitive: the router returns the list
`["compatibility", "consistency"]` (a list of node names is the fan-out), each
node has an edge to the fan-in point. `context_sufficient is False` still returns
`[END]` directly.

**Why not `Send`.** `Send` is dynamic dispatch — a conditional edge returning a
runtime-sized list of `Send(node, payload)` for map-reduce over a variable number
of items. This is two *known* nodes, always both; the right primitive is plain
edges. An earlier draft of the design document
(`.ai_context/Assistente_Sinistros_Apolices_Proposta_Completa_com_ERRATA.md`
sec. 6.3) used `Send` here and is wrong; its ERRATA and `state.py`'s docstring
already record the correction. Verified against the current LangGraph docs and
the installed langgraph 1.2.11.

**Why the concurrent writes are safe.** The nodes write disjoint channels —
`compatibility` vs `consistency`, one `LastValue` write each. The only shared
channel is `audit_trail`, which carries the `append_audit_events` reducer
([M4-01]); without it LangGraph raises `InvalidUpdateError` on the concurrent
append. No other channel is written by both.

**Why one branch's failure is loud.** No `error_handler` is registered — and
[M4-09], which was where that question was deferred to, deliberately did not add
one: it made the *checkpoint* node total instead (see below), which is where an
unrecoverable raise actually costs something. So if a branch raises a genuine
exception — the provider
unreachable after retries, say — LangGraph's runner cancels the sibling and
re-raises out of `.invoke()`; `apply_writes` never runs for that superstep, so no
partial state (`consistency` set, `compatibility` silently missing) is returned.
The failure surfaces rather than truncating one branch. The compatibility node's
internal degrade to `insufficient_information` for an *ungroundable* answer is a
valid result, not a failure, and does not abort the superstep.

**Why the fast model on the consistency leg pays off here.** Both calls run in
one superstep, so the parallel wall is about `max(t_compat, t_consist)` and the
saving is bounded by `min(t_compat, t_consist)`. The consistency node uses the
fast model ([M4-06]), so the parallel branch costs roughly one reasoning-model
call — the consistency check is close to free on the critical path. Measured:
the mean saving is ≈ one whole consistency call (~22 s/claim; 26% of the stage
on the current slow reasoning model, more on a faster one). Full run:
`docs/PARALLEL_ASSESSMENT.md`.

**The fan-in.** Both assessment edges point at the recommendation node
([M4-08]), which runs once after both branches finish (a node with an incoming
edge from each). `route_after_retrieval`'s insufficient-context path points there
too, so the recommendation node is the single terminal node.

**Enforcement.** `tests/unit/infrastructure/graph/test_claim_graph.py` (the
fan-out edges, the audit-trail merge across the superstep, and — the DoD's
"failure in one branch does not silently truncate the other" — a compiled-graph
run with a failing reasoning model asserting the raise);
`tests/unit/infrastructure/graph/test_state_merges.py` (the fan-in reducer and
the bare-channel race, unchanged).

Full method and the committed measurement: `docs/PARALLEL_ASSESSMENT.md`.

---

## The recommendation node consolidates; it does not re-decide — [M4-08]

**Decision.** One terminal node, `app/src/infrastructure/graph/nodes/recommendation.py`,
that every path routes through — the two assessment branches, the
insufficient-context path (`route_after_retrieval` now returns
`["recommendation"]` there), and `clarification_exhausted`. It reads the upstream
results and emits one `state.Recommendation`: `recommended_action`,
`justification`, `citations`, `consistency_flags`, `confidence`. The
`Recommendation` schema was defined by [M4-01]; no state-schema change.

**Why the citations can't drift.** The node never constructs a `Citation`, and
`schemas.RecommendationOutput` (what the model fills) has no citation field.
`citations` is a deduplicated copy of `compatibility.citations`, which [M4-05]
already hydrated from what retrieval returned. "Never introduce a citation no
upstream node produced" is therefore structural — there is no code path that
could — not a prompt instruction, and the unit test asserts the subset relation
directly.

**Why an insufficient upstream verdict stays unconfident.** `confidence` is
derived, not model-reported: it starts from `compatibility.confidence` (0.0 when
there is no assessment) and is clamped — an effective verdict of
`insufficient_information` (compatibility abstained, retrieval missed, or the
clarification loop was exhausted) caps it at `_INSUFFICIENT_CONFIDENCE_CEILING`
(0.3); an unresolved `attention` consistency flag caps it at
`_ATTENTION_FLAG_CONFIDENCE_CEILING` (0.7). An abstaining verdict cannot become a
confident recommendation.

**Why consistency flags stay separate.** `consistency_flags` is
`consistency.signals` verbatim — its own field, never merged into the verdict,
the action, or the confidence logic (only an `attention` flag's *ceiling* effect
touches confidence). Per [M4-06]: attention points, not part of the decision.

**Why the fast model, and only for the justification.** The legal reasoning is
[M4-05]'s; this is a summary. The model writes one paragraph — compatibility
finding first, then the clause ids, then the caveats — on the fast model, and
only on the path where a real assessment exists. The claimant-gaps and
retrieval-miss paths, and any transient model failure, fall back to a
deterministic template. The node is terminal, so it degrades, never raises
(like the consistency node's semantic leg).

**Enforcement.** `tests/unit/infrastructure/graph/test_recommendation.py` (unit,
fake fast model — the citation-subset invariant, the confidence clamps, the
flag pass-through, the degrade path, the no-model-call short-circuits);
`tests/unit/infrastructure/graph/test_claim_graph.py` (every terminal path
converges on the node, the three compiled-graph runs);
`tests/architecture/test_graph_node_conventions.py` and
`tests/architecture/test_scope_vocabulary.py` (auto-cover the new node and
prompt); `tests/eval/test_recommendation_baseline.py` (eval-marked — the two
structural guarantees re-checked on live output, skips without `LLM_PROVIDER`).

Full method and the committed measurement: `docs/RECOMMENDATION_NODE.md`.

---

## The human checkpoint pauses before anything is final — [M4-09]

**Decision.** A terminal `human_review` node
(`app/src/infrastructure/graph/nodes/human_review.py`) sits between the
recommendation node and `END`, and it is the only node with an edge to `END`. It
surfaces the whole recommendation, pauses on LangGraph's `interrupt()`, and
records the analyst's `approve` / `edit` / `reject` into `state.human_decision`.
Unconditional: the checkpoint is the product behaviour, not a mode, so
`build_claim_graph()` takes no flag that removes it, and the compiled graph
therefore requires a checkpointer and a `thread_id` at every call site.

**Why the decision sits beside the recommendation, not over it.** The node never
writes `state.recommendation`. An edited assessment lands in
`human_decision.edited_recommendation`, so what the machine produced and what the
human did with it are both on the record, separately — which is the only version
of this that is auditable after the fact.

**Why the node is pure until the pause and total after it.** LangGraph re-runs an
interrupted node from the top: everything above `interrupt()` executes twice
(once reaching the pause, once resuming), everything below it once. So the node
reads state and builds a payload above the line, and does its one side effect —
the durable audit write — below it. Symmetrically, nothing below the line may
raise: a resume value is stored in the checkpoint's pending writes, so a node
that raises after consuming one replays that same value on every subsequent
resume and the thread can never be finished. Hence a malformed decision
*re-asks* (a second `interrupt()` carrying the validation error) rather than
raising, and a failing audit sink degrades to a recorded event rather than
aborting. Both behaviours were verified against langgraph 1.2.11, not assumed.

**Why the audit trail is written again, to its own table.** Graph state is
already durable — that is what the checkpointer buys — but only LangGraph can
read it back, and `audit_event` is the record a compliance reader has to be able
to query. It is written once, at the checkpoint: the single point every path
reaches, the only place a side effect is safe from the re-execution above, and
the first moment the human decision exists. Keyed `(thread_id, sequence)`, which
makes the insert idempotent by construction rather than by an invented id.

**Why the checkpointer needs its own bring-up step.** `PostgresSaver` owns its
schema and migrates it itself, outside Alembic — so `make migrate` leaves a
database that is still not ready for the graph, and `make setup-checkpointer` is
the second half. Callers that do not create the schema get
`assert_checkpointer_ready`, which names that command, rather than an
`UndefinedTable` raised from inside a superstep.

**Enforcement.** `tests/unit/infrastructure/graph/test_human_review.py` (unit —
the pause, the payload, no side effect before it, the three decisions, the
original recommendation surviving an edit, the re-ask, the degrading sink);
`tests/unit/infrastructure/graph/test_checkpointer.py` (unit — the URL
conversion, the readiness probe, and the serializer allowlist, which silently
returns `dict` for an unregistered type);
`tests/unit/infrastructure/graph/test_claim_graph.py` (nothing reaches `END`
without passing the checkpoint); `tests/integration/test_human_checkpoint.py`
(integration — two separate OS processes, sharing only the database, prove the
paused run survives a restart).

Full contract, the verified `interrupt()` semantics and the table: `docs/HUMAN_CHECKPOINT.md`.

---

## Domain entities are frozen dataclasses that validate on construction — [M5-01]

**Decision.** The business layer (`app/src/domain/`) is `Policy`, `PolicyClause`,
`Claim`, `Assessment`, `HumanDecision`, the `Citation` value object and the
`SusepProcess` / `Cnpj` identifier value objects — every one a
`@dataclass(frozen=True)` whose `__post_init__` raises if the invariant it owns
is broken. Closed vocabularies are `enum.Enum` (`Verdict` reused unchanged from
[M4-01], `DecisionOutcome` new). Value objects take a strict constructor
(already-canonical input) plus a lenient `parse()` classmethod that normalises
I/O forms — `Cnpj.parse` applies the 14-digit zero-pad the upstream SUSEP
catalogue needs; `SusepProcess.parse` accepts the 17-digit filename stem. One
`domain/errors.py` holds the exception hierarchy. Standard library and typing
only — no Pydantic, no SQLAlchemy, no LangGraph.

**Why validate in `__post_init__`, and why the enum field still needs a guard.**
A frozen dataclass gives immutability and structural equality for free but does
**not** check its field types at runtime, so `Assessment(verdict="compatible",
…)` would construct happily and the DoD's "a verdict is one of the three
permitted values" invariant would be unenforced. Each entity therefore
`isinstance`-guards its enum fields explicitly (`VerdictNotPermittedError`) and
checks its cross-field rules (an assessment has ≥1 citation always — including
for `insufficient_information`; a `HumanDecision` always carries the
`assessment_id` it acted on, and an `edited_assessment` must revise that same
id). `Cnpj` verifies the real mod-11 check digits — a public, stable algorithm,
and a wrong-length or transposed-digit CNPJ is exactly the upstream-data bug
class the value object exists to catch; `SusepProcess` is format-only, because
SUSEP's process check-digit algorithm is not a published spec and a false
rejection of a real filing is unrecoverable.

**Why a twin of `infrastructure/graph/state.py`, not a move.** `state.py` is
Pydantic (LangGraph needs it) and forbidden in `domain/` by
`tests/architecture/test_layer_boundaries.py`. The graph keeps producing its
`CompatibilityAssessment` / `HumanDecision` inside a run; the domain dataclasses
are what the application ports ([M5-02]) and repositories ([M5-03]) speak in,
with the mappers between the two owned by [M5-03]. `Verdict` is the one type
already shared across both.

**Deviation on record.** (a) The DoD names the clause entity `Clause`; it ships
as `PolicyClause` so it does not shadow the existing 18-field parse-tree
`Clause` (`domain/clause_tree.py`) — the two are never imported together, and
both module docstrings state the split. (b) `domain/errors.py` centralises the
exception hierarchy, unlike the per-module `OrphanTextExceedsThresholdError`
pattern elsewhere in `domain/`: M5-01's errors are a cluster the API boundary
catches as one group. (c) `Assessment` omits consistency signals — no M5-01
invariant touches them; persisting `ConsistencyReport` is [M5-03]'s call. (d)
The ≥1-citation invariant is unconditional, so the compatibility node's
abstain-on-empty-retrieval output is not a persistable `Assessment` — it stays
in claim state and the audit trail, a fact [M5-02]/[M5-03] account for.
`ClauseProvenance` is left untouched; `Policy` is additive.

**Enforcement.** `tests/unit/domain/` — one `test_<module>.py` per entity, every
invariant with its rejection case (`test_assessment.py` on the ≥1-citation and
verdict-type rules, `test_human_decision.py` on the assessment reference and the
edit-consistency rules, `test_cnpj.py` parametrized over all 30 manifest CNPJs).
`tests/architecture/test_layer_boundaries.py` already AST-scans all of
`app/src/domain/` for a forbidden import and needs no change — the new
stdlib-only modules are covered automatically.
`tests/architecture/test_scope_vocabulary.py` now scans the whole `domain/`
package, not just `verdict.py`.

Full field tables, the invariant list and the relationship to `state.py`:
`docs/DOMAIN.md`.

---

## The application layer is ports + use cases; the orchestrator hides LangGraph — [M5-02]

**Decision.** `app/src/application/` gains the claim-assessment surface: five
ports (`app/src/application/ports/`), four use-case interactors
(`app/src/application/use_cases/`), and the DTOs they speak in
(`app/src/application/{assessment_record,orchestrator_result,consistency_flag,edited_assessment_input,errors}.py`).

- `ClauseRepository` — read-only lookup over the registered-product clause
  corpus. Stands apart from the transaction: the corpus is reference data the
  assessment use cases never write.
- `AssessmentRepository` — persists and queries `AssessmentRecord`. Its writes
  run inside a `UnitOfWork`; its reads do not.
- `UnitOfWork` — one transaction, exposing only `assessments`. The use cases
  take a `UnitOfWorkFactory` (`Callable[[], UnitOfWork]`), so each call opens
  its own unit — the session-per-transaction shape [M5-03] needs.
- `Clock` — `now() -> datetime` (tz-aware UTC). Formalises the
  `now=datetime.now(UTC)` parameter injection `consistency_checks.py` already
  uses, so the use cases' timestamps are assertable.
- `ClaimAssessmentOrchestrator` — `start(*, assessment_id, claim)` and
  `resume(*, assessment_id, decision)`, returning an `OrchestratorResult`.
  **No graph type crosses it**: no `ClaimState`, `Command`, `interrupt`,
  `thread_id`, or `infrastructure.graph` model. `assessment_id` is the sole run
  key (the implementation uses it as the LangGraph `thread_id`), so
  re-submitting a claim is simply a fresh `assessment_id` — the "one claim, a
  second thread" case `docs/HUMAN_CHECKPOINT.md` describes.

The use cases (`SubmitClaim`, `GetAssessment`, `SubmitHumanDecision`,
`ListAssessments`) are frozen-dataclass interactors — ports injected as fields,
one `__call__` — unlike the earlier pure-function pipeline use cases, because
they carry dependencies.

**Why `AssessmentRecord`, not `domain.Assessment`, is the stored unit.**
`Assessment`'s ≥1-citation invariant is unconditional by design (M5-01, above).
But the graph can finish an insufficient-context or clarification-exhausted run
with a recommendation that cites nothing, and `GET /v1/assessments/{id}` /
`ListAssessments` must still serve those — after the LangGraph thread, and its
checkpoint, may be gone. So the persisted/servable aggregate is
`application.assessment_record.AssessmentRecord`: the full lifecycle (system
verdict, prose, citations — possibly empty; the retrieval/clarification signals;
status; and, once settled, the analyst's `HumanDecision` recorded *beside* the
system's opinion). `AssessmentRecord.as_domain_assessment()` is the grounded
projection back — and it raises `CitationRequiredError` for an abstain record,
which is the invariant doing its job.

**Deviation on record.**
(a) `RetrievalService` is **not** delivered. No M5-02 use case consumes semantic
retrieval — the assessment use cases never search, and `Citation.excerpt`
already carries the hydrated clause text — so a `RetrievalService` port would be
a contract with no caller. This codebase lands a port with its first real
consumer (M4-04's `RetrievalPort`/`GraphRetrievalAdapter`, M4-09's
`AuditTrailSink`/`SqlAlchemyAuditTrailSink`), never ahead of one. Retrieval
stays internal to the graph (`infrastructure.graph.context.RetrievalPort`,
satisfied by `GraphRetrievalAdapter`), reached only through
`ClaimAssessmentOrchestrator`.
(b) The concrete `ClaimAssessmentOrchestrator` adapter (wrapping
`build_claim_graph`) and the real `Clock` (`SystemClock`) are **M5-04's**
composition root — M5-02 is fakes-only per its DoD ("contracts first, adapters
after"). The port shape is proven sufficient for that adapter: every
`OrchestratorResult` field maps to a concrete graph source, and both methods
reduce to one `compiled.invoke(...)` keyed by `assessment_id` (the verdict —
which `Recommendation` has no field for — is read from the recommendation
node's audit event).
(c) An `edit` decision always produces a *grounded* `Assessment` (≥1 citation,
every cited clause validated against `ClauseRepository` before the graph is
resumed). Keeping a 0-citation abstain unchanged is an `approve`, not an `edit`.
(d) The application import check
(`tests/architecture/test_layer_boundaries.py::test_application_imports_no_infrastructure_or_llm_sdk`)
adds `infrastructure`, `langchain_core` and `langchain_openai` to a
layer-scoped forbidden set — `_top_level_module` does not fold the latter two
into the existing `langchain` root, and LangGraph re-exports `langchain_core`
types, so "the orchestrator hides LangGraph entirely" needs all three barred.

**Enforcement.** `tests/unit/application/**` — every use case with its rejection
cases, driven through in-memory fakes (`tests/unit/application/fakes.py`), no
LLM, no database, no graph; `test_assessment_record.py` on the projection and
the status/decision pairing;
`tests/architecture/test_layer_boundaries.py::test_application_imports_no_infrastructure_or_llm_sdk`.

**Downstream.** [M5-03] implements the SQLAlchemy adapters behind these ports
(see the next section). [M5-04] adds the FastAPI endpoints, the LangGraph
orchestrator adapter, `SystemClock`, the `policy_ref` → retrieval-filter path,
and the domain/application-error → HTTP status mapping.

---

## Persistence adapters mirror the ports; the audit trail is append-only in the database — [M5-03]

**Decision.** `app/src/infrastructure/database/` gains the SQLAlchemy
implementations of the [M5-02] ports: `SqlAlchemyAssessmentRepository`,
`SqlAlchemyClauseRepository`, `SqlAlchemyUnitOfWork` (+
`sqlalchemy_unit_of_work_factory`), the `assessment` / `human_decision` tables
(`models.py`, migration `20260903_01`), and the row ↔ aggregate mapper
(`assessment_mapper.py`). `audit_event` becomes append-only at the database via
a `BEFORE UPDATE OR DELETE` trigger (migration `20260903_02`). Full column
rationale: `docs/DATABASE.md`.

**Why `citations` / `consistency_flags` are `JSONB`, not child tables.** They
are frozen value-object tuples with no identity, read whole with the record and
never queried by field — the same call as `audit_event.payload`. A normalised
child table buys nothing here and costs a join, an ordering column and a mapper
layer. `decisions` *is* its own table, as the DoD enumerates, because a decision
has a lifecycle (it settles a specific assessment) and its own foreign key.

**Why the `ClauseRepository` reads `chunk`.** There is no separate `clause`
table and [M5-02]'s port docstring already commits to this. A `PolicyClause` is
projected from the chunk rows whose `source_clause_ids` contains the wanted id —
the same reconstruction `graph_retrieval_adapter.build_clause_index` does, and
the id a `Citation` carries.

**Why a trigger, not a rule, for append-only.** An `ON UPDATE … DO INSTEAD
NOTHING` rule swallows the write silently; the trigger `RAISE`s, so a tamper
attempt fails loudly and the DoD's "cannot be updated" test can assert it. The
`INSERT … ON CONFLICT DO NOTHING` insert path never fires an `UPDATE`, so the
checkpoint node's idempotent re-write is unaffected.

**Two DoD items met differently, both recorded.** (a) "Alembic migrations: …
checkpointer tables" — met by `make setup-checkpointer` ([M4-09]), not a
migration duplicating `PostgresSaver.setup()`; the split is a settled decision
(`alembic/env.py`'s `UNMANAGED_TABLES` filter, `docs/DATABASE.md`). (b) "folds
the record write into the audit-sink transaction" — **deferred to [M5-04]**.
The audit sink writes inside the `human_review` graph node; wrapping that write
and the use case's record write in one transaction needs the orchestrator
adapter and the composition root to coordinate, neither of which exists until
[M5-04]. The `state.py` ↔ domain mapper (`docs/DOMAIN.md` "Deferred") is
[M5-04]'s for the same reason — no [M5-03] consumer.

**Enforcement.** `tests/unit/infrastructure/database/test_models.py` (the
`CHECK` sets tie back to the domain enums; column/nullable/index/FK assertions);
`tests/unit/infrastructure/database/test_assessment_mapper.py` (round-trip on
grounded / abstain / every decision outcome, no DB);
`tests/integration/test_assessment_repository.py`,
`test_clause_repository.py`, `test_unit_of_work.py`,
`test_audit_event_append_only.py` (real Postgres, one per repository plus the
append-only guarantee). `make test-integration` and CI's `integration` job pick
these up from the directory.

---

## The HTTP surface: FastAPI over the use cases, errors mapped at one edge — [M5-04]

**Decision.** `app/src/presentation/` gains the FastAPI app: `app.py`
(`create_app()` + the `lifespan` composition root), `dependencies.py`
(`Depends` providers building a use case per request off `app.state.components`),
`schemas.py` / `mappers.py` (Pydantic request/response models, pure
dataclass↔schema conversions), `errors.py` (the single error→HTTP edge), and
`routes/` (`assessments.py` — the five endpoints — and `health.py`). Three
infrastructure pieces the application layer was designed around land with it:
`infrastructure/graph/orchestrator.py`
(`LangGraphClaimAssessmentOrchestrator`, the concrete
`ClaimAssessmentOrchestrator` wrapping `build_claim_graph`),
`infrastructure/graph/state_mapper.py` (the `state.py` ↔ domain mapper — its
only consumer, so it is [M5-04]'s per `docs/DOMAIN.md`), and
`infrastructure/clock.py` (`SystemClock`). `infrastructure/graph/verdict_readout.py`
extracts the "read the verdict from the recommendation node's audit event"
logic that `scripts/eval_end_to_end.py` had inline; `infrastructure/rag/retriever_factory.py`
consolidates the ad-hoc retrieval-stack assembly the eval scripts each carried.
The new audit read model is a port (`AuditTrailReader`), a use case
(`GetAuditTrail`) and an adapter (`SqlAlchemyAuditTrailReader`), landing with its
first caller like every other port.

**Why.** (a) `POST /v1/assessments` returns **202** but runs the graph
synchronously in the handler for now — the real Redis queue and the
`PENDING/RUNNING/FAILED` run states are [M5-05]'s, which owns the queue; the id
in the 202 body and `Location` always resolves on `GET`. (b) `policy_ref` reaches
retrieval as a **text header** the adapter prepends to the narrative
(`[Apólice registrada: processo SUSEP …]`), byte-identical to the measured
headline arm of `scripts/eval_end_to_end.py::build_claim_text` — intake extracts
the process, `nodes/retrieval._build_filter` pre-filters on it, no graph change
and no eval re-run. (c) The orchestrator opens `open_claim_checkpointer` and a
fresh session **per call**; a pooled connection is "the M5 shape"
(`docs/DATABASE.md`), deferred to [M5-05]'s worker model. (d) The
`start`-must-pause / `resume`-must-finish contract stays the **use case's** to
enforce (per the port docstring) — the adapter reports `awaiting_review` as it
found it. (e) Errors map at one Starlette exception-handler layer: each
`application.errors` / `domain.errors` type → an HTTP status + a stable string
`code`, in the envelope `{"error": {"code", "message", "details"}}`; a client
branches on `code`, never on a status or a message.

**The transactional fold (the [M5-03]-deferred DoD item).** On the `resume` path
the composition root gives the graph a `_CapturingAuditSink` instead of the
committing `SqlAlchemyAuditTrailSink`: the `human_review` node's trail comes back
in `OrchestratorResult.audit_records`, and `SubmitHumanDecision` writes it
through the new `UnitOfWork.audit` writer in the **same transaction** as the
settled record — one `commit()`, both or neither. `resume` does no model calls
(it re-runs only `human_review`), so nothing long-running is pulled into the
transaction. The append is idempotent on `(thread_id, sequence)`, so the
self-healing retry (a second decision on a record still `AWAITING_REVIEW` whose
thread the first attempt finished) rewrites the same rows harmlessly. `start`
has no fold — the run pauses before it writes any trail; that window is
inherent (the checkpoint is a separate psycopg connection) and [M5-05]'s
run-status closes it.

**Deviation on record.** (a) `POST` blocks behind its 202 (above). (b)
`GET /v1/assessments/{id}/audit` returns `200 {"entries": []}` for an
`AWAITING_REVIEW` assessment — the durable trail is written once, in
`human_review`, after the decision, a deliberate [M4-09] design. (c) An analyst
`edit` drops the recommendation's consistency flags: an `EditedAssessmentInput` →
domain `Assessment` carries none, and flags are attention points kept beside the
verdict, not part of the decision ([M4-08]). (d) The `presentation/` layer is
not scanned by `test_layer_boundaries.py`; the `langgraph` import stays in
`infrastructure/`, and `domain` + `application` stay free of FastAPI/SQLAlchemy/
LangGraph as the M5 exit criteria require.

**Enforcement.** `tests/unit/presentation/**` (every endpoint's happy path and
every error→status mapping, driven through `tests/unit/application/fakes.py` via
`app.dependency_overrides` — no Postgres, graph or LLM);
`tests/unit/infrastructure/graph/test_{state_mapper,verdict_readout,orchestrator}.py`;
`tests/unit/infrastructure/test_clock.py`;
`tests/unit/application/test_get_audit_trail.py`;
`tests/unit/application/use_cases/test_submit_human_decision.py` (the fold — the
record and the trail roll back together). `tests/integration/test_assessment_api.py`
runs the whole flow against real Postgres (assessment/decision/audit tables + the
LangGraph checkpointer) with a fake model and stub retriever: submit → 202 → read
→ submit a decision → observe the resumed run and its durable trail.
`make test-integration` and CI's `integration` job pick it up; CI's `quality` job
runs the presentation unit tests.

**Downstream.** [M5-05] replaces the synchronous 202 with a Redis queue and adds
run-status states (below). [M5-06] **done** — `/ready` with per-check detail, one
JSON log line per event to stdout, and a correlation id accepted or minted per
request and propagated into every graph node log line and LLM call (via
`presentation/middleware.py`, `infrastructure/observability/`, and the `build.py`
node wrapper); see `docs/API.md`. [M5-09] adds the Compose `api` / `worker`
services, the Dockerfile and the CI image build, and wires the proxy-header /
trusted-host middleware from `ObservabilitySettings`.

---

## Asynchronous assessment: an RQ/Redis queue behind a non-blocking 202 — [M5-05]

**Decision.** `POST /v1/assessments` no longer runs the graph. `SubmitClaim`
persists an `application.assessment_job.AssessmentJob` (`PENDING`) and enqueues
it on an RQ queue; `RunAssessment` (a new use case) runs on
`ASSESSMENT_WORKER_CONCURRENCY` workers (`make worker` / `scripts/run_assessment_worker.py`),
drives the claim to the human checkpoint through the orchestrator port, and — on
success — writes the `AssessmentRecord` and flips the job to `SUCCEEDED` in one
transaction. New ports: `AssessmentQueue`, `AssessmentJobRepository` (added to
`UnitOfWork`). New infra: `infrastructure/queue/` (the RQ adapter, the job
function, the worker pool), `infrastructure/llm_errors.py` (the transient
classifier), `infrastructure/bootstrap.py` (the heavy singletons, now shared by
the API `lifespan` and the worker). `GET /v1/assessments/{id}` returns an
`AssessmentReadModel` spanning the lifecycle. Full walkthrough:
`docs/ASYNC_PROCESSING.md`.

**Why RQ, and why a separate aggregate.** RQ is the sync Redis-backed queue the
"reuse the queue pattern professionally" framing points at — native
`Retry(interval=[…])` for backoff, `FailedJobRegistry` for the dead-letter, and
`WorkerPool` for concurrency, with no second broker. The codebase is sync
throughout (SQLAlchemy/psycopg3, `compiled.invoke`, sentence-transformers), so a
sync worker is the honest shape. `AssessmentJob` is deliberately **not** part of
`AssessmentRecord`: the record's invariants (non-empty verdict/prose, the
≥1-citation projection) can't represent a claim that has not been assessed yet,
and weakening them to bolt on a `PENDING` state would undo M5-01/02. Two
aggregates, two tables; the read path composes them.

**Why retry stays at the job boundary.** The transient/real split
(`is_transient_llm_error` + RQ `Retry` + the `stop_retry_on_permanent` handler)
lives entirely in the queue layer. The six per-node `_invoke_with_retry` helpers
are untouched — they cover a mid-run blip (seconds); a sustained 429 now bubbles
past them to the job, which backs off for minutes, which is what
`docs/END_TO_END_EVALUATION.md` measured a rate limit needs. Keeping M4 node code
out of scope also keeps the M4 eval baselines reproducible.

**Deviations on record.** (a) `GET /v1/assessments` (list) stays record-only —
queued/failed jobs are not in the collection; the per-id `GET` is "status
tracking per assessment id". (b) `SubmitClaim` commits the job before enqueueing
it, so a crash in that window leaves a `PENDING` job that is never picked up — a
reconciler is left to operations. (c) The checkpointer connection is still
per-call, not pooled (`docs/DATABASE.md` "the M5 shape"). (d) `compose.yaml`
gains `redis`; the `api` / `worker` services and the Dockerfile are [M5-09].

**Enforcement.** `tests/unit/application/use_cases/test_run_assessment.py` (the
state machine), `test_submit_claim.py` (persist-then-enqueue),
`tests/unit/infrastructure/test_llm_errors.py` (the classifier truth table),
`tests/unit/infrastructure/queue/**` (the adapter and the retry gate),
`tests/unit/infrastructure/database/test_assessment_job_mapper.py`.
`tests/integration/test_assessment_job_repository.py` (real Postgres) and
`tests/integration/test_assessment_queue.py` (real Postgres + real Redis: submit
→ burst worker → completion, transient retry, dead-letter). `test_layer_boundaries.py`
adds `rq` / `redis` to the forbidden roots. CI's `integration` job gains a
`redis` service.

## Tracing is one callback handler on the run, plus two hand-written spans — [M5-07]

**Decision.** Trace the graph into a self-hosted Langfuse by installing its
LangChain `CallbackHandler` on the graph's run config in
`LangGraphClaimAssessmentOrchestrator._invoke`, and add exactly two spans by
hand for the work no LLM call passes through. `docs/OBSERVABILITY.md` is the
reader's guide; this section is the why.

**Why the orchestrator and not `build.py`.** [M5-06] already wraps every node in
`build._instrumented` for its log lines, so wrapping them again for spans is the
obvious move — and it is the wrong one. LangGraph nodes are runnables, so a
handler on the *run* already sees every node, and it sees something a node
wrapper never can: the `chain.invoke(messages)` calls **inside** the nodes,
with their prompts, completions and token usage. `_invoke` is also the only
place a whole run passes through, which makes it the only sensible flush point —
it covers the worker, the synchronous `resume` the API serves, and the eval
scripts at once. So `build.py`, all eight node modules and the [M4-01b]
`(state, runtime) -> dict` convention are untouched by this issue.

The handler must go on the run rather than on the model objects, because the
nodes deliberately pass no `config` to their own `chain.invoke` and rely on
LangChain's ambient child config; a handler bound to a model would be bypassed.

**Why two spans are still hand-written.** Callbacks see runnables. Retrieval is
deterministic Python — no LLM call — so to a callback handler the node is a
black box, and its candidate list, their scores and the [M3-07] gate's reasoning
are locals that mostly never reach state. Three gate fields (`threshold`,
`missing_category`, `closest_clause_ids`) are computed on every single run and
recorded *nowhere*: state keeps the boolean, the audit event keeps the trigger
name. Since those are the first thing you want when a verdict is wrong, the
retrieval node opens a span for them, and `GraphRetrievalAdapter` nests a second
one showing each candidate's hybrid rank against its cross-encoder rank.

**Why the graph layer owns a `TracePort`.** Same reason it owns `RetrievalPort`
and `AuditTrailSink`: a node depends on a capability, never on an adapter, and
`infrastructure.observability.tracing.LangfuseTracer` satisfies it structurally.
`infrastructure.rag` restates the same shape as `SpanRecorder` rather than
importing it, so the `graph -> rag` dependency direction is not reversed for a
span. The port deals in plain mappings and is a context manager, so the span's
latency is measured rather than reported, and a test fake is a handful of lines.

**A null object, not `| None`.** `GraphContext.tracer` defaults to `NO_TRACING`
rather than to `None` the way `audit_sink` does. The sink is consulted once, at
the checkpoint, where a branch is cheap and readable; a tracer is called from
node bodies, and the null object keeps those bodies free of `if tracer is not
None` noise for a call whose entire point is that it changes nothing. It also
means an untraced deployment runs the identical code path, not a second one.

**Optional means two things.** At the application layer,
`ObservabilitySettings.tracing_active` is the `TRACING_ENABLED` flag **and** both
keys; inactive, no Langfuse client is constructed at all. At the infrastructure
layer the four langfuse services sit behind a Compose `profiles: ["tracing"]`,
so plain `docker compose up -d` remains postgres + redis. And tracing never
raises into a run: every SDK call is guarded, so a broken tracer degrades to no
tracing, exactly as the compatibility node degrades rather than raising.

**Deviations on record.** (a) `GET /ready` deliberately does *not* check
Langfuse — readiness means "can serve an assessment", and an optional dependency
must not be able to take the service out of rotation. (b) Cost is registered as
list prices for the *pinned provider route*, not measured; re-pinning
`LLM_*_PROVIDER_ORDER` makes them stale, and [M5-10] owns the measured number.
(c) The langfuse services share this stack's Postgres (own database) and Redis
(db index 1 against the queue's db 0) instead of running their own — a
development-stack choice, stated rather than hidden. (d) The `langchain`
meta-package became a direct dependency: Langfuse's handler does
`import langchain` purely to branch on `langchain.__version__` and refuses to
import without it, though every symbol it uses comes from `langchain_core`.

**Enforcement.** `tests/unit/infrastructure/observability/test_tracing.py` runs
the **real compiled graph** against a real Langfuse client whose span exporter
writes to memory, and asserts a span per node, the retrieval span's candidates
and scores, the correlation id on the trace, and that a deliberately broken
tracer still runs the body it wraps — no server, no credentials, so it holds in
CI. `tests/unit/infrastructure/graph/test_retrieval.py` covers the retrieval
span's payload against a recording fake port.

---

## Retrieved and claimant text are data, never instructions — [M5-08]

**The threat model.** Every prompt in this system carries text from two
channels this project does not control: a clause excerpt extracted from a
third-party PDF (`docs/PARSING.md`), and the claim narrative a claimant typed.
Both reach an LLM call. Neither is sanitized for content — a filed insurance
document is full of imperative Portuguese by nature (*"o segurado é obrigado
a..."*, *"fica vedado..."*), so a keyword filter would misfire constantly and
solves the wrong problem anyway. The actual requirement is narrower and
mechanical: nothing read from either channel may change which instructions the
model follows, and nothing it decides may pick which document or clause gets
trusted. Three structural defenses, one empirical check.

**Defense 1 — delimiters, not a blocklist.** Every node prompt is built
through `prompts.scope_preamble.with_scope_preamble`, which now prepends
`prompts.untrusted_content.UNTRUSTED_CONTENT_NOTICE` alongside the scope
constraint — the same "one machine-enforceable copy" the scope preamble
already used, extended rather than duplicated. Every untrusted span — a
retrieved clause excerpt, the claim narrative, the entity facts intake
extracted from it, even the compatibility node's own reasoning when
`recommendation` summarises it — is wrapped by
`prompts.untrusted_content.wrap_untrusted` before it reaches a prompt:
`prompts.prompt_fragments.known_facts_block` / `clause_block` are the one
implementation every prompt builder calls (four near-duplicate private
helpers before this issue), and each of the five node functions wraps the
`HumanMessage` carrying `raw_claim_text` the same way. The notice tells the
model what the tag means; it does not filter content, so an imperative SUSEP
clause reads normally — it is simply never mistaken for an instruction to
*this* system.

**Defense 2 — reject malformed structured output, never coerce.** Every
`schemas.py` model now declares `extra="forbid"`: a field the model invents
fails validation rather than being silently dropped while the rest of the
response is coerced through. `compatibility.py` already rejected and retried
an ungrounded assertion ([M4-05]); `intake.py`'s bare `cast` on a `None`
parse — previously an uncontrolled `AttributeError` two lines later — now
raises `errors.SchemaValidationError` explicitly. The other three nodes'
existing "degrade to a deterministic fallback on a failed parse" behaviour
already satisfied this rule and needed no change: a fallback template is not
a coercion of the model's own malformed output, it is a different, safe path
taken instead of it.

**Defense 3 — document and clause trust is metadata-only, structurally.**
`retrieval._build_filter` builds the pre-filter from `entities` — intake's
deterministic classification of the claim, never a document id the model
named. `compatibility._grounding_errors` computes `valid_ids` from what
retrieval actually returned *before* the model call, and rejects and retries
any assertion whose `clause_ids` names anything else — an injected instruction
that tries to redirect the node to "cite clause 9.9 of document doc-99
instead" fails the same check as an ordinary hallucinated id, because it is
the same check. `recommendation.py` never constructs a `Citation` at all —
`RecommendationOutput` has no citation field, so there is no code path by
which the model could introduce one. There is no separate "ignore document
selection instructions" rule to maintain, because there is no code path that
reads one.

**The empirical check.** `make eval-prompt-injection`
(`scripts/eval_prompt_injection.py`) runs the real compatibility node, on the
real reasoning model, over four hand-authored adversarial fixtures
(`data/adversarial_injection/`): a poisoned clause excerpt demanding a
hijacked verdict, one additionally trying to name a foreign clause as more
authoritative, and a clean/injected claim-narrative pair for a
system-override and a role-change attempt. Results and method:
`docs/PROMPT_INJECTION.md`.

**Deferred.** The M5-08 issue's Appendix — an exploratory spike evaluating a
runtime prompt-injection classifier (`icephi`) as an optional, env-toggled
defense-in-depth layer — is explicitly out of this issue's scope and not
implemented here.

**Enforcement.**
`tests/unit/infrastructure/graph/prompts/test_untrusted_content.py` (fast,
network-free: every node prompt carries the notice, and a marker fed through
each untrusted argument appears only inside a `<untrusted-content>` span, in
every prompt builder);
`tests/unit/infrastructure/graph/test_schemas.py` (`extra="forbid"` rejects
an unexpected field, per schema);
`tests/unit/infrastructure/graph/test_intake.py` (`SchemaValidationError` on
a failed parse); `tests/unit/infrastructure/graph/test_compatibility.py`
(the existing grounding-retry tests, plus one naming the injection-defense
intent explicitly: a response that insists on a foreign document's clause id
is never trusted); `tests/eval/test_prompt_injection.py` (eval-marked, real
model, skips without `LLM_PROVIDER`).
