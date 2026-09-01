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
   see [M4-07] and [M4-09].

**Enforcement.** `tests/architecture/test_graph_node_conventions.py`
(unit-marked, runs in CI) fails on any class under `nodes/` that defines
`__call__` or subclasses a `Runnable*`, and on any exported node function whose
first two parameters are not `(state, runtime)`.
`tests/architecture/test_scope_vocabulary.py` already scans the same tree for
verdict-vocabulary drift.

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
on that flag; `"assess"` enters the compatibility node ([M4-05], which [M4-07]
widens into a fan-out), `"insufficient"` terminates at `END` until [M4-08]
consumes it.

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

---

## The compatibility node grounds every assertion in a retrieved clause — [M4-05]

**Decision.** The compatibility node
(`app/src/infrastructure/graph/nodes/compatibility.py`) is the node that answers
the question. It reads `state.citations` and `state.entities`, calls the
**reasoning** model (`GraphContext.reasoning_model`, pinned by
`LlmSettings.llm_reasoning_provider_order` — [M4-05] is its first consumer),
structured output into `schemas.CompatibilityOutput`, and writes a
`state.CompatibilityAssessment` (verdict / reasoning / citations / confidence).
`route_after_retrieval`'s `"assess"` key now enters it; `compatibility → END`.
[M4-07] widens that single edge into the fan-out to the compatibility and
consistency nodes.

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
