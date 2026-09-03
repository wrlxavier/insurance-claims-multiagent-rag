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
