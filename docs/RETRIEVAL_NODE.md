# Retrieval node — [M4-04]

The graph node that turns the intake node's structured entities into retrieved
clauses. It builds a retrieval query from the *entities* (not the raw claim
text), applies the classification's metadata pre-filter, calls the retrieval
port, hydrates a typed `state.Citation` per hit, and sets `context_sufficient`
from the [M3-07] insufficient-context gate. The router `route_after_retrieval`
in `infrastructure/graph/build.py` branches on that flag.

Code:

- `app/src/infrastructure/graph/nodes/retrieval.py` — the node, plus
  `_build_query` (joins `event_type` / `description` / `vehicle_info`) and
  `_build_filter` (SUSEP process when stated, else the product line — the
  [M4-10] amendment below replaced the original conjunction of the two;
  `None` when neither is known).
- `app/src/infrastructure/rag/retrieved_clause.py` — `RetrievedClause`, the
  port's return type (clause id + provenance + reranker score).
- `app/src/infrastructure/rag/graph_retrieval_adapter.py` —
  `GraphRetrievalAdapter`, the concrete `RetrievalPort`: hybrid RRF →
  cross-encoder rerank → optional exclusion co-retrieval → hydrate from the
  chunk corpus.
- `app/src/infrastructure/graph/build.py` — `add_node("retrieval", …)`, the
  `"proceed"` branch now enters retrieval, `route_after_retrieval`.

**Why measure it separately.** [M4-10] measures end-to-end verdict accuracy over
the synthetic claims. This document measures the retrieval node in isolation:
does it wire the M3 pipeline and the [M3-07] gate correctly, and does routing a
golden question through the node reproduce the M3 retrieval numbers? A
regression here (a dropped filter, a mis-mapped citation field, a gate signal
assembled wrong) would otherwise only surface as a diffuse end-to-end accuracy
drop in [M4-10].

**Scope.** The node depends only on `RetrievalPort`; swapping the retriever does
not touch the graph. What this issue does **not** own:

- The composition root that injects the real `GraphRetrievalAdapter` into
  `GraphContext` for the running service — [M4-09] / M5. Today the adapter is
  built by this eval script and by tests.
- The insurer CNPJ half of the pre-filter. Intake extracts a SUSEP process only
  "if stated"; it never extracts a CNPJ. So the node's filter is SUSEP process
  (often absent), else the product line — see the [M4-10] amendment below, which
  replaced the original conjunction of the two — one field weaker than the M3
  eval harness's SUSEP-process + CNPJ default path. On `golden-set-v1` this costs nothing (see
  below), because within a SUSEP process the CNPJ is redundant; a real claim
  with no stated process falls back to the product-line-only or unconstrained
  path, which [M4-10] will exercise.
- End-to-end verdict accuracy and citation coverage — [M4-10].

---

## Method

`make eval-retrieval-node` (`scripts/eval_retrieval_node.py`). Runs the real
`retrieval` node over every `data/golden_set/*.jsonl` question. LLM-free: each
question is fed as `entities.description`, with `entities.susep_process` and
`entities.product_line` taken from `data/policies/manifest.csv` for the
question's target document — so `_build_query` returns the question text and
`_build_filter` returns that document's process (plus, before the [M4-10]
amendment below, its product line; within one process the two select the same
chunks, so the measured numbers are the same either way). The
entity→query composition (multi-field join, fallback, null handling) is
unit-tested in `tests/unit/infrastructure/graph/test_retrieval.py`; this run
measures retrieval quality and the gate wiring.

Retriever: `GraphRetrievalAdapter` over hybrid RRF (`FusionStrategy.RRF`,
`CANDIDATE_DEPTH` per leg) + cross-encoder rerank
(`Alibaba-NLP/gte-multilingual-reranker-base`, `RERANK_CANDIDATE_DEPTH = 10`) +
exclusion co-retrieval (`RESERVED_EXCLUSION_SLOTS = 1`) — the [M3-08] best
configuration. `k = RETRIEVAL_K = 10`.

Metrics over the scorable questions (every type except `unanswerable`, which
carry no `reference_clause_ids`): Recall@10, MRR, nDCG@10, and the
foreign-document rate (fraction of returned citations whose `document_id` is not
the question's). Over the 23 `unanswerable` questions: gate recall (fraction the
node flags `context_sufficient = False`). Over the scorable questions: gate
false-abstention rate.

Ground truth is `reference_clause_ids` for the scorable set and the
`unanswerable` label for the gate — the same labels [M3-04] and [M3-07] scored
against. Needs Postgres with loaded + embedded chunks and the `embed` uv group.

---

## Results

Run of 2026-09-01, `golden-set-v1` (140 questions: 117 scorable, 23
unanswerable), CPU reranker. 140/140 processed, 0 exceptions.

### Retrieval quality (117 scorable questions)

| metric | value |
| --- | --- |
| Recall@10 | **92.3%** |
| MRR | **0.806** |
| nDCG@10 | 0.823 |
| Foreign-document rate | **0.0%** |
| Mean citations returned | 9.9 |

| question_type | n | Recall@10 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| direct_lookup | 64 | 95.3% | 0.851 | 0.872 |
| cross_document | 16 | 93.8% | 0.802 | 0.837 |
| definition | 18 | 88.9% | 0.815 | 0.833 |
| coverage_with_exclusion | 19 | 84.2% | 0.653 | 0.640 |

These are the [M3-08] benchmark-matrix numbers for the best configuration
(Recall@10 92.3%, MRR 0.806) reproduced through the graph node — the node adds
no retrieval loss, and the product-line filter it applies (which the M3 harness
does not) costs nothing on this set.

### Insufficient-context gate

| metric | value |
| --- | --- |
| Gate recall over `unanswerable` | **100.0%** (23/23) |
| Unanswerable missed | none |
| False-abstention rate over scorable | **0.0%** (0/117) |

Matches the [M3-07] calibration exactly (100% recall / 100% precision on
`golden-set-v1`). The node assembles the `GateSignals` from the same hybrid +
rerank pipeline the calibration used; exclusion co-retrieval, which the node
also runs, never re-scores rank-1, so the gate's signal is unchanged.

---

## Findings

- **The node is transparent to retrieval quality.** Every scorable-set number
  equals the [M3-08] best-config number. Query = the question text, filter =
  process + product line, `k = 10`: no clause is lost between the port and the
  `Citation` list.
- **`coverage_with_exclusion` stays the weak type** (84.2% Recall@10), as in
  [M3-05]/[M3-06] — the coverage-side gap co-retrieval cannot close. Nothing
  the node does changes this; it is a retrieval-pipeline property.
- **The gate wiring is correct.** `InsufficientContextResult.sufficient` →
  `context_sufficient` → `route_after_retrieval`, and all 23 unanswerable
  questions reach `END` via the `"insufficient"` branch.
- **`golden-set-v1` cannot separate the CNPJ filter's value.** Every question
  targets one document and the process/CNPJ pair is 1:1 within it, so dropping
  CNPJ from the node's filter is invisible here. The claims path where a
  process is not stated is [M4-10]'s to measure.

## Amended by [M4-10]: the filter is no longer a conjunction

This document described `_build_filter` as "SUSEP process when stated plus the
product line". [M4-10]'s end-to-end run over the synthetic claims showed that
ANDing them is a defect, and the node was changed: **a stated process now wins
alone**, and the product line constrains only the no-process fallback.

The reason `golden-set-v1` could not see this is the same reason it cannot
separate the CNPJ filter's value, stated above. Every golden question targets
one document *and* carries the manifest's own product line, so intake's
classification and the process never disagree there. They disagree exactly on a
product/claim mismatch — a CASCO-type event filed against an RCF-A / ASSIST /
GAR.EST / CARTA VERDE policy — which `golden-set-v1` contains none of and
`data/synthetic_claims/product_claim_mismatch.jsonl` contains eleven of. Under
the conjunction all eleven selected **zero chunks** before any ranking ran.

Nothing in this document's measured numbers changes: on `golden-set-v1` the two
filters select the same chunks, so the Results above still stand as the node's
retrieval-transparency check. The reasoning and the before/after are in
`docs/END_TO_END_EVALUATION.md`; the rule itself is in `docs/ARCHITECTURE.md`.

## What this means downstream

- [M4-05] (compatibility assessment) receives up to 10 `Citation`s with real
  excerpts (`display_text`) and reranker relevance scores, and a
  `context_sufficient` flag it must honour — a `False` there means the
  assessment node should not be reached with a confident verdict.
- [M4-07] made the sufficient-context path a fixed parallel fan-out to the
  compatibility and consistency nodes (`docs/PARALLEL_ASSESSMENT.md`); [M4-08]
  routes the `"insufficient"` path straight to the recommendation node, which
  turns `context_sufficient = False` into an `insufficient_information`
  recommendation (`docs/RECOMMENDATION_NODE.md`).
- [M4-09] / M5 build the composition root that hands a live
  `GraphRetrievalAdapter` to `GraphContext`; until then the adapter is
  constructed by `scripts/eval_retrieval_node.py` and the tests.
