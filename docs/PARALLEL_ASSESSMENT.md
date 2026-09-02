# The parallel assessment branches — [M4-07]

The compatibility node ([M4-05]) and the consistency node ([M4-06]) do not
depend on each other: one grounds a coverage verdict in retrieved clauses, the
other reads the claim for internal consistency. [M4-07] runs them **concurrently**
as *fixed parallel branches* off the retrieval node — one superstep, both nodes —
converging on `END` (the recommendation node once [M4-08] lands).

```
                   ┌─ compatibility ─┐
retrieval ─(gate)─►┤                 ├─► END   (→ recommendation, M4-08)
                   └─ consistency ───┘
```

Code: `app/src/infrastructure/graph/build.py` — `route_after_retrieval` and the
`add_conditional_edges` / `add_edge` wiring. Nothing else changed: the two node
modules, the state schema and its reducer were built for this.

## The primitive: fixed parallel branches, not `Send`

LangGraph has two fan-out mechanisms:

- **Fixed parallel branches** — a node (or a conditional edge) with edges to two
  or more *known* nodes. They run in the same superstep; a common successor with
  edges from all of them is the fan-in.
- **`Send`** — dynamic dispatch: a conditional edge returns
  `[Send("node", state), …]` for a number of items known only at runtime, each
  invoking the target with its own payload. This is the map-reduce primitive.

This is the first case: two named nodes, always both. `route_after_retrieval`
returns `["compatibility", "consistency"]` when the [M3-07] gate passed (a list
of node names is LangGraph's fan-out), or `[END]` when it did not. Both
assessment nodes then edge to `END`.

An earlier draft of the project's design document
(`.ai_context/Assistente_Sinistros_Apolices_Proposta_Completa_com_ERRATA.md`
sec. 6.3) specified `Send` here. That is wrong — its own ERRATA and
`state.py`'s module docstring already record the correction. Verified against
the current LangGraph documentation (`docs.langchain.com/oss/python/langgraph/graph-api`)
and the installed **langgraph 1.2.11**.

## Concurrent state writes are well defined

The two nodes write **disjoint** channels — `compatibility`
(`CompatibilityAssessment`) and `consistency` (`ConsistencyReport`) — so the
default `LastValue` channel receives exactly one write each.

The one channel both write in the superstep is `audit_trail`. It carries the
`append_audit_events` reducer (`state.py`), which concatenates
accumulated-first, so both branches' events survive in a deterministic order.
Without a reducer LangGraph raises `InvalidUpdateError` on the concurrent write —
`tests/unit/infrastructure/graph/test_state_merges.py` proves both the merge and
that bare channel's failure.

## A failure in one branch is loud, not silent

With no `error_handler` registered (and none is, on purpose), if one branch
raises, LangGraph's runner cancels the sibling and **re-raises out of
`.invoke()`**. `apply_writes` never runs for that superstep, so no partial state
dict — `consistency` populated, `compatibility` silently missing — is ever
returned. The failure surfaces; it does not truncate.

`tests/unit/infrastructure/graph/test_claim_graph.py::test_a_failure_in_one_assessment_branch_is_not_silently_swallowed`
drives the compiled graph with a reasoning model that always fails and asserts
the raise.

(The compatibility node still degrades an *ungroundable* answer to
`insufficient_information` internally — that is a valid result, not a failure.
Only a genuine exception, e.g. the provider being unreachable after retries,
aborts the superstep. Branch-level recovery via `add_node(error_handler=…)` is
[M4-09]'s concern, alongside the checkpointer.)

## Wall-clock gain

`scripts/eval_parallel_assessment.py` (`make eval-parallel-assessment`) runs the
[M4-07] fan-out/fan-in **once** per synthetic claim, over a pre-populated state
(entities from the claim narrative, citations from the claim document's
coverage/exclusion chunks — no ANN retriever, the retrieval step is identical
either way), with each node wrapped to record its own wall time:

- **parallel** = the measured wall of that single `.invoke` (both nodes in one
  superstep, on the sync runner's thread pool);
- **sequential** = `t_compatibility + t_consistency` from the same run — what the
  identical work costs back to back.

Each LLM call is issued exactly once, so there is no grounding-retry drift
between calls (the compatibility node retries an ungrounded answer) and no
provider prompt-cache artifact (a repeated identical prompt returns faster) to
distort the comparison. Regenerable; raw run in `eval/runs/parallel_assessment.md`.

<!-- RESULTS:BEGIN -->
**Run — 2026-09-02, 6 synthetic claims** spanning the `compatible`,
`insufficient_information` and `product_claim_mismatch` cohorts (seed-0 shuffle,
`--limit 6`); reasoning model `deepseek-v4-pro` (compatibility), fast model
`deepseek-v4-flash` (consistency), via OpenRouter. 0 errors, 0 parity failures.

| | sequential | parallel | saving |
| --- | ---: | ---: | ---: |
| total (6 claims) | 508.7 s | 376.8 s | **131.9 s (25.9%)** |
| mean / claim | 84.8 s | 62.8 s | 22.0 s |

- mean compatibility call (reasoning model): **62.8 s**
- mean consistency call (fast model): **22.0 s**

The mean saving (22.0 s) is the consistency call almost exactly — it runs fully
hidden under the longer compatibility call, every claim. The **percentage** gain
(25.9%) is a function of this environment's slow reasoning model: with
`t_compat ≈ 63 s`, `t_consist / (t_compat + t_consist) ≈ 26%`. On a faster
reasoning model the compatibility call shrinks, the consistency call does not,
and the percentage rises — but the **absolute** saving stays ≈ one consistency
call. Regenerate over the full claim set with `make eval-parallel-assessment`
(no `--limit`).
<!-- RESULTS:END -->

**Why the gain is what it is.** The two calls run in one superstep, so the
parallel wall is about `max(t_compat, t_consist)` plus overhead and the saving
is bounded by `min(t_compat, t_consist)`. The consistency node deliberately uses
the *fast* model (see `docs/CONSISTENCY_NODE.md`), so `t_consist < t_compat` and
the parallel branch costs roughly one reasoning-model call — the consistency
check is close to free on the critical path.

## Scope

- The recommendation node — [M4-08]. Until then both branches edge to `END`;
  [M4-08] repoints them at `"recommendation"`, which then runs once after both
  finish.
- The Postgres checkpointer, `interrupt()`, and `error_handler`-based branch
  recovery — [M4-09].
- End-to-end verdict accuracy over the synthetic claims — [M4-10].
