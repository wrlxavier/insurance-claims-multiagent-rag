# The recommendation node — [M4-08]

The graph's single terminal node. Every path reaches it — the sufficient-context
path after the compatibility ([M4-05]) and consistency ([M4-06]) branches
converge, the insufficient-retrieval path (`context_sufficient is False`), and
the exhausted clarification loop (`clarification_exhausted`) — and it emits one
`state.Recommendation` for a human reviewer: a recommended action, a scannable
justification (verdict → clauses → caveats), the aggregated citations, the
consistency flags kept **separate** from the verdict, and a confidence.

**It consolidates; it does not re-decide.** Everything load-bearing is computed
in Python from upstream state. The fast model writes only the `justification`
paragraph, and only when a real compatibility assessment exists.

Code:

- `app/src/infrastructure/graph/nodes/recommendation.py` — the node, plus
  `_posture` (the effective verdict + why we are here), `_aggregate_citations`
  (dedupe `compatibility.citations`), `_confidence` (derive + clamp),
  `_recommended_action` (fixed template per posture), `_draft_justification`
  (the fast-model call, degrades to `""`), `_fallback_justification` (the
  deterministic template).
- `app/src/infrastructure/graph/schemas.py` — `RecommendationOutput`, carrying
  **only** `justification`. No citation field, no verdict field, no confidence
  field — so the model can neither invent a citation nor overstate confidence.
- `app/src/infrastructure/graph/prompts/recommendation.py` —
  `build_recommendation_prompt`, wrapped in `with_scope_preamble`.
- `app/src/infrastructure/graph/build.py` — `add_node("recommendation", …)`;
  `route_after_retrieval` returns `["recommendation"]` on the insufficient path;
  `compatibility` / `consistency` / `clarification_exhausted` all edge to it;
  `recommendation -> END`.

**Why measure it separately.** [M4-10] measures end-to-end verdict accuracy and
runs a faithfulness judge. This document measures the recommendation node in
isolation: does the consolidation preserve the two guarantees the DoD makes
structural — no citation the upstream nodes did not produce, and no confident
recommendation on an `insufficient_information` verdict — on live output, and how
do the posture, the confidence and the flags behave across the golden set. A
regression here (a citation leaking in, an abstention presented confidently, a
flag dropped) would otherwise surface only as diffuse noise in [M4-10].

**Scope.** What this issue does **not** own:

- The human checkpoint: `interrupt()` before the decision is recorded, the
  approve/edit/reject capture, the Postgres checkpointer — [M4-09]. The
  recommendation node is side-effect-free and produces the artefact [M4-09]
  surfaces.
- End-to-end verdict accuracy, the confusion matrix, the RAGAS / LLM-judge
  faithfulness measurement — [M4-10].
- The API response schema that exposes the recommendation — M5.

---

## What the node decides deterministically vs. asks the model

The rule: **the verdict, the action, the clauses and the confidence are
Python; the prose paragraph a reviewer reads is the model's.**

| Field | Source | How |
| --- | --- | --- |
| `recommended_action` | deterministic | One fixed template per *posture* (`claimant_gaps` / `retrieval_miss` / `compatible` / `incompatible` / `inconclusive` / `no_assessment`). Framed as "route to a human reviewer" / "ask the claimant" — never a real-world coverage outcome. |
| `citations` | deterministic | `compatibility.citations` deduplicated by `clause_id`, order preserved. `[]` when there is no assessment. The node builds no `Citation`; [M4-05] already guarantees these are a retrieved subset. |
| `consistency_flags` | deterministic | `consistency.signals` verbatim. Its own field; never merged into the verdict or the action. |
| `confidence` | deterministic | `compatibility.confidence` (0.0 with no assessment), then clamped: `insufficient_information` effective verdict → ≤ 0.3; an unresolved `attention` flag → ≤ 0.7. |
| `justification` | **fast model** | One paragraph, Brazilian Portuguese: compatibility finding first, then the clause ids, then the consistency caveats. Only on the assessed path; the claimant-gaps / retrieval-miss paths and any model failure use a deterministic template. |

**The posture** is the effective verdict plus why we are at the node. Precedence:
`clarification_exhausted` (claimant never supplied enough) outranks
`context_sufficient is False` (retrieval missed), which outranks a real
compatibility verdict. The first two are always `insufficient_information`.

**Not attempted, by design.** The node does not re-run any assessment, weigh the
consistency signals against the verdict, or produce a coverage decision. A
consistency signal is an attention point for the reviewer, never a verdict input
(per [M4-06]); the recommendation confidence is capped by an unresolved
`attention` flag but the verdict itself is untouched.

---

## Method

- **Unit** — `tests/unit/infrastructure/graph/test_recommendation.py`
  (`pytest -m unit`, fake fast model, in CI): the citation-subset invariant
  across every path (including a model response that names a fake clause id),
  the confidence clamps, the flag pass-through, the degrade-to-template path,
  and the no-model-call short-circuits on the claimant-gaps / retrieval-miss
  paths. `tests/unit/infrastructure/graph/test_claim_graph.py` asserts every
  terminal path converges on the node.
- **Whole node (live)** — `make eval-recommendation`
  (`scripts/eval_recommendation.py`) runs the real retrieval + compatibility +
  consistency + recommendation nodes over the 42 verdict-labelled golden
  questions (`coverage_with_exclusion` + `unanswerable`), reasoning model for
  compatibility, fast model for consistency and the justification. Needs
  Postgres with loaded + embedded chunks, the `embed` uv group, and `LLM_*` in
  `.env`. Writes `eval/runs/recommendation.{md,json}` +
  `recommendation_predictions.jsonl`.

Reported: the citation-grounding rate (expected 100%), the max confidence on an
`insufficient_information` posture (expected ≤ 0.3), the consistency-flag
pass-through rate (expected 100%), the posture-derivation wiring check (the
audit posture is `_VERDICT_POSTURE` of the compatibility verdict — expected
100%), the confidence distribution by posture, and a light justification check
(does the prose name a retrieved clause id when the verdict is settled).
`tests/eval/test_recommendation_baseline.py` asserts the three structural rates
strictly — they are node invariants, not bets.

### Limitations

- **No per-recommendation ground truth.** The golden set has no labelled
  "correct recommendation"; the eval reports whether the structural guarantees
  hold and how the derived fields distribute, not accuracy.
- **The golden set has no `compatible` label and no incomplete claims, and the
  eval runs the compatibility node directly.** Every labelled question is
  `incompatible` or `insufficient_information`, and the recommendation node is
  fed the compatibility assessment on every row (the assessed path). The
  `compatible`, `claimant_gaps` and `retrieval_miss` postures are exercised only
  by the unit tests and by [M4-10]'s synthetic claims.
- **Justification faithfulness is checked only shallowly here** (does it name a
  retrieved clause id). A real faithfulness judge over the consolidated output
  is [M4-10].

---

## Results

Run of 2026-09-02, `data/golden_set` verdict-labelled subset (42 questions: 19
`coverage_with_exclusion`, 23 `unanswerable`), reasoning model
`deepseek/deepseek-v4-pro-0813`, fast model `deepseek/deepseek-v4-flash-0731`,
CPU reranker. 42/42 scored, 0 errors.

### Structural guarantees (re-checked live)

| guarantee | value |
| --- | --- |
| Citation-grounding rate | **100.0%** (42/42) |
| Max confidence on an `insufficient_information` posture | **0.30** (ceiling 0.30) |
| Consistency-flag pass-through rate | **100.0%** (42/42) |
| Posture is the compatibility verdict's mapping (wiring) | **100.0%** (42/42) |

### Derived-field distribution

| posture | n | confidence min / mean / max |
| --- | ---: | --- |
| `incompatible` | 16 | 0.70 / 0.88 / 1.00 |
| `inconclusive` | 26 | 0.00 / 0.24 / 0.30 |

- `inconclusive` (26): every row ≤ 0.30 — 19 clamped down to exactly 0.30 from a
  higher compatibility confidence, 5 at 0.00, 2 in between.
- `incompatible` **with ≥ 1 `attention` flag** (5): confidence capped at exactly
  0.70 every time.
- `incompatible` **with no flag** (11): confidence passes through untouched,
  0.90–1.00.
- Justification names a retrieved clause id on a settled verdict: **43.8%**
  (7/16).
- LLM justification degraded to the deterministic template: **1** row
  (`unanswerable-002`).

No `compatible`, `claimant_gaps`, `retrieval_miss` or `no_assessment` posture
occurred — the golden set carries no `compatible` label and the eval runs the
assessed path only (see Limitations).

---

## Findings

- **The two DoD guarantees hold on live output, not just in the fakes.** Every
  recommendation clause id (across 42 real reasoning-model assessments) was one
  retrieval returned, and no `insufficient_information` posture carried a
  confidence above 0.30. The node builds no `Citation` and `RecommendationOutput`
  has no confidence field, so this is structural — the live run is a regression
  check, and it is clean.
- **Both confidence ceilings bind and are visible in the data.** The
  `insufficient_information` clamp pulls 19 of 26 `inconclusive` rows down to
  0.30 (the compatibility node's own confidence on those was higher); the
  unresolved-`attention`-flag clamp pins all 5 flagged `incompatible` rows to
  exactly 0.70 while the 11 unflagged ones keep their 0.90–1.00. The clamps are
  not cosmetic.
- **Consistency stays separate.** 18 signals across 16 rows, all carried verbatim
  into `consistency_flags`; none changed a `recommended_action` or a posture. A
  flag caps confidence and nothing else, exactly as [M4-06] requires.
- **The degrade-never-raise path fired once for real.** On `unanswerable-002` the
  fast model's justification call did not yield usable prose (a transient failure
  or a blank parse, after the retries); the node fell back to the deterministic
  template — a full paragraph that still names the two grounded clause ids —
  recorded `model=None` in the audit event, and emitted a well-formed
  `Recommendation` at confidence 0.30. No exception reached the graph.
- **The justification prose under-cites.** On settled verdicts it names an
  explicit clause id only 43.8% of the time — the fast model tends to describe
  the exclusion ("a exclusão para danos decorrentes de desgaste") rather than
  paste its id, even though the id sits in the structured `citations` list beside
  it. This is not a grounding risk (the `citations` field is structural and was
  100% grounded), but the paragraph a reviewer reads would be crisper if it
  quoted ids. A real faithfulness judge over the justification is [M4-10]; if it
  confirms the pattern, the prompt should insist on the id inline.

---

## What this means downstream

- **[M4-09]** pauses with `interrupt()` on the `state.Recommendation` this node
  produced, then records the analyst's approve/edit/reject alongside it — never
  overwriting it. The node is side-effect-free, so a re-run after the interrupt
  reproduces the same artefact.
- **[M4-10]** runs the whole graph over the synthetic claims and scores the
  final recommendation: the three verdict classes with a confusion matrix, the
  product/claim mismatch subset, the "100% of assertions carry a corpus clause
  id" check (which the node's citation aggregation makes structural), and a
  faithfulness + context-relevance judge over the justification.
