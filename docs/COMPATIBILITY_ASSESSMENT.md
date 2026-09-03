# Compatibility assessment — [M4-05]

The graph node that answers the actual question: given the clauses retrieval
returned and the claim, is the described event consistent with the registered
product's conditions? It reads `ClaimState.citations` (the typed `Citation`s the
retrieval node wrote) plus the extracted `entities`, calls the reasoning model
once, and writes a `state.CompatibilityAssessment` — `verdict` (the M0-06
vocabulary), `reasoning`, `citations`, `confidence`.

The rule that shapes the node: **every assertion in the reasoning must cite at
least one retrieved clause id.** A `compatible` / `incompatible` output with an
ungrounded assertion is malformed — the node rejects it, feeds the specific
error back, and retries; it is never patched after the fact.

Code:

- `app/src/infrastructure/graph/nodes/compatibility.py` — the node, the
  grounding-retry loop (`_invoke_grounded` / `_grounding_errors`), the
  no-context guard (`_abstain`), the reasoning renderer.
- `app/src/infrastructure/graph/schemas.py` — `CompatibilityOutput` /
  `ReasonedAssertion` (the `with_structured_output` shape) and
  `CompatibilityVerdict` (the `Literal` mapped onto `domain.verdict.Verdict`).
- `app/src/infrastructure/graph/prompts/compatibility.py` —
  `build_compatibility_prompt` (scope preamble + the numbered clause list + the
  grounding and exclusion-weighing rules).
- `app/src/infrastructure/graph/build.py` — `route_after_retrieval`'s `"assess"`
  key now enters `compatibility`; `compatibility → END`.

**Why measure it separately.** [M4-10] measures end-to-end verdict accuracy over
the synthetic claims with the full three-class confusion matrix. This document
measures the assessment node in isolation: handed realistic retrieved context,
does it reach the labelled verdict, and is every assertion it makes tied to a
clause? A regression here (a prompt that stops weighing exclusions, a grounding
check that lets an ungrounded citation through) would otherwise surface only as
a diffuse end-to-end drop.

**Scope.** This issue builds the node, wires the `"assess"` branch to it, and
measures it on the golden set. It does **not** build the consistency node
([M4-06]), the parallel branches ([M4-07] made the `"assess"` path a fixed
parallel fan-out to this node and the consistency node — see
`docs/PARALLEL_ASSESSMENT.md`), the recommendation node ([M4-08]), or the
checkpointer ([M4-09]). Full three-class verdict accuracy and the citation
coverage gate are [M4-10]'s.

---

## Method

`make eval-compatibility` (`scripts/eval_compatibility.py`). For every golden
question that carries a verdict label it runs the real retrieval node
(`GraphRetrievalAdapter` — hybrid RRF + cross-encoder rerank + exclusion
co-retrieval, `k = 10`, the [M3-08] best config, identical to
`docs/RETRIEVAL_NODE.md`) to populate `citations` and the [M3-07]
`context_sufficient` flag, then the real `compatibility` node on the reasoning
model (`LLM_MODEL_REASONING`, pinned to `LLM_REASONING_PROVIDER_ORDER`).

The question text is fed as `entities.description`, with `entities.susep_process`
/ `entities.product_line` from `data/policies/manifest.csv` for the question's
target document — the same LLM-free entity construction
`scripts/eval_retrieval_node.py` uses, so `_build_query` / `_build_filter` get
realistic inputs without an intake call.

The compatibility node is invoked **directly**, bypassing
`route_after_retrieval`, so the node's own `insufficient_information` judgement
is measured even on the `unanswerable` rows the gate would otherwise divert to
`END`.

**Ground truth.** `GoldenQuestion.expected_verdict`. Only two `question_type`s
carry one:

| question_type | n | label |
| --- | ---: | --- |
| `coverage_with_exclusion` | 19 | `incompatible` |
| `unanswerable` | 23 | `insufficient_information` |

`direct_lookup` / `definition` / `cross_document` questions are factual lookups
with no compatible/incompatible framing (`expected_verdict` is null) and are not
scored here.

**Reported**, overall and by `question_type`:

- verdict accuracy — the 3×3 confusion matrix (`compatible` / `incompatible` /
  `insufficient_information`), overall accuracy, per-class precision and recall;
- citation grounding — how many assessments degraded to
  `insufficient_information` because the model never grounded its reasoning
  after `MAX_GROUNDING_ATTEMPTS = 3`; every emitted assertion is clause-grounded
  by construction;
- reference-clause overlap for `coverage_with_exclusion` — how often the emitted
  citations include the labelled `reference_clause_ids` (bounded above by
  M4-04's 84.2 % Recall@10 for this type: the node can only cite an exclusion
  retrieval actually surfaced);
- the [M3-07] gate's `context_sufficient` alongside the node verdict, so the two
  abstention layers stay visible apart.

Needs `LLM_*` in `.env`, Postgres with loaded + embedded chunks, and the `embed`
uv group. Writes `eval/runs/compatibility.{md,json}` +
`eval/runs/compatibility_predictions.jsonl`.

### Limitations

- **No `compatible`-labelled question.** The golden set exercises the
  `incompatible` and `insufficient_information` verdicts only. The
  `compatible` branch of the node — and the confusion between `compatible` and
  the other two — is measured by [M4-10] over the synthetic claims.
- **The question is not a claim narrative.** Golden questions are phrased as a
  claims analyst would ask (`coverage_with_exclusion` rows are already
  scenario-shaped), but they are not the informal free text a policyholder
  writes. [M4-10] runs the node on real synthetic narratives.
- **Retrieval noise is folded in.** A wrong verdict can be a retrieval miss (the
  deciding clause was never returned) rather than a reasoning error; the
  reference-overlap number and the M4-04 Recall@10 ceiling bound this.

---

## Results

Run of 2026-09-02, `data/golden_set` verdict-labelled subset (42 questions),
reasoning model `deepseek/deepseek-v4-pro-0813` (provider order `['alibaba']`),
retriever = hybrid RRF + rerank + exclusion co-retrieval. 42/42 scored, 0
errors. Regenerate with `make eval-compatibility`; the run lands in
`eval/runs/compatibility.{md,json}`.

**Overall accuracy: 88.1%** (37/42).

### Verdict accuracy

| verdict | support | precision | recall |
| --- | ---: | ---: | ---: |
| compatible | 0 | — | — |
| incompatible | 19 | 100.0% | 73.7% |
| insufficient_information | 23 | 85.2% | 100.0% |

| expected \ predicted | compatible | incompatible | insufficient_information |
| --- | ---: | ---: | ---: |
| compatible | 0 | 0 | 0 |
| incompatible | 1 | 14 | 4 |
| insufficient_information | 0 | 0 | 23 |

By question type: `coverage_with_exclusion` **73.7%** (19),
`unanswerable` **100.0%** (23).

### Citation grounding

- Assessments degraded to `insufficient_information` (ungrounded after 3
  attempts): **3** — `coverage_with_exclusion-001`, `-014`, `-018`.
- Every emitted assertion cites a retrieved clause: **by construction**.

### Reference-clause overlap (`coverage_with_exclusion`)

- All labelled reference clauses cited: **57.9%** (19 questions).
- At least one reference clause cited: **84.2%**.

---

## Findings

- **The abstention is precise and never misses.** All 23 `unanswerable`
  questions came back `insufficient_information` — recall 100% — and the
  [M3-07] gate had already set `context_sufficient=False` on all 23, so the
  node was never asked to settle a question the corpus cannot answer. Nothing
  in the `incompatible` set was lost to over-abstention *by the gate*: it
  passed all 19.
- **What accuracy costs is over-abstention by the node, not by the gate.** Every
  one of the 5 errors is an `incompatible` question the node declined to settle
  (4 → `insufficient_information`) or got backwards (1 → `compatible`). Its
  `insufficient_information` precision, 85.2%, is exactly this: 4 abstentions
  that should have been verdicts.
- **3 of those 4 abstentions are the grounding contract working as designed.**
  `coverage_with_exclusion-001`, `-014` and `-018` degraded after three failed
  grounding attempts — the node could not tie its reasoning to a retrieved
  clause, so it refused to answer rather than assert something uncited. That is
  the [M4-05] DoD's rule producing a *wrong verdict on purpose*, and it is the
  right trade for a compliance setting: an ungrounded `incompatible` is worse
  than an honest abstention.
- **Citation quality is bounded by retrieval, as predicted.** The node cites
  every labelled reference clause on 57.9% of `coverage_with_exclusion`
  questions and at least one on 84.2% — the latter equal to [M4-04]'s Recall@10
  for that question type. The node cannot cite what retrieval did not surface,
  and it does not try.
- **This number is not the system's accuracy, and cannot be.** The golden set
  carries no `compatible`-labelled question, so a third of the vocabulary is
  untested here and the 88.1% is measured over a two-class problem. The
  three-class figure over the synthetic claims is [M4-10]'s
  (`docs/END_TO_END_EVALUATION.md`), and it is much lower — for reasons that
  live upstream of this node.
- **Device note.** This run's embedder and cross-encoder ran on CPU rather than
  the GPU, because [M4-10]'s end-to-end run held the GPU at the time. The two
  are numerically interchangeable here: CPU-computed embeddings match the stored
  GPU-computed vectors to a cosine of 0.99999998 (max element difference
  5.6e-05), far below any margin that could reorder a reranked list.

## What this means downstream

- **[M4-06]** (consistency node) is the other `"assess"`-branch node. [M4-07]
  fanned `"assess"` out to both with a fan-in; the `audit_trail` reducer makes
  the concurrent write well defined, and the two nodes write disjoint state
  channels (`compatibility` vs `consistency`). See `docs/PARALLEL_ASSESSMENT.md`.
- **[M4-08]** (recommendation) consumes `state.compatibility`. Its citations may
  only be ones an upstream node produced — the compatibility node's `citations`
  are already a subset of `state.citations`, so that invariant holds by
  construction here. An `insufficient_information` verdict must not become a
  confident recommendation ([M4-08] tests this).
- **[M4-10]** measures the three-class verdict accuracy this document cannot
  (no `compatible` label) over the 51 synthetic claims, and the automated
  citation-coverage check (every assertion carries a clause id that exists in
  the corpus).
