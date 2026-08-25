# Evaluation

This document is written incrementally across M2 and frozen at `golden-set-v1`
by [M2-07]. [M2-01] adds its first section: the golden-set schema and the
authoring protocol, fixed *before* any of the 90+ golden questions are
written, so question 90 is comparable to question 1.

## The golden-set schema

One row per question, stored as JSONL under `data/golden_set/`, one file per
`question_type` (`data/golden_set/<question_type>.jsonl`). Schema:
[`infrastructure.evaluation.golden_set_schema.GoldenQuestion`](../app/src/infrastructure/evaluation/golden_set_schema.py).

| Field | Type | Notes |
| --- | --- | --- |
| `schema_version` | str | Currently `"v1"`. |
| `question_id` | str | `{question_type}-{3-digit sequence}`, e.g. `direct_lookup-001`. Unique across the whole golden set. |
| `document_id` | str | The target document. Must exist in `data/policies/manifest.csv`. |
| `question` | str | Phrased as a claims analyst would ask — never echoes the referenced clause's wording. |
| `reference_clause_ids` | list[str] | Exhaustive: every clause id needed to answer the question, no more, no fewer. Must exist in the parsed corpus (`build/parsed_clauses.jsonl`). Empty only for `unanswerable` questions. |
| `question_type` | enum | `direct_lookup`, `coverage_with_exclusion`, `cross_document`, `unanswerable`, `definition`. |
| `difficulty` | enum | `easy`, `medium`, `hard` — how hard the retrieval/reasoning step is *within* a `question_type`. Adversarial intent is already carried by `question_type`; `difficulty` is not a second way of encoding that. |
| `expected_verdict` | enum or null | `compatible`, `incompatible`, `insufficient_information` — the same vocabulary every agent verdict uses (`SCOPE_PREAMBLE`, `app/src/infrastructure/graph/prompts/scope_preamble.py`). Null for `direct_lookup`/`definition` questions, which are factual lookups with no compatible/incompatible framing. `unanswerable` questions are always `insufficient_information`. |
| `notes` | str | Free text: rationale, what specifically the question is expected to expose. |
| `authored_at` | str or null | ISO 8601 date. Needed for [M2-02]'s blind re-labelling pass (≥14 days after first authoring). |

A retriever that returns only half of `reference_clause_ids` is measurably
wrong — completeness is part of the ground truth, not a nice-to-have.

## `question_type` definitions

- **`direct_lookup`** — the answer is one clause, no adversarial framing.
- **`coverage_with_exclusion`** — the correct answer requires retrieving both
  a coverage clause and an exclusion that limits it; both ids belong in
  `reference_clause_ids`.
- **`cross_document`** — targets same-insurer document pairs where the
  correct clause sits in one document and a plausible near-duplicate sits in
  the other.
- **`unanswerable`** — the answer is absent from the corpus by construction
  (deductibles, insured amounts, premiums, policy periods, endorsements).
  `reference_clause_ids` is empty and `expected_verdict` is
  `insufficient_information`.
- **`definition`** — the answer is a defined term from a policy's glossary
  clause.

## Authoring rules

1. **Questions are phrased as a claims analyst would ask them**, not as a
   restatement of clause wording. A question that echoes the clause's own
   phrasing tests string matching, not retrieval.
2. **`reference_clause_ids` is exhaustive.** Missing a clause the question
   actually depends on makes the ground truth wrong, not just incomplete —
   a retriever that finds every listed clause but misses the missing one
   would score a false 100%.
3. **Author authority over LLM assistance.** An LLM may draft a question's
   phrasing, but only the author decides what counts as ground truth. If the
   same model that composes the system also generates and validates its own
   test, the resulting score is circular and has no evidentiary value in
   front of a reader asking "how did you validate this?"

### The three-layer authoring flow

Every golden question is produced through three steps, each with a single,
non-overlapping owner:

1. **Source-clause selection — author.** Before a question exists, the
   author picks the source clause (or coverage+exclusion pair) from the
   parsed M1 clause tree (`build/parsed_clauses.jsonl`) — **never from the
   raw PDF.** This choice already carries the clause's provenance metadata
   (`document_id`, `insurer`, `cnpj`, `bundle_section`), which is what ties
   the question to the right document among several similar insurers'
   filings — that tie is established at selection time, not discovered later
   by search.
2. **Phrasing — LLM.** Given the source clause, the LLM drafts the
   question's wording the way a claims analyst would ask it, without echoing
   the clause's literal text. This is where the LLM outperforms manual
   authoring and where most of the authoring speed gain comes from.
3. **Verification — author.** Two distinct checks, both the author's:
   - **Correctness**: does the referenced clause actually answer the
     question? A straightforward comparative read, not a legal judgment.
   - **Completeness**: is any clause missing from `reference_clause_ids`?
     This is where the project's largest risk concentrates, especially for
     adversarial question types.

A question is accepted only after both verification checks pass.

### `unanswerable` questions need textual proof, not LLM confirmation

Confirming that an `unanswerable` question's answer is genuinely absent from
the corpus cannot be delegated to an LLM asking "is this information in the
corpus?" — that asks the model to prove a negative across 30 documents. A
failure to retrieve a passage is not proof the passage doesn't exist. The
author confirms absence by direct textual search of the corpus.

### Structural validation

Schema conformance and `reference_clause_ids` existence are mechanical,
unambiguous checks — not part of the authoring judgment above — and are
enforced in code: `scripts/validate_golden_set.py` (`make
validate-golden-set`), run in CI so a corpus re-parse that breaks a
`clause_id` fails visibly.

## Independent second-reviewer pass (golden-set-v1)

Every golden question above was authored, labelled, and verified by the
same person who builds the system being evaluated — the self-bias problem
this section exists to disclose. [M2-01] originally planned to mitigate it
with a same-author blind re-labelling pass after a delay (`authored_at`'s
purpose, above). This section amends that plan: a delay is a proxy for
forgetting, and only guards against a lapse in recall — it says nothing
about whether the authoring rules themselves are clear enough for someone
else to apply. What follows replaces it with an **independent second
reviewer**, on a sample, checking exactly that.

**Sampling frame.** The sample is drawn only from golden-set-v1's five
`question_type` files under `data/golden_set/` — sourced from [M2-02]'s
`direct_lookup` questions and [M2-03]'s `coverage_with_exclusion`,
`cross_document`, and `definition` questions, plus [M2-05]'s `unanswerable`
questions. [M2-04]'s synthetic claims and [M2-05]'s product/claim-mismatch
claims are explicitly out of scope for this pass: neither carries a
`reference_clause_ids`/`expected_verdict` pair of the kind this check
tests against.

**Sample size and stratification.** Every stratum is sampled at ~20%
(rounded to the nearest question), except `unanswerable`, which is sampled
at a higher, independent rate (~50%). This is not a second mechanism: the
distribution doc's "additional targeted review" of the unanswerable subset
and the general 20% pass are the same review task, just applied to a
larger share of that one stratum, because a wrongly-labelled `unanswerable`
question is the one error that silently miscalibrates the
insufficient-context gate downstream ([M3-07]). The `cross_document`
stratum's draw is topped up, if needed, to guarantee at least one
HDI-brand-collision question (the only `cross_document` rows targeting a
HDI-branded document among this set) and at least one `bundle_section`
question (a reference clause whose parsed record carries a non-null
`bundle_section`), so neither trap case is missed by chance. Selection is
deterministic and seeded (`scripts/review_sample_selection.py`).

**Review packet.** For each sampled question, the reviewer receives exactly
two things: the question text, and the target document's full clause list
(clause id, title, and text, in document order) from
`build/parsed_clauses.jsonl`. `reference_clause_ids`, `notes`, and
`expected_verdict` are withheld — a reviewer who can see the author's
answer is doing review, not independent labelling.

**Adjudication rule — fixed here, before any review result is seen.** On
any disagreement between the author's original label and the reviewer's
independent label, the author's original label is retained in
golden-set-v1 unchanged. No question is dropped from the set on account of
a disagreement, and no further justification is required to keep the
author's label. The disagreement itself is still recorded in full — see
Results, below — not silently discarded.

**Results.** Sample size: 36 questions (29 general stratified + 7
`unanswerable` top-up), across all five `question_type` strata.
`reference_clause_ids` exact-match rate: 75.0% (mean Jaccard 0.84, mean F1
0.87). `expected_verdict` agreement (the 16 questions where one applies —
`coverage_with_exclusion` and `unanswerable`): 100%. Combining both
dimensions, the full agreement rate is 75.0% (27/36) — at this sample size,
a single disagreement moves that rate by ±2.8 percentage points; the
smaller per-`question_type` strata swing considerably more (e.g.
`coverage_with_exclusion`, n=4, swings ±25pp per disagreement). See
`eval/runs/golden_set_review_v1.md` for the full per-`question_type`
breakdown and `data/golden_set/review/review_v1.jsonl` /
`eval/runs/golden_set_review_v1.md`'s disagreement table for every
individual disagreement (question id, the author's label, the reviewer's
label, and the resolution each one received under the adjudication rule
above — the author's original label, unchanged, in every case).

This number should not be read as "the golden set is 75% correct." Every
disagreement inspected here was the reviewer omitting or adding one clause
adjacent to an already-largely-correct answer (or, in one `unanswerable`
case, citing a clause it simultaneously judged did not answer the
question) — not the reviewer finding a wrong `expected_verdict` or an
actually-answerable question mislabelled `unanswerable`. What this pass
does and does not establish — including how much weight a non-domain-expert
reviewer's disagreement should carry on `coverage_with_exclusion`'s
completeness judgments specifically — is covered in full when [M2-07]
freezes `golden-set-v1`.
