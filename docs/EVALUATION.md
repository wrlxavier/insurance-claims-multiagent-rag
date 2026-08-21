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
