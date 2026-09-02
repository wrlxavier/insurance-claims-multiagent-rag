# The consistency node — [M4-06]

The graph node that reads a claim for *internal* consistency — where the account
does not hold together on its own terms — and flags what it finds for a human
reviewer. It runs two legs and merges their signals into one
`state.ConsistencyReport`:

- **Deterministic** — plain Python over the entities intake extracted, run
  unconditionally, no model. Four checks: an `event_date` typed in the future or
  many years past; an `estimated_amount` that is negative or absurd; a field
  intake populated while also tagging it missing; a `product_line` whose
  registered definition is contradicted by a token in the event text.
- **Semantic** — one call on the *fast* model, structured into
  `schemas.ConsistencyOutput`, for the judgement the first leg cannot make:
  narrative coherence, description vs. stated event type, vagueness where a
  claimant would give detail.

Every `ConsistencySignal` carries `source` (`"deterministic"` / `"llm"`) so the
two stay measurable apart. **The node returns signals, never a verdict** — it
decides nothing.

**This is not a fraud detector, and the project does not claim it is.** The
data carries no fraud labels and the method — a handful of range checks plus an
LLM reading for coherence — is not one. See `docs/SCOPE.md`.

Code:

- `app/src/infrastructure/graph/consistency_checks.py` — the deterministic leg:
  `check_date_coherence`, `check_amount_plausibility`,
  `check_internal_contradictions`, `check_product_line_event_type_mismatch`,
  and `run_deterministic_checks`. Pure functions; no langgraph, no LLM. `now` is
  injected so every check is testable in isolation.
- `app/src/infrastructure/graph/nodes/consistency.py` — the node: deterministic
  leg, then the fast-model call (best-effort — a failed call degrades to
  deterministic-only, it never raises), merged into `state.consistency`. Two
  `AuditEvent` rows, one per leg.
- `app/src/infrastructure/graph/schemas.py` — `ConsistencyOutput` /
  `ConsistencySignalItem`, the fast-model output shape. `check` is a `Literal`
  of the three semantic categories. No verdict field.
- `app/src/infrastructure/graph/prompts/consistency.py` —
  `build_consistency_prompt`, wrapped in `with_scope_preamble`.

**Why measure it separately.** [M4-10] measures end-to-end verdict accuracy.
This document measures the consistency node in isolation: do the deterministic
guard rails fire where they should and stay silent where they should, and what
does the semantic leg actually flag on the synthetic set? A regression here (a
guard rail misfiring on every coherent claim, the prompt drifting into verdict
territory) would otherwise show up only as diffuse noise downstream.

**Scope.** What this issue does **not** own:

- The graph edges. [M4-07] wired the consistency and compatibility nodes as
  fixed parallel branches from the retrieval node, with a fan-in, and measured
  the wall-clock gain (`docs/PARALLEL_ASSESSMENT.md`). This issue left `build.py`
  untouched — the node had a standalone compiled-graph unit test only.
- The recommendation node's consumption of `consistency_flags` — [M4-08].
- End-to-end accuracy over the synthetic claims — [M4-10].

---

## The deterministic / LLM boundary

The rule: **string / number / set equality is Python; "do these two prose fields
disagree in meaning" is the model's.**

| Concern | Leg | What it checks |
| --- | --- | --- |
| Date coherence | deterministic | `event_date` parses as an absolute date and is after today → `attention`; or more than seven years past → `info`. A relative phrase or an unparseable string → nothing. |
| Amount plausibility | deterministic | `estimated_amount` ≤ 0 → `attention`; < R$100 → `info`; > R$2,000,000 → `attention`. Bands are arbitrary sanity rails, not fraud thresholds. |
| Internal field contradictions | deterministic | `estimated_amount` set while `valor_franquia_limite` is still in `missing_information` (or `event_date` / `data_evento_vigencia`) → `info`. Intake disagreeing with its own extraction. |
| Product-line / event-type mismatch | deterministic | A closed contradiction table for the two lines with an *absolute* definitional constraint: GAR.EST + a (non-negated) external cause; ASSIST + vehicle-value indemnification language. Positive, un-negated collision only; never proposes a line; silent when `product_line` is `None`. CASCO / RCF-A are deliberately absent — see Limitations. |
| Narrative coherence | LLM | Does the described sequence of events hold together; does a later statement contradict an earlier one. |
| Description vs. event type | LLM | Does the free description match the labelled `event_type` (a paraphrase / synonymy judgement). |
| Unexpected vagueness | LLM | A material loss described with none of the detail a claimant would normally give. |

**Not attempted, by design.** Whether the event date falls inside the policy
period — the motivating example for this issue — is *not* checkable here. The
corpus is registered product conditions, not contracts: it carries no vigência,
insured amount or deductible (`docs/SCOPE.md`), so there is no period to compare
a date against. Intake's `data_evento_vigencia` missing-information tag records
that gap instead.

---

## Method

- **Deterministic leg** — proven by
  `tests/unit/infrastructure/graph/test_consistency_checks.py` (every check in
  isolation, no LLM), which runs in CI (`pytest -m unit`).
- **Whole node** — `make eval-consistency` (`scripts/eval_consistency.py`) runs
  the real fast model over all 51 synthetic claims (40 `claims.jsonl` + 11
  `product_claim_mismatch.jsonl`): intake, then the consistency node. LLM-free
  dependencies only — no Postgres, no `embed` group, no retrieval stack. Needs
  `LLM_*` in `.env`. Writes `eval/runs/consistency.{md,json}` +
  `consistency_signals.jsonl`.

Reported, by cohort (14 `compatible` / 13 `incompatible` /
13 `insufficient_information` / 11 `mismatch`):

- deterministic `attention` false-positive rate on the `compatible` cohort — the
  headline deterministic metric;
- deterministic signal counts by check;
- LLM signal counts by category;
- zero-signal ("clean pass") rate and the LLM-failure count.

### Limitations

- **No per-signal ground truth.** A "consistency signal" is a judgement call;
  the eval reports distributions, not accuracy.
- **Date and amount checks fire near-zero on this corpus.** The synthetic
  narratives carry no absolute dates (every `event_date` is a relative phrase)
  and state an amount about 4% of the time (`docs/INTAKE_EXTRACTION.md`). This
  is a property of the corpus, not a bug — the checks are production guard rails
  for an intake form that *does* carry a structured date and amount, and they
  are unit-tested in isolation.
- **The product-line check is an intake-misclassification cross-check, not a
  claim-vs-document check.** `product_claim_mismatch.jsonl` describes damage to
  the insured's own vehicle aimed at a non-CASCO document — a mismatch between
  the claim and its target document. This node never sees the target document,
  and intake classifies those events as `CASCO` correctly, so the deterministic
  product-line check is *expected* to stay silent on that cohort. It fires only
  where intake's own classification disagrees with the event text intake itself
  extracted.
- **Only GAR.EST and ASSIST are checked, with a negation guard.** The first
  eval run flagged three coherent `compatible` claims: two GAR.EST narratives
  that say "sem colisão" / "nada a ver com batida" (the claimant establishing a
  self-caused failure — a negation the check now scans for) and one CASCO claim
  whose terse extracted description mentioned "o outro veículo" without a
  possessive for the insured's own car. CASCO vs. RCF-A turns on *who* was
  damaged, which a condensed description renders unreliably, so those two lines
  were removed from the table — only the two lines with an absolute
  definitional constraint remain.
- **Not a fraud detector.** Stated again because it matters: neither the data
  nor the method supports that claim.

---

## Results

Two runs, both `deepseek/deepseek-v4-flash-0731` on `baidu/fp8`, all 51 claims,
0 errors. Regenerate with `make eval-consistency` (output in
`eval/runs/consistency.{md,json}` + `consistency_signals.jsonl`, git-ignored).

**Run 1** (before the fix below) flagged **3 of 14 `compatible` claims** with a
deterministic `attention` — a 21.4% false-positive rate, all
`product_line_contradicts_event`: two GAR.EST narratives whose extracted
description said "sem colisão" / "nada a ver com batida" (the claimant
establishing a self-caused failure), and one CASCO claim whose terse description
said "o outro veículo" with no possessive for the insured's own car. Fix: a
negation guard on the peril match, and CASCO / RCF-A removed from the
contradiction table (they turn on *who* was damaged, unreliable from a condensed
description). See the Limitations note above.

**Run 2** (after the fix):

### Deterministic false-positive discipline

- `attention` false-positive rate on the `compatible` cohort: **0.0%** (0/14).

### Signals by cohort

| cohort | n | claims with a deterministic `attention` | zero-signal claims |
| --- | ---: | ---: | ---: |
| compatible | 14 | 0 | 13 |
| incompatible | 13 | 0 | 13 |
| insufficient_information | 13 | 0 | 13 |
| mismatch | 11 | 0 | 11 |

Deterministic signals fired **0 times** across all 51 claims: no absolute dates
and almost no stated amounts in the corpus (a property of the synthetic set, not
a bug — the date and amount checks are production guard rails, unit-tested in
isolation), and every claim's `product_line` was classified consistently with
its own event text.

### LLM signal distribution

- Run 2: **1 signal total** — `description_event_type_mismatch` (`info`) on
  `compatible-001` (the extracted `event_type` said the stone cracked the
  mirror; the narrative describes only a scratch on the mirror housing).
- Run 1, for contrast: `description_event_type_mismatch` ×1 (`compatible`),
  `narrative_coherence` ×3 (`incompatible`). `unexpected_vagueness` fired in
  neither run.
- One LLM-leg failure (`mismatch-011`, run 2): the node degraded to the
  deterministic signals alone and recorded `llm_failed=True` in its
  `semantic_judgement` audit event — it did not raise, as designed.

---

## Findings

- **The deterministic guard rails do not misfire — after one correction.** The
  first run showed the product-line check is only safe on the two lines with an
  absolute definitional constraint, and only with a negation guard; the amount
  and date checks never fired (nothing to fire on in this corpus) and never
  false-fired.
- **The semantic leg is very quiet on this corpus, and that is the correct
  outcome.** The synthetic claims were built so their inconsistencies are
  claim-vs-clause (does the event fit the filed conditions), not claim-vs-itself
  (does the story hold together). A node that only signals internal
  inconsistency should, and does, stay silent on a set of internally coherent
  narratives. The one signal it raised (a description that overstates the
  damage relative to the narrative) is a legitimate call.
- **Run-to-run variance in the semantic leg is real** (3 signals vs. 1 across
  two identical runs) and expected from a single fast-model pass. The node's
  value is the deterministic leg plus a best-effort semantic scan, not a
  calibrated detector — consistent with `docs/` guidance that a target is a bet
  and the analysis is the deliverable.
- **This is not a fraud detector**, and this measurement is not evidence that it
  could be one. It is evidence that the two legs behave as specified: arithmetic
  runs unconditionally and cheaply, judgement runs once and degrades safely, and
  neither invents a problem.

---

## What this means downstream

- **[M4-07]** wires this node and the compatibility node as fixed parallel
  branches. They write disjoint state channels (`consistency` vs.
  `compatibility`); the `audit_trail` reducer handles the concurrent append.
- **[M4-08]** consolidates `state.consistency.signals` into the recommendation
  as `consistency_flags`, kept separate from the compatibility verdict —
  attention points, not part of the decision.
- **[M4-10]** catalogues end-to-end failures; a consistency signal is a hint for
  that triage, never a verdict input.
