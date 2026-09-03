# End-to-end verdict accuracy and citation coverage — [M4-10]

Every other M4 document measures one node. This one runs the **whole compiled
graph** over a claim narrative and asks whether the answer at the far end is
right: intake → the clarification loop → retrieval → the parallel
compatibility/consistency branches → the recommendation node → the [M4-09]
human checkpoint, then the verdict, scored against the claim's label.

It is the number M4's exit criteria name, and the only one that can catch a
failure living *between* two nodes rather than inside one. It found exactly
such a failure — see **Findings**.

Code:

- `scripts/eval_end_to_end.py` — the runner (`make eval-end-to-end`), plus
  `build_claim_text` (the policy header), `failure_cause` (the catalogue's
  attribution rule) and `snapshot_payload` (the committed citation artefact).
- `scripts/validate_citation_coverage.py` — the CI gate
  (`make validate-citation-coverage`), replaying
  `eval/end_to_end_citations.json` offline.
- `app/src/infrastructure/evaluation/verdict_metrics.py` — the confusion
  matrix and per-class rates, shared with [M4-05]'s eval so the two published
  matrices cannot drift apart in definition.
- `app/src/infrastructure/evaluation/judge.py` — the faithfulness /
  context-relevance judge, **including the committed prompts**.
- `app/src/infrastructure/graph/reasoning_format.py` — `render_reasoning` /
  `parse_reasoning`, the round-trip that lets a citation check read the
  assertions back out of graph state.

**Why measure it separately.** Every upstream document measures one node
against a fixed, favourable input: [M4-05] and [M4-08] feed the compatibility
and recommendation nodes a golden question with its document already resolved
from the manifest. That is the right way to test a node and the wrong way to
test a system. Here the only input is what a policyholder wrote.

**Scope.** What this issue does **not** own: the ≥75% reference value's
*disposition* — whether M4 closes on the number measured — is
`MILESTONES.md`'s; the API that would expose a recommendation is M5's; and the
README results table is [M6]'s.

---

## Method

- **The claim set**: all **51** synthetic claims —
  `data/synthetic_claims/claims.jsonl` (40: 14 `compatible`, 13
  `incompatible`, 13 `insufficient_information`) plus
  `data/synthetic_claims/product_claim_mismatch.jsonl` (11, all
  `incompatible`). The DoD says "30 synthetic claims"; the finalised set grew
  past that during M2, the same drift [M4-02] records in
  `docs/INTAKE_EXTRACTION.md`. All 51 are scored, in four cohorts — the three
  verdict classes, and `mismatch` as its own, which is the DoD's "report the
  product/claim mismatch subset separately".
- **The graph, compiled and invoked per claim**, one `thread_id` each, on an
  `InMemorySaver` carrying the project's own `build_checkpoint_serializer()`
  allowlist. Nothing here needs to survive the process and an eval should not
  leave rows in the service's checkpoint tables — but the serializer must be
  the real one, or LangGraph silently rebuilds every state model as a plain
  `dict` (`docs/HUMAN_CHECKPOINT.md`, finding 4) and the scoring below would
  read a verdict off a mapping.
- **The checkpoint is resumed, not skipped.** `human_review` interrupts
  unconditionally; the runner resumes with a canned `approve`. That approval is
  a mechanical resume, **not a human judgment**: every number in this document
  scores the *system's* recommendation, and no claim here was reviewed by a
  person.
- **The verdict** is read from the recommendation node's posture, which the
  node records in its audit event — `Recommendation` carries no verdict field
  by [M4-08]'s design. `compatible`/`incompatible` map to themselves;
  `inconclusive`, `claimant_gaps`, `retrieval_miss` and `no_assessment` all map
  to `insufficient_information`, because each of them means "this system does
  not know".
- **The claim carries its policy.** See the next section — it is the one
  harness decision that changes what is measurable.
- **The judge** (`--judge`): faithfulness per assertion, context relevance per
  retrieved clause, three passes each, majority-voted, with the unanimity rate
  and the per-pass spread published beside the mean.

### The policy header, and why it exists

A real claim is filed *against* a policy. The graph has no channel for that —
`ClaimState`'s input is `{claim_id, raw_claim_text}` — and the synthetic
narratives deliberately never state a SUSEP process. So by default the runner
prepends one line to the narrative:

```
[Apólice registrada: processo SUSEP 15414.610650/2024-59]
```

Intake already extracts `susep_process` "if stated", so this uses the existing
path rather than adding one. It tells the graph nothing about the *event* —
only which product the claim was filed against.

Without it the product/claim-mismatch cohort is not merely hard, it is
**unanswerable**: those 11 claims describe a CASCO-type event filed against an
RCF-A / ASSIST / GAR.EST / CARTA VERDE policy, and a system that never learns
which product was bought cannot call the mismatch. Reporting 0/11 there would
be reporting a foregone conclusion, not a measurement.

`--no-policy-header` runs the other arm and measures what the header is worth.
That arm also answers the question `docs/RETRIEVAL_NODE.md` left open: on
`golden-set-v1` every question targets one document and the process/CNPJ pair
is 1:1 within it, so the pre-filter's value is invisible there. "The claims
path where a process is not stated is [M4-10]'s to measure."

### The failure catalogue

Every wrong verdict is attributed to exactly one cause, first match wins, in
causal order — an exhausted clarification loop never reaches retrieval, so it
must be tested before any retrieval signal; a retrieval miss starves the
assessment, so it must be tested before any assessment signal.

| cause | signal in state |
| --- | --- |
| `claimant_gaps` | `clarification_exhausted` — the claimant never supplied a load-bearing fact |
| `retrieval_miss` | `context_sufficient is False`, or no `reference_clause_ids` among the returned citations |
| `parsing_error` | the compatibility node's `ungrounded_after` marker: three grounding attempts failed and it degraded |
| `reasoning_error` | a reference clause **was** retrieved and the verdict is still wrong |

The DoD names three causes. `claimant_gaps` is a fourth, because
`docs/ARCHITECTURE.md` already reserves `clarification_exhausted` as "a
distinct failure mode [M4-10] catalogues" — a claim the graph could not assess
because the claimant omitted a fact is not a retrieval, parsing or reasoning
failure, and folding it into any of the three would misattribute it.

One thing deliberately **not** in the catalogue: the recommendation node's
justification falling back to its deterministic template. It is a real
degradation and is reported separately, but it cannot cause a wrong verdict —
[M4-08] computes every load-bearing field in Python and lets the model write
only the prose.

### Citation coverage, and why it is a script rather than only a test

The DoD wants "100% of assertions carry a clause id, and every id exists in
the corpus. CI fails otherwise." CI runs `pytest -m unit` **before** it fetches
the corpus artifacts, and installs neither the `embed` group nor an LLM key, so
the gate cannot be a live run and the corpus half cannot be a unit test. It is
therefore built the way [M3-07] built its gate: `make eval-end-to-end` writes a
**committed** snapshot, `eval/end_to_end_citations.json`, and
`make validate-citation-coverage` replays it offline as its own CI step, right
after `make validate-golden-set`. Three properties:

1. every assertion on a **settled** verdict carries ≥1 clause id — abstentions
   are excluded, because `_abstain` writes citation-free prose *by design* and
   failing them would flag the node for behaving correctly;
2. every cited id exists in `build/parsed_clauses.jsonl`, so a re-parse that
   renames a `clause_id` breaks visibly;
3. every recommendation citation exists in the corpus **and** was retrieved for
   that claim — [M4-08]'s structural guarantee, re-checked on a real run.

Missing input is an error, never a skip. A gate that passes quietly when its
evidence is absent is not a gate.

### Limitations

- **The judge shares no model family with the system under test, and that is
  the strongest claim available — not independence.** It is still an LLM
  grading LLM output, pinned to the same Gemini model `docs/EVALUATION.md`
  already uses as the golden set's second reviewer, for the same reason.
- **51 claims is a small population.** A single claim moves overall accuracy
  by ~2 percentage points; the `mismatch` cohort (n=11) moves by ~9 points per
  claim, and the three-class cohorts (n≈13) by ~8. Per-cohort figures should
  travel with their `n`.
- **The claims are synthetic and same-authored.** `docs/EVALUATION.md`'s "What
  this evaluation cannot establish" applies here unchanged, and the synthetic
  claims never even got the golden set's independent second-reviewer pass —
  they were explicitly out of that pass's scope.
- **`reference_clause_ids` on a claim is "at least these", not exhaustive.**
  The exhaustiveness rule is a *golden-question* authoring rule. Reference
  recall is therefore reported as a diagnostic for the failure catalogue, never
  as a retrieval score comparable to M3's.
- **The run is not deterministic.** Two smoke runs over the same three claims
  disagreed on one of them, because intake's product-line classification
  varies. Every number here is one sample.

---

## Prediction, pre-registered

Written into the approved implementation plan **before any run**, including
before the three-claim smoke run used to shake out harness bugs, and reproduced
here unedited. Deviations are reported in **Findings**, not absorbed.

1. **Baseline (policy header, filter as-is):** the mismatch cohort scores
   ≤ 1/11, essentially all landing on `insufficient_information` via an empty
   filter → `context_sufficient is False`.
2. **After the `_build_filter` fix:** the mismatch cohort recovers to ≥ 7/11.
3. **`insufficient_information` cohort (13):** the strongest class, and mostly
   reached via `clarification_exhausted`, not via the retrieval gate.
4. **Overall three-class accuracy over the 51 lands in 60–80%** — i.e. the
   `MILESTONES.md` reference value of ≥75% is genuinely at risk, and a miss is
   a publishable outcome, not a failure of the run.
5. **No-header arm:** mismatch ≤ 2/11, and overall accuracy at least 10 points
   below the header arm.
6. **Judge:** faithfulness ≥ 0.85 (grounding is structurally enforced
   upstream); context relevance 0.5–0.7 (retrieval always returns k=10, whether
   or not ten clauses are relevant).

---

## Results

Three arms, all 51 claims, one run each. **Graph completion was 100% in every
arm — 51/51 claims reached the human checkpoint and produced a recommendation,
with no unhandled exceptions.** That is M4's first exit criterion, and it is the
one number here that is not close.

| # | arm | filter | overall accuracy | mismatch cohort |
| --- | --- | --- | ---: | ---: |
| 1 | policy header | conjunction (as it shipped) | **35.3%** | **0/11** |
| 2 | policy header | process wins (fixed) | **56.9%** | **5/11** |
| 3 | no header | process wins (fixed) | _not completed_ | _not completed_ |

### Arm 1 — the baseline, and the defect it found

Run of 2026-09-02, 51/51 scored, 0 errors, reasoning model
`deepseek/deepseek-v4-pro-0813`, fast model `deepseek/deepseek-v4-flash-0731`.

| expected \ predicted | compatible | incompatible | insufficient_information |
| --- | ---: | ---: | ---: |
| compatible | 4 | 1 | 9 |
| incompatible | 1 | 2 | 21 |
| insufficient_information | 1 | 0 | 12 |

| verdict | support | precision | recall |
| --- | ---: | ---: | ---: |
| compatible | 14 | 66.7% | 28.6% |
| incompatible | 24 | 66.7% | 8.3% |
| insufficient_information | 13 | 28.6% | 92.3% |

(`incompatible` support is 24, not 13: the mismatch cohort's 11 claims carry
that label too.)

| cohort | n | accuracy | zero clauses retrieved | mean reference recall |
| --- | ---: | ---: | ---: | ---: |
| compatible | 14 | 28.6% | 5 | 0.29 |
| incompatible | 13 | 15.4% | 5 | 0.23 |
| insufficient_information | 13 | 92.3% | 9 | 0.08 |
| **mismatch** | 11 | **0.0%** | **11** | 0.00 |

| failure cause | n |
| --- | ---: |
| retrieval_miss | 26 |
| claimant_gaps | 3 |
| reasoning_error | 3 |
| parsing_error | 1 |

Retrieval: reference-clause recall 16.4% (micro), 19.6% of claims retrieved at
least one labelled clause, 41.2% retrieved anything at all from their own
policy document. Compatibility degraded after three failed grounding attempts
on 6 claims; the recommendation node never fell back to its template.

**The distribution is the finding, exactly as the DoD says it would be.** 26 of
the 33 wrong verdicts are retrieval misses, and they are not near-misses:
**30 of 51 claims retrieved zero clauses**, and **not one claim retrieved from
the wrong document**. The filter was either exactly right or completely empty —
never merely imprecise. A ranking problem does not look like that; an empty
selection does.

### Arm 2 — the same measurement after the fix

Identical in every respect except `_build_filter`. 51/51 scored, 0 errors.

| expected \ predicted | compatible | incompatible | insufficient_information |
| --- | ---: | ---: | ---: |
| compatible | 6 | 1 | 7 |
| incompatible | 3 | 12 | 9 |
| insufficient_information | 2 | 0 | 11 |

| verdict | support | precision | recall |
| --- | ---: | ---: | ---: |
| compatible | 14 | 54.5% | 42.9% |
| incompatible | 24 | 92.3% | 50.0% |
| insufficient_information | 13 | 40.7% | 84.6% |

| cohort | n | arm 1 | **arm 2** | zero clauses retrieved | mean reference recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| compatible | 14 | 28.6% | **42.9%** | 3 | 0.57 |
| incompatible | 13 | 15.4% | **53.8%** | 1 | 0.38 |
| insufficient_information | 13 | 92.3% | **84.6%** | 6 | 0.08 |
| **mismatch** | 11 | 0.0% | **45.5%** | 1 | 0.09 |

| failure cause | arm 1 | **arm 2** |
| --- | ---: | ---: |
| retrieval_miss | 26 | **11** |
| claimant_gaps | 3 | **5** |
| reasoning_error | 3 | **5** |
| parsing_error | 1 | **1** |

Retrieval: reference-clause recall 29.5% (from 16.4%), 33.3% of claims
retrieved at least one labelled clause (from 19.6%), and **78.4% retrieved
from their own policy document (from 41.2%)**.

**The cleanest statement of what the fix did**: after it, **every one of the 40
claims that reached retrieval got clauses back**, and the 11 that came back
empty are *exactly* the 11 whose clarification loop exhausted before retrieval
ever ran — the two sets are identical. In arm 1, 20 claims reached retrieval
and got nothing.

### The judge

Model `google/gemini-3.7-flash` — a different family from the DeepSeek models
that produced both the assertions and the justification — three passes per
item, majority-voted, over arm 2.

| metric | value | items | unanimous across passes | per-pass spread |
| --- | ---: | ---: | ---: | ---: |
| faithfulness (supported) | **93.8%** | 128 assertions | 95.3% | 0.000 |
| context relevance | **59.3%** | 398 clauses | 84.2% | 0.068 |

Partially supported 3.9%, unsupported 2.3%. Faithfulness was identical on all
three passes (0.938, 0.938, 0.938); context relevance moved between 0.533 and
0.601.

### Citation coverage

`make validate-citation-coverage`, over the committed
`eval/end_to_end_citations.json` from arm 2:

- **91/91 assertions on a settled verdict carry at least one clause id — 100%.**
- Every one of those ids exists in `build/parsed_clauses.jsonl` (4,925 ids).
- All **123** recommendation citations exist in the corpus *and* were returned
  by retrieval for that same claim.
- 24 settled verdicts across 51 claims; abstentions are excluded by design.

The same held in arm 1 (50/50 assertions), so this is not a property the fix
bought — it is structural, and both runs are evidence of that.

### Arm 3 — attempted, not completed

The no-policy-header arm was attempted three times and abandoned each time: the
fast model (`deepseek/deepseek-v4-flash-0731`, provider-pinned to `baidu/fp8`
with no fallback) returned HTTP 429 under sustained load, and the shared retry
policy (`DEFAULT_LLM_RETRY_MAX_ATTEMPTS = 3`, 5 s apart) is a ~15-second window
— far too short for a rate limit measured in tens of minutes. The first attempt
errored 20 of its 21 processed claims, the second errored on its first, and the
third — after a ten-minute pause during which a single probe call succeeded —
errored 7 of 9. A one-off probe does not clear the limit; sustained load
re-trips it immediately.

**No number from those attempts is reported here**, and none should be: a run
whose claims failed on infrastructure rather than on the graph measures the
provider, not the system. What the attempts do establish is a harness
limitation worth recording — the retry policy is tuned for a transient blip,
not for a sustained quota, and a long eval against a pinned single provider with
`allow_fallbacks=False` has no way to ride one out.

Rerunning it needs nothing but a quieter provider:

```bash
LLM_PROVIDER=openai LLM_REASONING_PROVIDER_ORDER='["alibaba"]' \
  PYTHONPATH=app/src uv run --group embed python -m scripts.eval_end_to_end \
  --no-policy-header --no-judge --out-stem end_to_end_no_policy
```

Until it runs, **prediction 5 is untested** and `docs/RETRIEVAL_NODE.md`'s
question — what the pre-filter is worth when a claim states no process — stays
open. Note that the two completed arms do not depend on it: every [M4-10] DoD
item is answered by arms 1 and 2.

---

## Findings

- **The graph completes on every claim. Its accuracy is a separate question,
  and the answer is 56.9%.** Both completed arms ran 51/51 with zero unhandled
  exceptions, every claim reaching the human checkpoint and producing a
  recommendation. M4's first exit criterion is met without qualification. The
  verdict accuracy is **below** the ≥75% reference value, and this document is
  the failure catalogue `MILESTONES.md` requires in that case.

- **The most valuable thing this measurement produced is a defect, not a
  number.** The retrieval pre-filter ANDed a stated SUSEP process with intake's
  narrative-derived product line. Those two fields answer different questions —
  which product was *bought* versus what the claimant *described* — and on a
  product/claim mismatch they disagree by construction. The conjunction then
  selects nothing, and a knowable `incompatible` becomes
  `insufficient_information`. Fixing it moved overall accuracy **35.3% → 56.9%**
  and the mismatch cohort **0/11 → 5/11**. Per `MILESTONES.md`'s rule, the fix
  and its re-measurement happened in this pass, and both states are published.

- **What identified the defect was the absence of a middle.** In the baseline,
  30 of 51 claims retrieved *zero* clauses and **not one** retrieved from the
  wrong document. Ranking problems do not look like that — they produce
  plausible-but-wrong results, not empty ones. A database count with no model in
  the loop then confirmed the mechanism outright: under the conjunction, all 11
  mismatch claims have an empty search space, and so would 9 of the 13 documents
  the main claim set targets whenever intake reads the event as CASCO — the
  reading a corpus that is 71% CASCO invites. After the fix, **every one of the
  40 claims that reached retrieval got clauses back**, and the 11 that did not
  are exactly the 11 whose clarification loop exhausted first.

- **Retrieval is still the largest failure cause, but it is now a ranking
  problem rather than an empty-set one.** 11 of the 22 remaining errors are
  retrieval misses. The system retrieves from the right document 78.4% of the
  time but surfaces a labelled reference clause for only 33.3% of claims
  (reference recall 29.5%). Against [M3-08]'s 92.3% Recall@10 on
  `golden-set-v1`, that gap is the honest cost of the real task: a golden
  question is written to name what it wants, and a claim narrative is a
  policyholder describing a bad afternoon. The same retriever, the same corpus,
  a different query distribution.

- **Citation coverage is structural, and both arms prove it.** 91/91 assertions
  on a settled verdict carried a clause id in the headline arm, 50/50 in the
  baseline, and every id in both existed in the corpus. All 123 recommendation
  citations were clauses retrieval had actually returned. This is not a number
  the fix improved — it held even when retrieval was returning nothing, because
  [M4-05] rejects an ungrounded assertion rather than post-editing one, and
  [M4-08] can only copy what it was given.

- **The judge says the assertions are faithful and the context is noisy.**
  Faithfulness **93.8%** (95.3% of assertions judged unanimously across three
  passes; the aggregate was *identical* on all three) against context relevance
  **59.3%** (spread 0.068). Read together: when the system asserts something, the
  clause it cites nearly always supports it — but roughly two of every five
  clauses handed to the assessor had no bearing on the claim. That is the same
  finding as the retrieval bullet above, measured from the other end, and it is
  the more actionable of the two numbers.

- **The system's characteristic error is abstention, and that is the direction
  it should err in.** It answered `insufficient_information` on 27 of 51 claims
  where 13 carry that label: `incompatible` precision is 92.3% but recall only
  50.0%, while `insufficient_information` recall is 84.6% at 40.7% precision.
  When it commits to a verdict it is usually right; it just declines often. For
  a system whose output a human reviews before anything is recorded, a false
  "I don't know" costs a review and a false "incompatible" costs a wrong denial
  — and the [M4-05] grounding contract and the [M3-07] gate are both designed to
  buy exactly this trade. It is worth stating plainly that the 56.9% is not a
  system that is wrong 43% of the time; it is a system that abstains too much.

- **Two of the six pre-registered predictions missed, and neither has been
  edited.** Prediction 2 said the mismatch cohort would recover to **≥7/11**; it
  reached **5/11**. Prediction 4 said overall accuracy would land in **60–80%**;
  it landed at **56.9%**, just below the band. Both misses point the same way —
  the fix helped substantially but less than predicted, because it restored the
  *search space* without making the *ranking* inside it any better, and the
  reference-recall figures show that is where the remaining loss sits.
  Prediction 5, on the no-header arm, is **untested** — that arm did not
  complete (see Results). Predictions 1, 3 and 6 held: the baseline mismatch cohort scored 0/11 (≤1),
  `insufficient_information` was the strongest class in both arms (92.3% then
  84.6%) and was mostly reached via `claimant_gaps` (7 of its 12 correct answers
  in the baseline), and the judge came in at 93.8% faithfulness (≥0.85) with
  context relevance inside the predicted 0.5–0.7.

---

## What this means downstream

- **M5-02 / M5-04 should give the policy a real input channel.** This issue put
  the registered policy in front of the graph by writing it into the claim text,
  because `ClaimState`'s input is `{claim_id, raw_claim_text}` and nothing else.
  That is defensible for a measurement harness and wrong for a service: a real
  submission knows its policy as a *field*, and should not depend on a claimant
  remembering to type a SUSEP process into free prose. `SubmitClaim` should
  carry a policy identifier, and the retrieval filter should read it from there
  rather than from whatever intake managed to extract. Quantifying what that is
  worth is exactly what arm 3 was for, and it has not run — so the size of the
  cost is still unmeasured, while its existence is not in doubt: without the
  policy the mismatch cohort is unanswerable by construction.
  *Update ([M5-02]):* `SubmitClaim` now takes `policy_ref: SusepProcess | None`
  and threads it onto the domain `Claim`, so the port carries the policy as a
  field. `ClaimState` still has no policy input, so *how* the [M5-04]
  orchestrator adapter feeds it to retrieval (a header line like this harness,
  or a new `ClaimState` field) is [M5-04]'s call — the port does not constrain
  it. `Claim.policy_ref` stays optional until [M5-04] makes it required.
- **[M6]'s README results table** takes its end-to-end accuracy, and the
  per-cohort figures, from this document — with the `n` beside each, per the
  Limitations above.
- **The [M3-07] gate was calibrated on questions and now runs on claims.** Its
  published 100% / 100% on `golden-set-v1` was measured over 23 unanswerable
  *questions*; a claim narrative is a different input distribution, and this run
  is the first evidence of how the gate behaves on one. Recalibrating it against
  claims is a candidate follow-up, not something this issue changed — moving a
  threshold that a published number depends on is a decision for the project
  owner, under the pre-registration protocol `MILESTONES.md` sets out.
- **[M4-05]'s and [M4-08]'s node measurements stay the right place to catch
  regressions inside a node.** This document deliberately cannot: an end-to-end
  number moves for many reasons at once, which is exactly why the per-node evals
  exist and why this one reports a failure catalogue rather than a single
  figure.
