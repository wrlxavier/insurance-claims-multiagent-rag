# Milestones

Numeric exit criteria for every milestone. The point of writing
these down is to close a phase on evidence, not on the feeling that it looks
good enough.

Every criterion below is one of three things: a number to measure, a file
that must exist, or a command that must exit zero.

A milestone may close with a **bad** number. That is a valid outcome, not a
failure of process, provided the number was actually measured and the cause
is written down next to it. An unmeasured "looks fine" is worse than a
measured 40%. Undocumented silence about a bad number is the only truly
unacceptable outcome.

## On continuous improvement vs. issue-splitting

Measurement issues in this project have a recurring failure mode of their
own: a measurement finds a gap, the gap is written up, a new issue is
scoped to fix it, and the fix lands only to be measured by yet another
issue. [M1-08] → [M1-04c]/[M1-05b] → [M1-08b] → [M1-04d] → [M1-08c] is four
hops for two real code changes.

From [M1-08c] onward the working rule is: **when a measurement diagnoses a
cause it can act on, fix it and re-measure in the same pass**, and record
the intermediate states rather than the final number alone. [M1-08c] did
this — it measured [M1-04d], disabled it on evidence, fixed the underlying
detector, and re-measured, reporting all four states. The intermediate
numbers are what justify the decisions, so they are published, not
discarded.

This does not license unbounded scope, and in particular it does not
license moving the goalposts. A change that alters what a milestone's
number *means* is a separate decision from work that improves the number,
and belongs to the project owner, not to the measurement. [M1-08c] hit
exactly that case: it found that most of its residual boundary gap was the
criterion asking a flat question about a hierarchical structure. It did
**not** redefine the criterion on its own — it published the evidence,
stopped, and put the choice to the owner, who decided to correct it.

When a criterion does get corrected, the protocol [M1-08c] used is the
standard: state the change, predict the outcome in writing *before*
re-measuring, leave that prediction unedited, and report every deviation
from it. Two of [M1-08c]'s predictions missed and both are published. A
criterion corrected after it failed is only credible if the correction was
falsifiable — otherwise it is indistinguishable from tuning until the bar
is cleared.

## Status

| Milestone | Name                       | Status      |
| --------- | -------------------------- | ----------- |
| M0        | Foundations                | done |
| M1        | Policy parsing              | done        |
| M2        | Ground truth                | done        |
| M3        | Retrieval                   | done        |
| M4        | Agent graph                 | todo        |
| M5        | Service and hardening       | todo        |
| M6        | Release and portfolio       | todo        |

---

## M0 — Foundations

Turn a data-only repository into a working project: Python skeleton with
Clean Architecture layers, quality toolchain, CI, and the corpus-integrity
script the documentation already promises. Also fixes the project's scope
statement in writing, in every place a reader or a future prompt might look.

**Exit criteria:**
- `make check` (lint, format, type-check, tests) passes locally and in CI.
- The text-layer audit covers 30/30 documents and its result is recorded in
  the manifest and in `docs/DATA_SOURCES.md`.
- `MILESTONES.md` exists with numeric exit criteria for every milestone.
- The scope statement appears in `README.md`, `NOTICE.md`, `data/README.md`
  and the prompt template directory.

## M1 — Policy parsing

Turn 30 PDFs and 1,709 pages into a clause tree with reliable type metadata.
Retrieval quality is bounded above by parsing quality, so parsing accuracy is
measured and published as a first-class project metric, with its failure
modes described rather than hidden.

**Exit criteria:**
- 30/30 documents produce a clause tree with no unhandled exceptions.
- The two documents without a usable text layer are recovered, with ≥95% of
  their pages yielding non-empty text.
- Stratified validation of 50 clauses: ≥90% correct clause boundaries,
  ≥85% correct clause type. **Met (98.0% / 92.0%)** under the boundary
  criterion as clarified by [M1-08c] — a clause record is judged against
  its own span, its numbered sub-clauses being separate records. Under the
  original, un-clarified criterion the same corpus measures 84.0% / 86.0%;
  see `docs/PARSING.md` for both, and for why the clarification was made
  and pre-registered.
- ≥95% of clauses carry complete provenance (document id, SUSEP process,
  insurer CNPJ, product line, page range).
- `docs/PARSING.md` published with method, per-document results and known
  failures.

**[M1-08b] re-measurement (2026-08-19).** After [M1-04c] (boundary/heading
fixes) and [M1-05b] (real LLM classifier) landed, the corpus was rebuilt
(`clause_segmentation_version=v6`) and a fresh 50-clause stratified sample
was drawn and validated (automated LLM judgment against source-PDF page
images, replacing [M1-08]'s manual review by explicit project-owner
decision — see `docs/PARSING.md`'s opening note). Result: **boundary
accuracy 82.0% (41/50) — below the ≥90% bar; type accuracy 78.0% (39/50)
— below the ≥85% bar.** Both improved sharply from [M1-08]'s baseline
(60.0%/40.0%), and provenance accuracy stayed at 100.0% (50/50), but
neither bar is met.

**Explicit call: a follow-up is scoped, not closed as known debt.** The
remaining gap is concentrated in two diagnosed failure-mode clusters (both
documented in `docs/PARSING.md`'s second-measurement failure modes):
adjacent numbered sub-clauses merging at depth 3-4, and clause content
truncating at a page boundary. Both are cases the deterministic
text/font-based heading detector structurally cannot see, since it never
looks at the rendered page. [M1-04d] scopes a targeted, last-resort
vision-LLM boundary-detection pass limited to clauses the deterministic
pass already flags as suspicious (not a wholesale replacement of the
segmentation pipeline); [M1-08c] re-measures after it lands and makes the
close/known-debt call this entry defers. M1 stays **in progress**, not
closed, until that follow-up resolves.

**[M1-08c] re-measurement (2026-08-20).** [M1-04d]'s vision-LLM
boundary-escalation pass was run for the first time (its code had merged
but never actually executed), measured, found net harmful, and disabled;
the failure it was meant to fix was then addressed in the deterministic
detector and re-measured in the same pass. Four states were scored — all
four are in `docs/PARSING.md`, because the intermediate ones are the
evidence for the decisions:

| state | boundary | type | criterion |
| --- | ---: | ---: | --- |
| [M1-08b] baseline | 82.0% | 78.0% | original |
| escalation as [M1-04d] shipped it | 76.0% | 84.0% | original |
| escalation + safety guards | 80.0% | 80.0% | original |
| v7 segmentation fix, escalation off | 84.0% | 86.0% | original |
| **same corpus, criterion corrected** | **98.0%** | **92.0%** | clarified |

The first four rows differ by what the parser did; the last differs only by
what the question meant. Under the original criterion the parser reached
84.0% / 86.0% — type passing, boundary not. Provenance stayed 100.0%
(50/50) throughout.

**Escalation is disabled on evidence, not on preference.** Because it never
restructures the tree, its samples matched [M1-08b]'s 50/50 on `clause_id`
— a genuine matched pair — and the per-clause diff showed **4 boundary
regressions and 0 fixes**. Root-causing found a structural flaw: the
correction step reassigns whole pages of lines between neighbours, but a
page routinely carries several clauses (one glossary page carries 13). Two
guards were added (page exclusivity; identity retention) and showed that
only **28 of its 136 corrections were sound**; even guarded, it still
produced 1 regression and 0 fixes. It stays opt-in and off, with the guards
retained so any future re-enablement starts safe.

**What actually moved the numbers** was a deterministic fix [M1-04d] could
never have delivered: the adjacent-sibling merges both [M1-08b] and this
issue named as the largest cluster are *intra-page*, while escalation works
at page granularity and never auto-applies a split. Doc 8 typesets `6.3`
bold and its own siblings `6.4`/`6.5` in the body font, so the bold gate
swallowed them. A font-independent numbering-continuity fallback
(`_detect_sibling_continuation`, `clause_segmentation_version` v6 → v7)
split 4 of the 5 sampled merges and grew the corpus 4469 → 4925 clauses
(+10.2%) with no document losing clauses. Two operational bugs found along
the way were also fixed in code: a stale clause-tree cache that had made
the entire escalation pass a no-op (0 of 252 proposed corrections applied),
and validation resumability keyed on sample position rather than on the
claim, which silently reused a prior measurement's judgments.

**The residual gap turned out to be mostly a measurement-definition
problem, not a parsing one.** Every remaining boundary failure was checked
against the tree: **6 of the 8 were clauses whose children hold the content
the validator reported as missing** (sample #22, `18:3` "Definições", was
marked truncated for not covering `3.4`–`3.10`, which exist as 55 child
clauses spanning pages 4–11). The criterion judged a hierarchical parser's
node against a flat whole-section span, penalizing the parser *more* as
segmentation got finer.

**Fourth state — criterion corrected, pre-registered (2026-08-20).** On the
project owner's explicit decision, the validator's boundary question was
changed to judge a record against its own span, sub-clauses being separate
records. Nothing else changed: no parsing code, no sampling, no corpus. To
keep this a test rather than a re-tune, the expected outcome was **written
to `docs/PARSING.md` before the run** — naming the 5 clauses that should
flip, the 2 that should not, and 1 uncertain — and left unedited
afterwards. Result: **boundary 98.0% (49/50) — PASS; type 92.0% (46/50) —
PASS; provenance 100.0%.**

Both deviations from the prediction are published rather than absorbed:
sample #9 flipped although it has no children (the two runs disagreed on
printed page label vs. PDF page index, so the conservative reading is
**96.0%, excluding it**), and type accuracy moved although only the
boundary question changed (the hierarchy context reframes what each record
is). Because of the second deviation, 86.0% and 92.0% are not a clean
before/after — the claim that holds under either framing is that **type
clears ≥85% regardless**.

**Explicit call: M1 closes.** Both bars are met under the corrected
criterion (98.0% / 92.0%), and the type bar is met under the original one
too. The boundary bar is met only under the corrected criterion — stated
plainly here rather than blurred: `84.0%` and `98.0%` describe the same
corpus under two different questions, and the earlier 60.0%/82.0% figures
belong to the original question. Recorded as known debt, not silently
carried: one genuine boundary defect remains (sample #46, `21.3.4`
absorbing `21.3.5`), four type misjudgments all trace to the same
unaddressed parent-context gap (the classifier never sees a clause's
enclosing section title), and the validation method has no second-rater or
self-agreement control, with two run-to-run flips observed directly during
this issue. None of these blocks M2; all are documented in
`docs/PARSING.md`.

The deliverable of [M1-08c] is not the 98%. It is the chain that produced
it: running [M1-04d] for the first time, proving with a matched-pair diff
that it regressed boundaries and fixed none, finding the structural reason
(page-granular reassignment on pages that carry many clauses), disabling it
on evidence, locating the real cause of the largest failure cluster in the
deterministic detector (siblings typeset inconsistently), fixing that, and
then discovering that most of what remained was the metric measuring the
wrong thing. Two silent bugs were found and fixed along the way — a stale
cache that had made the entire escalation pass a no-op, and validation
resumability keyed on sample position, which reused a prior measurement's
judgments.

30/30 documents produce a clause tree with no unhandled exceptions
(verified as part of [M1-08b], after fixing two newly-discovered
segmentation bugs — see `app/src/application/use_cases/
clause_segmentation.py`'s module docstring for the doc 11 heading-gap fix,
and `scripts/build_clause_tree.py`'s `KNOWN_LARGE_CLAUSE_IDS`/
`KNOWN_HIGH_ORPHAN_DOCUMENT_IDS` for the documented, evidence-backed
exemptions covering the rest). Provenance accuracy (100.0%, 50/50) and
`docs/PARSING.md` publication are both satisfied.

## M2 — Ground truth

Build the golden set and the synthetic claim set before any retrieval code
is written, so the golden set does not end up describing what the retriever
already does. Also writes the evaluation protocol, including an honest
account of the fact that one person authors the questions, labels the
answers and builds the system.

**Exit criteria:**
- ≥90 golden questions spanning ≥20 documents, each with reference clause
  ids.
- ≥30 synthetic claims labelled across three verdicts: compatible,
  incompatible, insufficient information — including ≥8 product/claim
  mismatch cases.
- ≥15 questions whose answer is absent from the corpus by construction
  (deductibles, insured amounts, policy periods).
- Blind re-labelling of a 20% sample ≥14 days after first pass; agreement
  rate measured and published in `docs/EVALUATION.md`, whatever it is,
  reference value ≥90% — if below, the cause is documented; disagreements
  resolved and recorded regardless of the rate.
- `docs/EVALUATION.md` complete; golden set tagged `golden-set-v1`.

**[M2-07] amendment (2026-08-25).** The blind-relabelling criterion above
(delay-based, same author, ≥90% reference) was replaced during authoring by
an independent second-reviewer pass — see `docs/EVALUATION.md`'s
"Independent second-reviewer pass" and "What this evaluation cannot
establish" sections for the full rationale and what the replacement does
and does not establish. Measured full-agreement rate: 75.0% (27/36) —
below the ≥90% reference value carried over from the criterion it replaced.
The cause is documented, not hidden: every disagreement in the sample was a
near-miss clause addition or omission on an already-largely-correct answer,
never a verdict-level or unanswerable-mislabelling error, and the review
sample is small enough (per-`question_type` strata as low as `n=3`-`4`)
that a single disagreement swings the stratum rate by ±25-33 points. The
reference value is not relaxed; the number stands next to its cause per
this file's own rule above. 140 golden questions authored, tagged
`golden-set-v1` in git. **M2 closes.**

## M3 — Retrieval

Build and measure the classical RAG pipeline: clause-aware chunking,
embeddings in pgvector, lexical retrieval, hybrid fusion, cross-encoder
reranking and an explicit insufficient-context gate. The hard case this
milestone must handle is the exclusion clause sitting three paragraphs away
from the coverage clause it cancels.

**Exit criteria:**
- Recall@10 and MRR on `golden-set-v1` measured and published for the best
  configuration, reference values Recall@10 ≥ 0.85 and MRR ≥ 0.60 — if
  below, the cause is documented.
- Benchmark matrix committed with all four configurations measured: lexical
  only, dense only, hybrid, hybrid + rerank.
- Exclusion-clause recall reported as a separate number from overall recall.
- Insufficient-context gate precision on the unanswerable subset measured
  and published, reference value ≥80% — if below, the cause is documented.
- Insufficient-context gate recall on the unanswerable subset: 100%,
  enforced by an automated test asserting that every question in the
  unanswerable set triggers the gate, not by review.
- `make build-index` rebuilds the full index from `data/policies/raw/` on a
  clean machine.

**[M3-05] outcome (2026-08-29).** Cross-encoder reranking
(`Alibaba-NLP/gte-multilingual-reranker-base`) of the [M3-04] hybrid RRF top-10,
filtered to each question's SUSEP process + CNPJ.
`RERANK_CANDIDATE_DEPTH` is set to **10** — the shallowest point — from the
`make tune-reranking` sweep (`docs/RERANKING.md`): reranking the hybrid top-10
lifts overall MRR 78.8 → 80.6, R@1 64.5 → 65.8, R@5 83.6 → 86.9 and nDCG@10
80.3 → 82.0, with R@10 unchanged at 91.5% and exclusion-clause recall unchanged
at 92.6% (25/27); every deeper depth trades the top-rank gain away, and depth
≥ 30 pushes exclusion clauses out of the top-10 (the DoD's named regression) and
drops R@10 below the no-rerank baseline. Both M3 exit values still clear their
bars. Three of five pre-registered predictions missed and are published rather
than absorbed, per this file's protocol: the metric lift is real but ~half the
predicted magnitude; the curve peaks at the shallowest depth, not at k ≈ 30–50;
and CPU reranking latency (~9.7 s/query p50 at depth 10 on a Ryzen 5 5600H) is
~50–100× the "tens to low-hundreds of ms" predicted — which makes the CPU
cross-encoder a batch-eval / GPU-serving tool, not a live-path component. The
four-configuration benchmark matrix and `make build-index` remain [M3-08].

**[M3-07] outcome (2026-08-30).** The insufficient-context gate
(`app/src/infrastructure/rag/insufficient_context_gate.py`), a deterministic
decision over the hybrid RRF + rerank signals. It abstains when nothing was
retrieved, when the rank-1 reranked score is below
`TOP_SCORE_ABSTAIN_THRESHOLD` (**0.46**), or — for a question that asks for a
specific policy-instance value of a `docs/SCOPE.md`-absent fact — when that
score is below the stricter `INSTANCE_VALUE_TOP_SCORE_THRESHOLD` (**0.84**).
On `golden-set-v1` it abstains on **all 23** `unanswerable` questions
(**recall 100%**, the exit criterion, locked by a unit test over a committed
signal snapshot) with **zero false positives** (**precision 100%**, ≥ the ≥80%
reference), and no false negatives. The result is a calibration, not an
evaluation: a pure top-score gate manages only 25% precision at 100% recall
(FP 69) because the reranker scores a clause that merely *discusses* a
deductible / a policy period as highly as one that answers a real question —
six of the 23 sit in the answerable band. The instance-value rule closes that
gap, but its two thresholds each sit in a ~0.05-wide gap on this one
23-question set and its value-vs-rule keyword patterns are fitted to it;
`golden-set-v2` must re-run `make eval-insufficient-context-gate`. Two of seven
pre-registered predictions missed (the "clean-absent questions score low"
framing — they don't; the "only the 3 decoys are high" framing — six are), one
was beaten (precision recovered to 100%, not the low-to-mid 80s). Full method,
sweep, fragility table and prediction-vs-actual: `docs/INSUFFICIENT_CONTEXT_GATE.md`.
Wiring the gate into the graph is [M4-04]; the benchmark matrix is [M3-08].

**[M3-08] outcome (2026-08-30). M3 closes.** The benchmark matrix
(`scripts/benchmark_retrieval_matrix.py`, `make eval-retrieval-matrix`,
committed in `docs/RETRIEVAL_BENCHMARK.md`) scores every configuration on the
117 scorable `golden-set-v1` questions over the `--filter default`
SUSEP-process + CNPJ path. **Best configuration — hybrid RRF + cross-encoder
rerank (depth 10) + exclusion co-retrieval (1 slot): Recall@10 92.3%, MRR
80.6%, nDCG@10 82.3%, exclusion-clause recall 100% (27/27), foreign-document
rate 0.0%** — both exit bars (Recall@10 ≥ 0.85, MRR ≥ 0.60) cleared, and every
exclusion reference in the golden set retrieved. The four core rows reproduce
their [M3-03]/[M3-04]/[M3-05] committed numbers exactly and are byte-identical
across two matrix runs (retrieval is deterministic; tolerance is fp32 reranker
rounding across hardware, ≤ ~1 question). **RRF-vs-weighted, re-opened with the
reranker in the loop: RRF wins every metric** (Recall@10 92.3 vs 89.3, MRR 80.6
vs 80.4) — the reranker erased the fusion-stage MRR edge weighted had, and
weighted's weaker candidate *set* is what remains; `DEFAULT_FUSION_STRATEGY`
unchanged, and the [M3-04] deviation is now a measured result. **Real-embedding
ANN A/B** (`make benchmark-ann-index-real`, second measurement in
`docs/EMBEDDINGS.md`): HNSW recall@10 vs. exact is **0.9932** (vs the 0.448
synthetic), but the verdict is unchanged — the filtered default path's planner
reads the btree partition (`Bitmap Heap Scan`), never the index, so
`make build-index` does not build it. **`make build-index`** rebuilds the index
end to end (`parse` → `build-chunks` → `migrate` → `load-chunks` →
`embed-chunks`); it needs the same `LLM_*` + Tesseract environment as
`make parse`, and a file-target prerequisite skips the parse stage when the
parsed corpus is already present. **Deviation, stated per this file:** the DoD's
`make eval-retrieval` is the [M2-06] random self-test, so the matrix is realised
as the new target `make eval-retrieval-matrix`. Pre-registered predictions: the
four-core reproduction held; "keep RRF" held but for a stronger reason than
predicted (RRF wins outright, not on a trade-off); the ANN prediction held.
Not re-run: rerank-depth / `k1`,`b` / `RRF_K` re-tuning — the bars are cleared
and each was tuned on this golden set.

## M4 — Agent graph

Assemble the LangGraph state graph: intake with a conditional clarification
loop, retrieval, two assessment nodes running in parallel, recommendation
synthesis, and a persisted human checkpoint before anything is recorded as
final. Every assertion cites a clause id, and deterministic checks run as
Python code rather than at an LLM's discretion.

**Exit criteria:**
- The graph completes on 100% of the synthetic claim set with no unhandled
  exceptions.
- End-to-end verdict accuracy against the three-class labels measured and
  published, reference value ≥75% — if below, the cause is documented with
  a failure catalogue (retrieval miss, parsing error, reasoning error).
- Citation coverage 100%: zero assessment assertions without a clause id,
  enforced by an automated check, not by review.
- Checkpoint verified across a process restart: state persists, resumes,
  and the human decision is recorded alongside the original system
  recommendation.
- The deterministic/LLM boundary in the consistency node is documented in
  `docs/ARCHITECTURE.md`.

**[M4-10] measurement (2026-09-02).** The whole compiled graph was run over all
**51** synthetic claims (40 + 11 product/claim mismatch; the criterion's "30" is
M2-era drift, noted the same way [M4-02] notes it). Full method, three-arm
tables, failure catalogue and judge results: `docs/END_TO_END_EVALUATION.md`.

| exit criterion | measured | verdict |
| --- | --- | --- |
| Graph completes on 100% of the claim set | 51/51, 0 unhandled exceptions, in both completed arms | **met** |
| End-to-end 3-class accuracy, reference ≥75% | **56.9%** | **below**, catalogued |
| Citation coverage 100%, enforced automatically | 91/91 assertions; CI step `make validate-citation-coverage` | **met** |
| Checkpoint survives a process restart | [M4-09]'s two-process integration test | **met** |
| Consistency deterministic/LLM boundary documented | `docs/ARCHITECTURE.md` ([M4-06]) | **met** |

**The measurement found a defect and fixed it in the same pass**, under the
protocol this document sets out. The retrieval pre-filter ANDed a stated SUSEP
process with intake's narrative-derived product line — two fields that answer
different questions ("which product was bought" vs "what did the claimant
describe") and that disagree by construction on a product/claim mismatch. The
conjunction then selected nothing at all. A database count settled the mechanism
with no model involved: **all 11 mismatch claims had an empty search space**, as
would 9 of the 13 documents the main claim set targets whenever intake reads the
event as CASCO. Both states are published:

| | overall | compatible | incompatible | insufficient_information | mismatch |
| --- | ---: | ---: | ---: | ---: | ---: |
| before the fix | 35.3% | 28.6% | 15.4% | 92.3% | **0.0%** |
| after the fix | **56.9%** | 42.9% | 53.8% | 84.6% | **45.5%** |

**Why 56.9% is below the bar, stated rather than explained away.** 11 of the 22
remaining errors are retrieval misses — now a ranking problem, not an empty-set
one: the system reaches the right document 78.4% of the time but surfaces a
labelled clause for only 33.3% of claims. An independent judge (a different
model family, three passes) scored the assertions **93.8% faithful** against
**59.3% context relevance**, which says the same thing from the other end: what
the system asserts is nearly always supported by the clause it cites, and about
two in five retrieved clauses have no bearing on the claim. The characteristic
error is abstention — `incompatible` precision 92.3% at 50.0% recall — which for
a system a human reviews before anything is recorded is the direction to err in.

**Two of six pre-registered predictions missed and are published unedited** (the
mismatch cohort recovered to 5/11 against a predicted ≥7/11; overall accuracy
landed at 56.9% against a predicted 60–80%). A third arm — the same measurement
with no policy reference in the claim text, which would price what the
pre-filter is worth — was attempted twice and abandoned both times to provider
rate limiting (HTTP 429); no number from it is reported, and that question stays
open.

**Explicit call: M4's disposition is the project owner's.** Four of five exit
criteria are met; the fifth is measured, below its reference value, and
catalogued — the case this document's opening explicitly permits a milestone to
close on. The status table is left at `todo` pending that call.

## M5 — Service and hardening

The notebook-to-service crossing. Wrap the graph in a FastAPI service
following the Clean Architecture already published in
`fastapi-clean-architecture-api`, with real persistence, asynchronous
processing for long documents, tracing, and a Docker Compose stack anyone
can run. Latency and cost stop being unknowns and become measured numbers.

**Exit criteria:**
- `docker compose up -d` starts api + postgres/pgvector + redis (+ tracing)
  with healthchecks, and the assessment flow works end to end through the
  stack.
- Three endpoints working: submit assessment, read assessment state, submit
  human decision to resume the graph.
- Domain and application layers import no FastAPI, SQLAlchemy, LangGraph or
  LangChain symbols — enforced by an import test, not by convention.
- CI runs lint, type-check, unit tests, integration tests against Postgres,
  and builds the image.
- p95 latency and mean cost per assessment measured and recorded.

## M6 — Release and portfolio

Ship it. Documentation that lets a reviewer understand the system in five
minutes, a demo video of the local UI, a slide deck for LinkedIn, and a
tagged release whose README leads with measured numbers rather than
architecture diagrams.

**Exit criteria:**
- README leads with the results table: parsing accuracy, Recall@10, MRR,
  end-to-end verdict accuracy, latency, cost.
- `docs/` complete: ARCHITECTURE, DATA_SOURCES, PARSING, EVALUATION,
  CONTRACT.
- Demo video recorded and linked; slide deck exported to PDF.
- `v0.1.0` tagged with release notes; CI green on `main`.
