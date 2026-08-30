# The insufficient-context gate

The [M3-07] stage: a deterministic decision over the retrieval signals the
pipeline already computes, that turns *"the corpus does not contain this"* into a
first-class outcome instead of a confident wrong answer. The contract lives in
code at `app/src/infrastructure/rag/insufficient_context_config.py`
(thresholds + fingerprint) and
`app/src/infrastructure/rag/insufficient_context_gate.py` (the pure decision
function); this document is the method, the evidence and the numbers.

**Why this matters.** The policy corpus is *condições gerais* of registered
SUSEP products. By construction it holds none of: a contracted deductible, an
insured amount, a premium, a policy period, an endorsement (`docs/SCOPE.md`).
[M2-05] authored 23 golden questions that ask for exactly those facts. Without a
gate, M4's assessment nodes get a fluent, well-cited clause that *discusses* the
fact and produce an answer with a real clause id and a wrong number — the single
most expensive error this project can make.

**Scope.** This issue builds the gate, calibrates its thresholds on the 23
`unanswerable` questions from `golden-set-v1`, and reports precision / recall /
the false-negative cases. It does **not**:

- wire the gate into the LangGraph retrieval node — that is [M4-04], which
  "sets the insufficient-context flag from the [M3-07] gate and routes
  accordingly"; the gate is a pure `GateSignals -> InsufficientContextResult`
  function and M4-04 assembles the input;
- produce the four-configuration benchmark matrix or `make build-index` — [M3-08];
- re-tune BM25 / RRF / the reranker depth / the co-retrieval slot count — those
  stay their pinned `docs/*.md` values;
- add a query-intent classifier or an LLM answerability check — the gate is
  deterministic retrieval-signal logic.

---

## Method

### The signals

`GateSignals` (`insufficient_context_gate.py`) is what the caller hands the gate,
built from the pipeline it already ran — **hybrid RRF + cross-encoder rerank**,
filtered to the question's SUSEP process + CNPJ (the `docs/HYBRID_RETRIEVAL.md` /
`docs/RERANKING.md` base). The gate reads:

- `top_score` — the rank-1 reranked clause's score. The reranker
  (`Alibaba-NLP/gte-multilingual-reranker-base`) emits a **sigmoid score in
  [0, 1]**; on `golden-set-v1` the pooled distribution over answerable
  candidates runs ~0.06–0.97, p50 ≈ 0.55.
- `reranked_scores` — every candidate's score, for the score-distribution
  features the sweep evaluated and rejected (below).
- `retrieved_clause_types` — for the clause-type-coverage diagnostic (below).
- `n_returned` vs `k_requested` — the "shortfall as signal" contract
  (`docs/HYBRID_RETRIEVAL.md`): a filtered search legitimately returning fewer
  than `k`.

**Calibrate on hybrid + rerank, not `+ co-retrieval`.** Exclusion co-retrieval
([M3-06]) is a structural post-process that never re-scores; the rank-1 reranked
score is byte-identical with or without it. The gate logically sits right after
reranking.

### The rule

`evaluate_gate(question, signals)` abstains (`sufficient=False`) when **any** of:

1. **`NO_CONTEXT`** — `n_returned == 0` (the filtered partition admitted nothing).
2. **`LOW_RELEVANCE`** — `top_score < TOP_SCORE_ABSTAIN_THRESHOLD` (**0.46**).
   The best clause is a poor topical match; retrieval found nothing that could
   answer.
3. **`UNVERIFIED_INSTANCE_VALUE`** — the question asks for a *specific*
   policy-instance value of a `docs/SCOPE.md`-absent fact
   (`needs_verified_instance_value`) **and**
   `top_score < INSTANCE_VALUE_TOP_SCORE_THRESHOLD` (**0.84**). For those
   questions a topically-strong clause that merely *discusses* the deductible /
   limit / premium / period / endorsement is the exact failure mode, so
   retrieval must clear a much higher bar before the context is trusted.

`needs_verified_instance_value(question)` is `classify_missing_information(q) !=
OTHER` **and** `asks_for_instance_value(q)`:

- `classify_missing_information` — deterministic keyword match onto the five
  `docs/SCOPE.md` "what the corpus is not" categories (`DEDUCTIBLE`,
  `INSURED_AMOUNT`, `PREMIUM`, `POLICY_PERIOD`, `ENDORSEMENT`), priority order
  `ENDORSEMENT` first (an endorsement question is about a policy-instance change
  even when it also names a premium or a limit — matches the [M2-05] authoring
  of `unanswerable-007..-010`). Also the label on every abstention's structured
  result.
- `asks_for_instance_value` — a value/date request (`valor exato/nominal/total`,
  `montante`, `em qual data`, `número do endosso`, a `contratada` / `fixada`
  figure) that is **not** a rule / manner / quantity / yes-no question (`de que
  forma`, `como`, `quais`, `quantas`, `se pode`, `onde`, `há cobertura`, …). The
  linguistic distinction between "what is the number" and "how is the number
  determined" — a condições gerais document genuinely answers the second.

### The structured result

An abstention returns an `InsufficientContextResult`, not a bare `False`
([M3-07] DoD "a structured 'insufficient' result naming what is missing"):
`trigger` (which rule fired), `missing_category`, a one-sentence `explanation` in
the `docs/SCOPE.md` framing (SCOPE-compliant vocabulary — never
"covered"/"denied"), `top_score`, `threshold`, and `closest_clause_ids` (the top
3 the retriever did surface, so a reviewer sees what it weighed).

### The `[M1-09]` per-constant decision

Both thresholds determine the exact precision/recall the M3 exit criterion is
measured against, so per [M1-09]'s rule (a value that moves a published number is
experimental design, folded into `config_fingerprint()`) they are **module
constants** in `insufficient_context_config.py`. `config_fingerprint()` reads
them at call time and is stamped into the calibration run and the committed
snapshot. **This issue introduces no new `.env` key** — the analog of
`docs/RERANKING.md`'s "zero" and `docs/EXCLUSION_CO_RETRIEVAL.md`'s "zero". The
`_INSTANCE_VALUE_RE` / `_STRUCTURAL_RE` patterns are code, not data or `.env`:
they are part of the calibrated decision and change the fingerprint's meaning by
changing what the sweep measured, though not its literal digest.

### The harness

`scripts/eval_insufficient_context_gate.py` (`make eval-insufficient-context-gate`,
run as `python -m scripts.eval_insufficient_context_gate` under `uv run --group
embed`, needs Postgres with loaded + embedded chunks) runs the pipeline over
**all 140** golden questions — `evaluate_questions` skips `unanswerable`
questions and drops the reranker scores, so this reimplements the loop the way
`scripts/tune_reranking._rerank_at_max_depth` does. It sweeps the primary floor,
reports the operating point, the per-rule attribution, the calibration fragility,
the decoy split, the per-category top-score table and the clause-type
diagnostic, and writes:

- `eval/runs/insufficient_context_gate.{md,json}` — gitignored, regenerable;
- `eval/insufficient_context_gate_signals.json` — a **committed** per-question
  signal snapshot. `tests/unit/infrastructure/rag/test_insufficient_context_gate.py`
  replays it to enforce the 100%-recall guarantee in `make check` with no
  Postgres, no GPU, no `embed` group; the eval-marked
  `tests/eval/test_insufficient_context_gate_baseline.py` re-derives the signals
  live as a retrieval-drift guard. Committing an eval artifact is a deliberate
  deviation from [M3-04]/[M3-05]/[M3-06] (which ship eval-marked tests only),
  justified by the DoD's "enforced by an automated test ... not by review";
  precedent is `eval/parsing_quality_sample.csv`.

---

## Pre-registered prediction

From the [M3-07] implementation plan, approved **before** the calibration run,
reproduced unedited:

> 1. the 20 clean-absent unanswerable questions score well below the answerable
>    median (~0.55): top-score ~0.15–0.45;
> 2. the 3 decoys (`unanswerable-002/-003/-011`) score higher (~0.50–0.75) and
>    set the 100%-recall threshold;
> 3. at that threshold the ≥80% precision bar is *likely missed* (FP > 5), cause
>    = the decoys (a score-based gate cannot tell "a franquia clause stating a
>    formula" from "a franquia clause stating the number" — the query is
>    lexically about franquia either way);
> 4. a secondary term (`mean_top5` or `asks_for_instance_value`) recovers
>    precision into the low-to-mid 80s by letting the top-score threshold drop;
> 5. clause-type coverage is unimplementable as an independent runtime signal →
>    measured, not shipped;
> 6. the `n_returned < k_requested` shortfall term fires 0 times on the 23;
> 7. recall is 100% by construction and locked by the unit test — the risk it
>    guards is future retrieval drift lifting a decoy over the line.

---

## Outcome (2026-08-30)

`make eval-insufficient-context-gate`, `golden-set-v1`, hybrid RRF + rerank @
`--filter default`, reranker fingerprint `777c0503f1073d52`, gate fingerprint
`c29b8ebee67be01b`. Scoring on an RTX 3050 (device-independent).

### Top-score distributions

| group | n | min | mean | max |
| --- | ---: | ---: | ---: | ---: |
| decoy (`-002/-003/-011`) | 3 | 0.726 | 0.775 | 0.830 |
| clean-absent unanswerable | 20 | 0.367 | 0.509 | 0.830 |
| answerable | 117 | 0.483 | 0.794 | 0.966 |

The prediction's framing is wrong: **the decoys are not uniquely high.** Three
*clean-absent* questions — `-001` (0.722), `-004` (0.830), `-019` (0.756) — tie
them. Six unanswerable questions in total sit at 0.72–0.83, squarely inside the
answerable band. The reranker scores a clause that is *about* the deductible /
the policy period as highly as one that answers a real question, whether or not
that clause states a number.

### A pure top-score gate cannot clear the bar

The primary floor alone (rule 3 disabled):

| `TOP_SCORE_ABSTAIN_THRESHOLD` | recall | precision | TP | FP | FN |
| ---: | ---: | ---: | ---: | ---: | ---: |
| ~0.831 (the 100%-recall point) | 100.0% | **25.0%** | 23 | 69 | 0 |
| 0.60 | 73.9% | 73.9% | 17 | 6 | 6 |
| 0.55 | 69.6% | 88.9% | 16 | 2 | 7 |
| 0.46 | 39.1% | 100.0% | 9 | 0 | 14 |

For 100% recall the floor must exceed the highest `unanswerable` top-score,
0.830, which abstains on **69 of the 117 answerable questions** — precision
25.0%, bar (FP ≤ 5) missed by an order of magnitude. Prediction 3 held (bar
missed) but the cause is broader than the decoys: at 0.46 the floor already
catches 9 of the 23 with zero false positives, but the other 14 need a score
above 0.46 and are indistinguishable from real answers by score alone.

### The instance-value rule closes the gap

Every unanswerable question asks for a specific figure or date of a
`docs/SCOPE.md`-absent fact; `needs_verified_instance_value` is True for all 23.
Only **2 of 117** answerable questions also trip it — `cross_document-002` and
`cross_document-004`, whose answers really are in the corpus and which the
reranker scores 0.966 and 0.883, both clear of the 0.84 strict floor.

At the pinned `(TOP_SCORE_ABSTAIN_THRESHOLD, INSTANCE_VALUE_TOP_SCORE_THRESHOLD)`
= **(0.46, 0.84)**:

| metric | value |
| --- | --- |
| recall (unanswerable subset) | **100.0%** (23/23) |
| precision | **100.0%** (23/23; **0 false positives**; bar: FP ≤ 5) |
| false negatives (answer when it should abstain) | **none** |
| caught by the primary floor | 9 (`-005/-006/-009/-013/-014/-016/-021/-022/-023`) |
| caught by the instance-value rule | 14 (`-001/-002/-003/-004/-007/-008/-010/-011/-012/-015/-017/-018/-019/-020`) |

Prediction 4 is **beaten**: `asks_for_instance_value` recovered precision to
100%, not the low-to-mid 80s — but on a 23-point fit, see Limitations.

### Prediction vs actual

| # | predicted | actual | call |
| --- | --- | --- | --- |
| 1 | clean-absent ~0.15–0.45, well below the answerable median | mean 0.509, range 0.367–0.830; three clean-absent questions score 0.72–0.83 | **missed** |
| 2 | only the 3 decoys are high (~0.50–0.75) | 6 questions at 0.72–0.83; decoys reach 0.83 | **missed** |
| 3 | pure top-score misses the 80% bar (FP > 5) | 25.0% precision, FP 69, at 100% recall | **held** (cause broader) |
| 4 | a secondary term recovers to low-to-mid 80s | recovers to 100% (0 FP) | **beaten** — but a 23-point fit |
| 5 | clause-type coverage: measured, not shipped | diagnostic only (`condition` 83, `definition` 64, `coverage` 46, …) | **held** |
| 6 | shortfall term fires 0 times | every question returns `n=10`; it fires 0 times | **held** |
| 7 | recall 100%, unit-test-locked | recall 100%, locked by the snapshot unit test | **held** |

Also considered and **rejected**: `mean(reranked_scores[:5])` and
`n_above(floor)` as OR-terms — no threshold pair reaches 100% recall with them,
because for the 6 high unanswerable questions the *whole* top-5 neighbourhood is
topically strong. Score distribution carries no separating power the top score
doesn't already have.

---

## Verdict

**Keep the gate at `(TOP_SCORE_ABSTAIN_THRESHOLD, INSTANCE_VALUE_TOP_SCORE_THRESHOLD)`
= (0.46, 0.84).** On `golden-set-v1` it abstains on all 23 `unanswerable`
questions (100% recall, the [M3-07] DoD guarantee, locked by a unit test) with
zero false positives (100% precision), clearing the M3 exit reference value of
≥80% with margin.

**The 100%-recall guarantee is real; the precision number is a calibration, not
an evaluation.** The primary floor and the strict floor were each placed inside a
gap on this one 23-question set:

| floor | value | gap it sits in | width |
| --- | ---: | --- | ---: |
| `TOP_SCORE_ABSTAIN_THRESHOLD` | 0.46 | highest floor-only unanswerable 0.437 → lowest answerable 0.483 | 0.046 |
| `INSTANCE_VALUE_TOP_SCORE_THRESHOLD` | 0.84 | highest instance-value unanswerable 0.830 → lowest answerable instance-value question 0.883 | 0.053 |

Both gaps are narrow. A `golden-set-v2` question landing in either one moves the
number. The `_INSTANCE_VALUE_RE` / `_STRUCTURAL_RE` patterns were tuned against
exactly the 23 `unanswerable` questions and the 9 answerable questions that
mention the same terms — a fitted classifier, not a held-out result. What is
*not* a fit: the primary-floor idea (a poor topical match means retrieval found
nothing), the five-category anchor (it is `docs/SCOPE.md`'s own list), and the
value-vs-rule linguistic distinction the instance-value rule turns on.

For M4 this matters less than it looks: the gate is the *first* of several
layers. M4-05's compatibility node reads the retrieved clauses and can still
answer `insufficient_information` on its own; M4-06's consistency node is a
second deterministic check. The gate's job is to stop the confident-wrong-number
answer before it starts, and on the evidence available it does.

---

## Limitations

- **Calibrated on 23 points, no held-out set.** Precision 100% / 0 FP is a fit.
  The fragility table above is the honest read; `golden-set-v2` must re-run
  `make eval-insufficient-context-gate` and re-check the patterns.
- **The reranker cannot distinguish a topical clause from an answering one.**
  For 6 of the 23 questions the best clause scores 0.72–0.83 — the answerable
  band. The gate only catches them because they *also* ask for a specific value
  (rule 3); a future unanswerable question that asks a *rule* question about an
  absent fact and retrieves a strong topical clause would slip through.
- **No runtime query-intent model.** `needs_verified_instance_value` is a
  keyword heuristic over the question text. The [M3-07] DoD's "clause-type
  coverage relative to what the question needs" is realised here as
  question-need analysis (we infer the need from the question) because the
  clause-type-coverage form needs a query-intent component that does not exist
  before M4's intake node. The clause-type distribution of the unanswerable
  top-k (`condition` 83, `definition` 64, `coverage` 46, `procedure` 23,
  `exclusion` 13) is reported as a diagnostic only.
- **`classify_missing_information` is best-effort.** `unanswerable-023` ("prêmio
  adicional após a emissão do endosso") is labelled `ENDORSEMENT`; [M2-05]'s
  authoring `notes` frame it as a premium question. Both are honest
  "policy-instance fact absent by construction" statements — the label never
  gates, only explains.
- **Product-line coverage is 2 of 5 for the answerable class.** The 117 scorable
  questions are CASCO / CARTA VERDE only (`docs/EVALUATION.md`); the 23
  `unanswerable` questions span RCF-A, ASSIST and GAR.EST too, but the
  false-positive rate is measured only against CASCO / CARTA VERDE.
- **CPU reranker latency is inherited from [M3-05]** (~10 s/query p50). The gate
  adds no model call of its own; it reads signals the pipeline already computed.

---

## Deferred / handed to later issues

- **Wiring into the LangGraph retrieval node** and the routing decision —
  [M4-04].
- **A held-out unanswerable set** — a `golden-set-v2` authoring task; the gate
  cannot be *evaluated* (as opposed to calibrated) without one.
- **A real query-intent classifier** to replace the `needs_verified_instance_value`
  keyword heuristic — future work, most naturally alongside M4's intake node.
- **The four-configuration benchmark matrix** — [M3-08].
