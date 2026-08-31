# Retrieval benchmark matrix

The [M3-08] consolidation: every retrieval configuration built across M3, scored
on `golden-set-v1` through one harness, in one table, with the exact config
behind each row and a reproducible build path. The runner is
`scripts/benchmark_retrieval_matrix.py` (`make eval-retrieval-matrix`); this
document is the committed table, the per-question-type analysis, and the
verdict.

**Scope.** This issue produces the comparison table and `make build-index`. It:

- runs **lexical only / dense only / hybrid RRF / hybrid RRF + rerank / hybrid
  RRF + rerank + co-retrieval** on the `--filter default` SUSEP-process + CNPJ
  path — the system's real retrieval path (`docs/HYBRID_RETRIEVAL.md`);
- adds one row, **hybrid weighted + rerank + co-retrieval**, to re-open the
  RRF-vs-weighted fusion call `docs/HYBRID_RETRIEVAL.md` deferred here, now with
  the reranker and co-retrieval in the loop;
- re-runs the **ANN-earns-its-place A/B on the real embeddings** (M3-02 measured
  synthetic vectors only) — full method and numbers in `docs/EMBEDDINGS.md`,
  summarised here;
- implements **`make build-index`** end to end.

It does **not**:

- re-run every configuration unfiltered — the unfiltered degradation table stays
  in `docs/HYBRID_RETRIEVAL.md` (the pre-filter is the default path, not an
  optimisation);
- re-tune the reranker candidate depth, BM25 `k1`/`b` or the RRF constant with
  the full pipeline in the loop — the M3 exit bars are cleared and each value
  was tuned on this same golden set (`docs/RERANKING.md`,
  `docs/LEXICAL_RETRIEVAL.md`, `docs/HYBRID_RETRIEVAL.md`); the `make tune-*`
  targets do this if a later issue wants it;
- broaden scorable golden-set coverage past CASCO / CARTA VERDE — new authoring
  ([M2], golden-set-v2).

---

## Method

### The harness

`scripts/benchmark_retrieval_matrix.py` drives the [M2-06] harness exactly as
`scripts/tune_reranking.py` does: one `argparse.Namespace` per configuration,
`_open_retriever` builds the leg → fusion → rerank wrapper → co-retrieval
wrapper, a timing proxy wraps `retrieve()`, and `evaluate_questions` scores it.
No metric code is re-implemented. Every configuration is scored on the same
117 scorable questions (the 23 `unanswerable` carry no reference clauses and are
excluded from every ranking metric — `docs/EVALUATION.md`); all reference CASCO
(114) or CARTA VERDE (3) documents, all `text` extraction mode.

Output: `eval/runs/retrieval_benchmark_matrix.{md,json}` (regenerable,
gitignored), each row stamped with its `RetrievalRunConfig` and every
`config_fingerprint()`. The committed numbers are transcribed below.

### The `make eval-retrieval` deviation

The M3-08 DoD says `make build-index && make eval-retrieval` reproduces the
committed numbers. `make eval-retrieval` is the [M2-06] random self-test
(`tests/eval/test_retrieval_eval_collapse.py`, `docs/EVALUATION.md` both depend
on it), so the matrix is realised as **`make eval-retrieval-matrix`** — a
new target, not a repurposed one. Recorded here and in `MILESTONES.md` per the
repo's state-every-deviation convention.

### Latency and cost

The latency column is per-query `retrieve()` wall-clock over the 117 questions
on the dev machine (AMD Ryzen 5 5600H; RTX 3050 for model scoring), **embedding
and reranker caches warm** — so it is the retrieval-machinery cost with the
model passes served from cache. It is complete for `lexical` / `dense` /
`hybrid` (no model, or the cached query vector), and for the reranked rows it is
the machinery around a cached cross-encoder. The **cold model costs are additive
and measured elsewhere**:

- the query embedder — one short-query forward pass through
  `gte-multilingual-base` (`docs/EMBEDDINGS.md`); a few tens of ms, cached here
  (every `golden-set-v1` query vector is already on disk);
- the CPU cross-encoder at depth 10 — **~9.7 s/query p50** (`docs/RERANKING.md`),
  the number that rules it out of an interactive path; the eval runs it on the
  GPU (~2 min for the whole golden set) and the cache makes every re-run a dict
  lookup;
- exclusion co-retrieval — pure structural lookup over the parsed corpus, `< 1 ms`.

**Dollar cost per query: $0.00.** Both models run locally via
sentence-transformers — no API, no per-token charge — so no price constant is
introduced ([M1-09] stale-pricing rule). The reproducible cost is machine time.

---

## Pre-registered prediction

The four core configurations and the co-retrieval row were each measured and
their predictions pre-registered in [M3-03] / [M3-04] / [M3-05] / [M3-06]; this
matrix confirms they reproduce under one harness (prediction: **exact
reproduction** — retrieval is deterministic) and adds the cross-configuration
latency. The two genuinely new questions are predicted here, unedited:

> **Fusion.** Weighted score fusion keeps a small MRR / R@1 edge over RRF even
> with the reranker and co-retrieval on top — the reranker reorders the top-`k`
> but cannot fully erase the better top-rank ordering weighted's normalised
> score fusion produced at the fusion stage (`docs/HYBRID_RETRIEVAL.md`: MRR
> 83.1 vs 78.8). RRF wins or ties Recall@10. Net call: **keep RRF** as
> `DEFAULT_FUSION_STRATEGY` — Recall@10 is M3's exit metric and feeds the LLM
> context, and the reranker captures most of weighted's top-rank advantage
> anyway.
>
> **ANN.** HNSW recall@10 vs. exact on the real embeddings is **≥ 0.95** — real
> vectors have cluster structure, unlike the 0.448 the synthetic-vector M3-02
> benchmark measured. The planner still reads the `(susep_process, cnpj)` btree
> partition for the default filtered path and never touches the index.
> `make build-index` does **not** call `create_hnsw_index`.

---

## Outcome (2026-08-30)

`make eval-retrieval-matrix` →
`eval/runs/retrieval_benchmark_matrix.{md,json}`. Embedding
`7ea39a621eaee88e`, hybrid RRF `279ed8ee0a668227`, lexical `ef0a2dd0c1dfb4e4`,
reranker `777c0503f1073d52` (depth 10), co-retrieval `7ed4c97c4e8f1cb4`
(1 slot). Run twice: every Recall / MRR / nDCG / exclusion number is
byte-identical across runs.

### The matrix

| configuration | R@1 | R@5 | R@10 | MRR | nDCG@10 | exclusion recall | foreign-doc | latency ms (p50 / mean, warm) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random (self-test) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% (0/27) | 95.9% | 0.0 / 0.0 |
| lexical | 62.9% | 83.4% | 87.2% | 79.0% | 78.5% | 85.2% (23/27) | 0.0% | 12.8 / 13.0 |
| dense | 56.8% | 79.0% | 84.3% | 71.8% | 73.3% | 70.4% (19/27) | 0.0% | 1.4 / 1.8 |
| hybrid RRF | 64.5% | 83.6% | 91.5% | 78.8% | 80.3% | 92.6% (25/27) | 0.0% | 17.5 / 18.0 |
| hybrid RRF + rerank | 65.8% | 86.9% | 91.5% | 80.6% | 82.0% | 92.6% (25/27) | 0.0% | 17.3 / 17.5 |
| **hybrid RRF + rerank + co-retrieval** | **65.8%** | **87.3%** | **92.3%** | **80.6%** | **82.3%** | **100.0% (27/27)** | **0.0%** | 18.0 / 18.3 |
| hybrid weighted + rerank + co-retrieval | 66.2% | 86.0% | 89.3% | 80.4% | 81.1% | 96.3% (26/27) | 0.0% | 17.8 / 18.2 |

Every filtered configuration clears M3's Recall@10 ≥ 0.85 bar. The best
configuration, **hybrid RRF + rerank + exclusion co-retrieval**, clears both:
**Recall@10 92.3%, MRR 80.6%** (bars 0.85 / 0.60). The four core rows reproduce
their [M3-03] / [M3-04] / [M3-05] committed numbers exactly.

### By question type — Recall@10 and MRR

| question_type | metric | lexical | dense | hybrid RRF | + rerank | + rerank + co-retrieval | weighted + rerank + co-retrieval |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_lookup` (n=64) | R@10 | 90.6% | 91.7% | 95.3% | 95.3% | **95.3%** | 93.8% |
| | MRR | 76.3% | 78.5% | 80.7% | 85.1% | **85.1%** | 84.8% |
| `coverage_with_exclusion` (n=19) | R@10 | 73.7% | 63.2% | 78.9% | 78.9% | **84.2%** | 76.3% |
| | MRR | 87.7% | 58.2% | 68.2% | 64.8% | **65.3%** | 66.1% |
| `cross_document` (n=16) | R@10 | 93.8% | 87.5% | 93.8% | 93.8% | **93.8%** | 87.5% |
| | MRR | 78.1% | 81.2% | 82.1% | 80.2% | **80.2%** | 78.1% |
| `definition` (n=18) | R@10 | 83.3% | 77.8% | 88.9% | 88.9% | **88.9%** | 88.9% |
| | MRR | 80.6% | 53.9% | 80.4% | 81.5% | **81.5%** | 81.5% |

**Which configuration wins where, and why:**

- **`direct_lookup` (55% of the scored set) — hybrid RRF, then the reranker.**
  Fusion lifts Recall@10 90.6 / 91.7 → 95.3 (each leg rescues the other's
  misses); the cross-encoder then lifts MRR 80.7 → 85.1 by reading the query and
  the exact-term clause together. Co-retrieval is inert here (these questions
  retrieve coverage clauses, but the eviction rule only ever drops a
  non-reference supporting clause).
- **`coverage_with_exclusion` (the hard type) — co-retrieval is decisive.**
  Recall@10 78.9 → 84.2 and pooled exclusion-clause recall 92.6% → **100%
  (27/27)**: the two exclusion references the reranked base missed are the two
  co-retrieval injects (`docs/EXCLUSION_CO_RETRIEVAL.md`). The reranker slightly
  *lowers* this type's MRR (68.2 → 65.3) — a fluent coverage clause scores above
  the terse exclusion that limits it, the documented [M3-05] effect — but that
  is top-rank ordering, not presence: the DoD metric (is the exclusion in the
  top-10 at all) goes to 100%. Note lexical's high MRR here (87.7%) against a low
  Recall@10 (73.7%): when BM25 finds a cov+excl reference it ranks it near the
  top, but it misses more of them than the hybrid does.
- **`cross_document` and `definition` — fusion saturates them; rerank and
  co-retrieval leave Recall@10 untouched.** RRF already reaches 93.8% / 88.9%;
  the reranker nudges MRR ~2 pt either way (`definition` +1.1, `cross_document`
  −1.9 — the cross-encoder can demote the right clause on a two-document
  question) and Recall@10 not at all.
- **Nowhere does `dense` alone win.** Filtered BM25 is the stronger single leg on
  every type except `direct_lookup` (where dense edges it 91.7 vs 90.6) — the
  same reversal of the pre-registered prediction `docs/HYBRID_RETRIEVAL.md`
  records.

### Prediction vs actual

| prediction | actual | call |
| --- | --- | --- |
| Four core configs reproduce their committed per-stage numbers exactly | lexical 87.2 / dense 84.3 / hybrid RRF 91.5 / +rerank 91.5, MRR 80.6 — identical to [M3-03]/[M3-04]/[M3-05], byte-identical across two matrix runs | **held** |
| Weighted fusion keeps a small MRR / R@1 edge over RRF with the reranker on top; RRF wins/ties Recall@10; keep RRF | **RRF wins every metric**: Recall@10 92.3 vs 89.3 (**+3.0 pt**), MRR 80.6 vs 80.4, nDCG 82.3 vs 81.1, exclusion recall 100% vs 96.3%. The reranker fully erased weighted's fusion-stage MRR edge (both ~80.5), and weighted's worse candidate *set* costs it 3 pt of Recall@10 | **missed** — the "keep RRF" conclusion holds, but for a stronger reason: with the reranker in the loop there is no trade-off left to weigh |
| Real-embedding HNSW recall@10 vs. exact ≥ 0.95; planner still btree; no index in build-index | **0.9932**; the filtered path plan is `Bitmap Heap Scan` (btree partition), not the HNSW index | **held** |

---

## RRF vs. weighted fusion — resolved

`docs/HYBRID_RETRIEVAL.md` kept RRF over weighted at the fusion stage as a
**stated deviation** from its pre-registered rule ("pick weighted if it wins MRR
by > 2 pt"; weighted won MRR by 4.3 pt), on the argument that Recall@10 matters
more and the reranker would capture weighted's top-rank advantage. [M3-08] tests
that argument directly, with the reranker and co-retrieval both in the loop:

| | RRF + rerank + co-retrieval | weighted + rerank + co-retrieval |
| --- | ---: | ---: |
| Recall@10 | **92.3%** | 89.3% |
| MRR | **80.6%** | 80.4% |
| nDCG@10 | **82.3%** | 81.1% |
| exclusion-clause recall | **100.0% (27/27)** | 96.3% (26/27) |

**Verdict: keep RRF; `DEFAULT_FUSION_STRATEGY` unchanged.** The reranker did
exactly what `docs/HYBRID_RETRIEVAL.md` predicted it would — it erased weighted's
MRR edge (80.6 vs 80.4, was 83.1 vs 78.8 before reranking). What is left is
weighted's *weaker candidate set*: it feeds the reranker a top-100 that is 3
points worse on Recall@10 and one exclusion clause short, and the reranker
cannot recover a clause that was never fused in. The M3-04 deviation is now a
measured result, not an argument.

---

## `make build-index`

`make build-index` rebuilds the full searchable index from `data/policies/raw/`:

```
build/parsed_clauses.jsonl (make parse)  →  build-chunks  →  check-embedding-input-length
  →  migrate  →  load-chunks  →  embed-chunks
```

It needs the same environment as `make parse` — the `LLM_*` keys in `.env` (the
clause classifier) and Tesseract (OCR for the two text-layer-less documents) —
plus a running Postgres. The `build/parsed_clauses.jsonl` prerequisite is a
**file target**, so the expensive parse stage (OCR + LLM classification + the
vision boundary pass) is **skipped when the parsed corpus is already present**;
`build-chunks` still runs but is served from the classification cache, `migrate`
is idempotent, `load-chunks` is an idempotent upsert by `chunk_id`, and
`embed-chunks` exits before loading the model when no vector is missing — so a
re-run of `make build-index` is cheap. `make fetch-corpus-artifacts` (the
pre-built corpus + LLM caches, committed as `dist/corpus-artifacts.tar.gz`)
provides `build/parsed_clauses.jsonl` for inspection, but `build-chunks` still
constructs the classifier, so `LLM_*` must be set even then.

`make build-index` stops at the **embedded, queryable Postgres table** (exact
`<=>` search over the filtered partition). It does **not** create the HNSW index
— see below.

### The HNSW index does not earn its place, on the real embeddings either

`make benchmark-ann-index-real` (full report: `docs/EMBEDDINGS.md`, dated second
measurement) builds the HNSW index on the real 4,540 vectors inside a
rolled-back transaction and measures it against the 117 real golden queries:

| | synthetic (M3-02) | real ([M3-08]) |
| --- | ---: | ---: |
| `CREATE INDEX` | ~0.8 s | 0.39 s |
| index size | ~9 MB (0.69× table) | 8.8 MB (0.45× table) |
| HNSW recall@10 vs. exact, full corpus | 0.448 (does not transfer) | **0.9932** |
| exact, single partition (the default path) | ~0.6 ms | 0.58 ms |
| filtered-path plan with the index present | btree bitmap scan | **btree bitmap scan** |

Real recall is 99.3%, not 44.8% — but the verdict is unchanged. The default
retrieval path is filtered to one SUSEP process + CNPJ, and there the planner
reads the `(susep_process, cnpj)` btree partition and sorts it exactly in
~0.6 ms; it never touches the HNSW index, so on the path that matters the index
is inert. It only speeds the *unfiltered* full-corpus scan (9.8 ms → 0.6 ms),
which is a degradation mode, not the default. Against that: 8.8 MB, a build
step, and a 0.7% recall penalty on a path nobody uses. The definition stays in
`infrastructure.rag.ann_index` — one `create_hnsw_index` call away when the
corpus grows ~10×.

### Reproducibility and tolerance

Verified locally (this is a manual measurement — CI runs no retrieval eval and
never installs the `embed` group, as with every M3-03…07 doc). On a machine with
`LLM_*` configured, Tesseract installed, and the `embed` uv group:

```
docker compose up -d postgres && make migrate
make build-index          # from raw; skips the parse stage if build/parsed_clauses.jsonl exists
make eval-retrieval-matrix
```

**Tolerance.** Retrieval is fully deterministic — BM25 with `doc_id` tie-breaks,
exact `<=>` vector search, stable-sorted reranking, RRF / weighted fusion,
structural co-retrieval — so every Recall / MRR / nDCG / exclusion number
reproduces **exactly** given the same corpus and the pinned models on
equivalent hardware. This was confirmed: two matrix runs produced byte-identical
metrics. The one source of drift is fp32 rounding in the cross-encoder across
different GPU / CPU hardware, which could reorder a borderline `(question,
clause)` pair and move a reranked-configuration metric by **≤ ~1 question
(≈ 0.9 pt at n = 117)**. The corpus itself reproduces byte-identical via
`make fetch-corpus-artifacts` (`README.md`); a `make parse` from raw differs
only on `build/manifest.json`'s `built_at_utc`. Latency is hardware-specific and
reported with the machine. Cost is $0.00.

---

## Verdict

**Ship hybrid RRF + cross-encoder rerank (depth 10) + exclusion co-retrieval (1
reserved slot), over the SUSEP-process + CNPJ pre-filter.** On `golden-set-v1`:
Recall@10 **92.3%**, MRR **80.6%**, nDCG@10 82.3%, exclusion-clause recall
**100% (27/27)**, foreign-document rate **0.0%** — both M3 exit bars cleared,
and the milestone's specific hard case (the exclusion clause three paragraphs
from the coverage it cancels) handled: every exclusion reference in the golden
set is retrieved.

The pre-filter is the load-bearing decision — it takes every leg from below
M3's bar to above it (`docs/HYBRID_RETRIEVAL.md`). On top of it, fusion buys ~4
pt of Recall@10 over the better single leg, the reranker buys ~2 pt of MRR and
~3 pt of R@5 (concentrated in `direct_lookup`), and co-retrieval closes the
exclusion gap to zero. RRF beats weighted fusion once the reranker is in the
loop, resolving the M3-04 deviation.

---

## Limitations

- **Product-line coverage stays 2 of 5.** The 117 scorable questions reference
  only CASCO (114) and CARTA VERDE (3); RCF-A, ASSIST and GAR.EST appear only
  among the excluded `unanswerable` questions — the `golden-set-v1` limit
  `docs/EVALUATION.md`, `docs/HYBRID_RETRIEVAL.md` and `docs/RERANKING.md` all
  record. `by_product_line` is populated for 2 lines.
- **Latency is a warm-cache, single-machine number.** It isolates the retrieval
  machinery; the cold model costs are cited from `docs/EMBEDDINGS.md` and
  `docs/RERANKING.md`, not re-derived. The CPU cross-encoder (~9.7 s/query)
  makes the reranked configurations a batch-eval / GPU-serving tool, not an
  interactive-path component as written — M4 picks its own serving hardware
  (`docs/RERANKING.md`).
- **The dense leg reads the dev database** and the models load locally; not
  hermetic the way the file-based lexical eval is. The config fingerprints pin
  the contract.
- **`coverage_with_exclusion` top-rank precision is still soft** (MRR 65.3%).
  Co-retrieval guarantees the exclusion is *present*; ordering it above the
  fluent coverage clause is the reranker's job and it does the opposite. M4's
  assessment node reads all `k`, so presence is what matters — but a smaller
  context budget would expose this.

---

## Deferred / handed to later issues

- **Re-tuning `RERANK_CANDIDATE_DEPTH` / BM25 `k1`,`b` / `RRF_K` with the full
  pipeline in the loop** — the `make tune-*` targets do this; not re-run because
  the M3 bars are cleared and each value was tuned on this same golden set. A
  later issue with a changed corpus or golden-set-v2 should.
- **Broadening scorable golden-set coverage** past CASCO / CARTA VERDE — new
  authoring, golden-set-v2.
- **The HNSW index** — one `create_hnsw_index` call away when the corpus grows
  ~10× (`docs/EMBEDDINGS.md`).
- **Wiring the [M3-07] insufficient-context gate into the graph** — [M4-04].
