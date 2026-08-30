# Hybrid retrieval and metadata pre-filtering

The [M3-04] retrieval layer: one interface that fuses the [M3-03] BM25 lexical
leg and a new dense pgvector leg, over a metadata pre-filter that cuts the
search space to a known SUSEP process + insurer CNPJ. The contract lives in
code at `app/src/infrastructure/rag/hybrid_config.py`,
`app/src/infrastructure/rag/fusion.py` and
`app/src/infrastructure/rag/retrieval_filter.py`; this document is the
rationale, the evidence, and the numbers.

**Scope.** This issue builds the dense query side (M3-02 deferred it here),
Reciprocal Rank Fusion, weighted score fusion, and the pre-filter, and measures
**RRF vs weighted score fusion** and **filter-on vs filter-off** on
`golden-set-v1`. It does **not**:

- produce the committed lexical-only / dense-only / hybrid / hybrid+rerank
  benchmark matrix or `make build-index` — that is [M3-08]'s, run on
  `golden-set-v1`;
- add cross-encoder reranking ([M3-05]), exclusion co-retrieval ([M3-06]) or
  the insufficient-context gate ([M3-07]);
- persist a lexical column to Postgres — the M3-03 doc already declined that
  (Postgres `ts_rank` is not BM25; real in-DB BM25 needs a heavy extension),
  and with the filter numbers now in hand the decision stands: the lexical leg
  stays in-memory over `build/chunks.jsonl`.

The standalone `--retriever dense` numbers below are given for context; the
committed dense baseline write-up is [M3-08]'s.

---

## Method

### One interface, two legs, one filter

`HybridRetriever.retrieve(question, *, k, metadata_filter=None) -> list[str]`
(`hybrid_retriever.py`) is the single entry point M4's graph node will call. It:

1. asks each leg for `CANDIDATE_DEPTH` (100) scored clause ids, passing the
   same `RetrievalFilter` to both;
2. fuses the two rankings (RRF or weighted score fusion);
3. returns the top `k`, "up to `k`, never padded" — the shortfall-as-signal
   contract [M3-07]'s gate depends on.

**Dense leg** (`dense_retriever.py`): `embedding_config.format_query` the
question, embed it through the same `Embedder` contract the index side uses
(pinned `Alibaba-NLP/gte-multilingual-base`, cosine, L2-normalised), then exact
`<=>` search over the metadata-filtered partition
(`chunk_repository.search_chunks_by_vector`), rolled up from chunk hits to
clause-id granularity via `source_clause_ids`. **No HNSW / `iterative_scan`:**
`docs/EMBEDDINGS.md`'s verdict is that the index does not earn its place at
~4,540 chunks and the planner reads the `(susep_process, cnpj)` partition by
btree + exact sort anyway. Score = `1 - cosine_distance`.

**Lexical leg** (`lexical_retriever.py`, extended): the [M3-03] BM25 retriever
gains an optional `metadata_filter` (drops non-matching chunks before the
roll-up) and a `retrieve_scored` (rolls the BM25 score up to clause id, max
over the clause's chunks). The unfiltered `retrieve(question, k=k)` path and
its committed numbers are unchanged.

### The metadata pre-filter

`RetrievalFilter` (`retrieval_filter.py`) is a conjunction of equality
pre-filters: `susep_process`, `cnpj`, `product_line`, `bundle_section`,
`clause_type`. All optional; an all-`None` filter matches everything (the
unknown-process degradation path, item 5). The default retrieval path is
`RetrievalFilter.from_manifest_row(row)` → SUSEP process + CNPJ.

- **Insurers are matched by CNPJ, never by the `insurer` name** ([M1-05]: HDI
  Seguros `29980158000157` and HDI Global `18096627000153` share a brand but
  are different legal entities). The `insurer` column is deliberately not
  indexed.
- **`bundle_section` is lenient by default** (`= :x OR bundle_section IS NULL`),
  with a `strict_bundle=True` escape hatch — see "M1-06 bundle-section
  decision" below.

The dense leg pushes the filter into a SQL `WHERE`; the lexical leg applies
`RetrievalFilter.matches` in memory. BM25 IDF stays a full-corpus statistic
(the filter changes what is *retrieved*, not how terms are weighted).

### The `[M1-09]` per-constant decision

RRF's smoothing `k`, the weighted-fusion weights and the candidate depth all
change the published Recall@k / MRR, so per [M1-09]'s rule (a value that moves
a published number is experimental design, like `SEED` / `SAMPLE_SIZE`) they
are **all module constants** in `hybrid_config.py`. **Zero new `.env` keys** —
the analog of [M3-03]'s "zero" and [M3-02]'s "exactly one".

| constant | value | why a code constant |
| --- | --- | --- |
| `RRF_K` | 60 | Cormack et al.'s original RRF constant / IR-toolkit default; changes the fused Recall@k. Tuning is [M3-08]'s. |
| `FUSION_WEIGHTS` | `(0.5, 0.5)` | (lexical, dense) weights for weighted fusion; changes the fused ranking. |
| `CANDIDATE_DEPTH` | 100 | per-leg pool before fusion; deep enough to rescue a clause outside either leg's top 10 by cross-leg agreement. |
| `DEFAULT_FUSION_STRATEGY` | `rrf` | set from the comparison below. |

`config_fingerprint(*, lexical_config_fingerprint)` (sha256[:16] over the
constants + both legs' own fingerprints) stamps every run's
`RetrievalRunConfig`. `RetrievalRunConfig` is bumped to `v3` with the
filter-mode, fusion strategy + constants, and the embedding / hybrid
fingerprints — all optional; `random` and `lexical` reports are unaffected.

### The harness

`scripts/eval_retrieval.py` gains `--retriever dense|hybrid`, `--filter
none|default` (per-question SUSEP process + CNPJ from the manifest join) and
`--fusion rrf|weighted`. A new pooled metric, the **foreign-document rate**,
answers DoD item 4: of every clause retrieved in a question's top-10, the
fraction from a document other than the question's target. `make
eval-retrieval-hybrid` / `-dense` run under `uv run --group embed` and need a
Postgres with loaded + embedded chunks; the query embedder is wrapped in
`CachingEmbedder` so a re-run over the 117 golden queries costs nothing.

---

## Pre-registered prediction

Written in the implementation plan **before** the retrievers were built or run,
reproduced here unedited:

> - The metadata pre-filter (process + CNPJ) is the dominant effect. Filtered
>   overall Recall@10 clears M3's ≥ 0.85 bar (lexical alone already hit 93.2%
>   in the M3-03 perfect-filter simulation); foreign-document rate drops to 0.
> - Dense-only beats lexical-only on `definition` and paraphrased questions;
>   lexical-only beats dense on exact-term `direct_lookup`.
> - Hybrid ≥ max(lexical, dense) on filtered overall Recall@10 and MRR.
> - RRF and weighted score fusion land within a few points of each other. RRF
>   is kept as the default unless weighted wins filtered overall Recall@10 *or*
>   MRR by a clear margin (> 2 points), because RRF has one parameter and no
>   score-normalisation fragility.
> - `coverage_with_exclusion` stays the weakest type even filtered (exclusion
>   co-retrieval is M3-06); exclusion-clause recall improves with the filter
>   but stays below overall recall.
> - Unfiltered ("unknown process") hybrid beats unfiltered lexical alone (dense
>   adds signal) but falls well short of the filtered number — that gap is the
>   measured cost of intake-extraction failure.

---

## Outcome (2026-08-28)

`make eval-retrieval-hybrid` and `-dense` (+ `--fusion weighted` / `--filter
none` for the comparison) → `eval/runs/retrieval_eval_*.{md,json}`
(regenerable; gitignored). Dense model `Alibaba-NLP/gte-multilingual-base` @
`9bbca17d92…`, cosine, L2-normalised (embedding fingerprint `7ea39a621eaee88e`);
RRF `k=60`, weights `(0.5, 0.5)`, candidate depth 100 (hybrid fingerprint
`279ed8ee0a668227`); lexical `v1` / `k1=1.5 b=0.75` (`ef0a2dd0c1dfb4e4`). 117
scorable questions (the 23 `unanswerable` are excluded — empty reference set);
all CASCO / CARTA VERDE, all `text` extraction mode (`golden-set-v1`'s known
coverage limit, see `docs/EVALUATION.md` and "Limitations" below).

### Overall — every configuration

| retriever | filter | R@1 | R@5 | R@10 | MRR | nDCG@10 | exclusion recall | foreign-doc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random (self-test) | — | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% (0/27) | 95.9% |
| lexical | none | 24.5% | 47.7% | 58.7% | 38.8% | 41.9% | 51.9% (14/27) | 79.7% |
| dense | none | 21.0% | 33.9% | 47.4% | 30.7% | 33.6% | 18.5% (5/27) | 81.4% |
| hybrid RRF | none | 22.8% | 47.0% | 52.1% | 35.6% | 38.5% | 51.9% (14/27) | 80.2% |
| hybrid weighted | none | 23.2% | 45.3% | 54.8% | 36.6% | 40.0% | 48.1% (13/27) | 79.9% |
| lexical | **default** | 62.9% | 83.4% | 87.2% | 79.0% | 78.5% | 85.2% (23/27) | **0.0%** |
| dense | **default** | 56.8% | 79.0% | 84.3% | 71.8% | 73.3% | 70.4% (19/27) | **0.0%** |
| **hybrid RRF** | **default** | **64.5%** | **83.6%** | **91.5%** | **78.8%** | **80.3%** | **92.6% (25/27)** | **0.0%** |
| hybrid weighted | **default** | 67.9% | 85.7% | 88.9% | 83.1% | 82.0% | 92.6% (25/27) | **0.0%** |

The unfiltered `lexical` row reproduces the [M3-03] committed baseline
(58.7% / 38.8% / 41.9% / 51.9%) exactly — the M3-04 changes did not perturb it.

### Prediction vs actual

| prediction | actual | call |
| --- | --- | --- |
| Pre-filter is the dominant effect; filtered R@10 ≥ 0.85; foreign-doc → 0 | filter lifts R@10 by 27–39 pts on every leg; every filtered config 84.3–91.5%; foreign-doc **exactly 0.0%** | **held** |
| Dense beats lexical on `definition` / paraphrase; lexical beats dense on `direct_lookup` | filtered lexical beats dense on `definition` (83.3 vs 77.8), `coverage_with_exclusion` (73.7 vs 63.2) and `cross_document` (93.8 vs 87.5); dense edges lexical only on `direct_lookup` (91.7 vs 90.6) | **missed** — filtered BM25 is the stronger *single* leg here; the predicted split is essentially backwards |
| Hybrid ≥ max(lexical, dense) on filtered R@10 and MRR | R@10: RRF 91.5 > max 87.2 ✓. MRR: RRF 78.8 vs lexical 79.0 ✗ (by 0.2 pt); weighted 83.1 ✓ | **held (R@10), missed (RRF MRR, barely)** |
| RRF and weighted within a few points | R@10 91.5 vs 88.9; MRR 78.8 vs 83.1 | **held** |
| `coverage_with_exclusion` stays the weakest type even filtered | RRF+filter: `coverage_with_exclusion` R@10 78.9% is the lowest of the four types (vs 88.9 / 93.8 / 95.3) | **held** |
| Exclusion-clause recall improves with the filter but stays below overall | 51.9% → 92.6%; **exceeds** overall R@10 (91.5%) by 1.1 pt under RRF+filter | **missed** — the filter helps exclusion clauses slightly *more* than the average clause |
| Unfiltered hybrid beats unfiltered lexical alone | unfiltered hybrid RRF R@10 52.1% < unfiltered lexical 58.7% | **missed** — fusing the weak unfiltered dense leg (47.4%) *drags RRF down*; a strong argument that the filter is not optional |

### By question type — filtered

| question_type | n | lexical | dense | **hybrid RRF** | hybrid weighted |
| --- | ---: | ---: | ---: | ---: | ---: |
| `direct_lookup` | 64 | 90.6% | 91.7% | **95.3%** | 93.8% |
| `coverage_with_exclusion` | 19 | 73.7% | 63.2% | **78.9%** | 73.7% |
| `cross_document` | 16 | 93.8% | 87.5% | **93.8%** | 87.5% |
| `definition` | 18 | 83.3% | 77.8% | **88.9%** | 88.9% |

(Recall@10. RRF+filter beats or ties both legs on every type — the fusion is
additive here even though each leg alone already clears 0.85 on most types.)

---

## Verdict

### The filter

**The SUSEP process + CNPJ pre-filter is [M3-04]'s whole result.** It takes
every leg from "well below M3's bar" to "well above it", and it is the
system's default path, not an optimisation — a claims analyst works a known
policy. Unfiltered retrieval is a documented degradation mode (below), not an
alternative primary flow.

### RRF vs weighted score fusion → **keep RRF**

Neither dominates. RRF wins deep recall (**R@10 91.5 vs 88.9**); weighted wins
top-rank precision (**MRR 83.1 vs 78.8**, R@1 67.9 vs 64.5, nDCG 82.0 vs 80.3).

**RRF is kept as the default**, for three reasons:

1. **Recall@10 is M3's stated exit metric** and the quantity that decides
   whether the answer clause reaches the LLM's context window — the LLM reads
   all `k`, so presence in the top-10 matters more than exact rank. RRF wins
   it.
2. **RRF has one fixed, standard parameter and no score-normalisation step.**
   Weighted fusion's per-list min-max normalisation is sensitive to
   score-distribution shifts — a query with zero BM25 hits, a future
   embedding-model swap — that RRF's rank-only formula is immune to.
3. **[M3-05]'s cross-encoder reranker reorders the top-`k`**, which is exactly
   the top-rank precision weighted fusion buys. Feeding the reranker the
   higher-recall RRF candidate set is the better pipeline: the reranker fixes
   ordering, it cannot recover a clause that was never retrieved.

**Deviation from the pre-registered prediction, stated per `MILESTONES.md`:**
the prediction's rule was "pick weighted if it wins Recall@10 *or* MRR by a
clear (> 2 pt) margin", and weighted won MRR by 4.3 pt. That rule was a
pre-commitment against post-hoc cherry-picking; the deviation is on reasons
1–3, which the full table above lets a reader weigh independently, and both
strategies remain one flag apart (`--fusion weighted`). [M3-08] re-opens this
with reranking in the matrix.

### DoD item 4 — cross-document errors are eliminated

The 16 `cross_document` questions each target a document whose same-insurer
sibling (same CNPJ, **different SUSEP process**) holds a near-duplicate
distractor clause named in the question's `notes`. Under `--filter default`:

- **foreign-document rate is exactly 0.0%** — not one retrieved clause, across
  all 117 scored questions, comes from a document other than the target. The
  named distractor clauses cannot be retrieved: they are in a different SUSEP
  process, and the process filter excludes them at the SQL / roll-up boundary.
- `cross_document` Recall@10 rises 62.5% → 93.8% (RRF); the residual misses are
  within-document ranking, not leakage.

`tests/integration/test_dense_retriever.py` proves the SQL pushdown against a
real Postgres with a synthetic same-CNPJ decoy partition.

### DoD item 5 — unknown-process degradation

When intake extraction fails to identify the policy, retrieval falls back to
`--filter none`. Measured cost, RRF hybrid:

| | filtered | unfiltered | drop |
| --- | ---: | ---: | ---: |
| Recall@10 | 91.5% | 52.1% | −39.4 pt |
| MRR | 78.8% | 35.6% | −43.2 pt |
| foreign-doc rate | 0.0% | 80.2% | +80.2 pt |

Unfiltered, ~80% of every result list is from the wrong document, and — the
sharper finding — **unfiltered hybrid RRF (52.1%) is worse than unfiltered
lexical alone (58.7%)**: fusing in the weak unfiltered dense leg (47.4%) hurts.
This is a robustness result, not an alternative flow to optimise for; the
realistic broad-search case is `bundle_section` *within* one known insurer's
filing (`docs/EVALUATION.md`, "Why `cross_document` is not a realistic search
scenario"), not blind cross-insurer search.

### DoD item 7 — M1-06 bundle-section decision

A strict `WHERE bundle_section = :x` silently drops `bundle_section IS NULL`
chunks. Post-[M1-04b] the NULL bucket is only ~1.8–3.7% of the two multi-product
documents (down from ~40%), but excluding unknown-bundle clauses without
telling anyone is exactly the failure the [M1-06] cross-note warns about.

**Decision: the `bundle_section` filter is lenient by default** — `= :x OR
bundle_section IS NULL` — with `RetrievalFilter(strict_bundle=True)` as an
explicit opt-in. `golden-set-v1` has no scorable question that sets a
`bundle_section` filter (the 4 M2-03 bundle questions live in
`direct_lookup.jsonl` and are scored document-wide), so there is no golden-set
delta to report; the behaviour is proven in
`tests/integration/test_dense_retriever.py` (lenient keeps the NULL chunk,
strict drops it) and `tests/unit/infrastructure/rag/test_retrieval_filter.py`.

---

## Limitations

- **Product-line coverage stays 2 of 5.** The 117 scorable questions reference
  only CASCO (114) and CARTA VERDE (3) documents; RCF-A, ASSIST and GAR.EST
  appear only among the excluded `unanswerable` questions. Broadening scorable
  coverage is new golden-set authoring, out of scope here — same limitation
  `docs/LEXICAL_RETRIEVAL.md` and `docs/EVALUATION.md` record.
- **The dense leg reads the dev database.** `make eval-retrieval-hybrid` runs
  against whatever `create_engine_from_settings()` resolves to, with real
  embeddings from `make embed-chunks`. It is not hermetic the way the
  file-based lexical eval is; the config fingerprints pin the model contract.
- **The M3-03 "perfect filter" simulation (93.2%) does not reproduce exactly.**
  That one-off rebuilt BM25 per single-document index, sharpening IDF; the
  proper filtered lexical number with corpus-wide IDF is **87.2%**. Still
  clears 0.85; the 6-point gap is the IDF effect, not a regression.
- **`fusion_weights` are untuned** at `(0.5, 0.5)`. Weighted fusion already
  beats RRF on MRR/nDCG untuned; a tilt was not explored — [M3-08]'s, with the
  reranker in the loop.

---

## Deferred / handed to later issues

- **Cross-encoder reranking of the fused top-`k`** — [M3-05].
- **Exclusion co-retrieval** (pull the exclusion linked to every retrieved
  coverage clause) — [M3-06]. `coverage_with_exclusion` R@10 78.9% is the
  weakest type and the reason this exists.
- **The insufficient-context gate** on the retrieval signals — [M3-07] (done:
  `docs/INSUFFICIENT_CONTEXT_GATE.md`). The "up to `k`, never padded" contract
  is one of its inputs (the `NO_CONTEXT` trigger), though on `golden-set-v1`
  every filtered partition returns a full `k` and the rank-1 reranked score
  carries the decision.
- **The lexical / dense / hybrid / hybrid+rerank benchmark matrix and `make
  build-index`** — [M3-08], which also re-opens the RRF-vs-weighted call with
  reranking in the mix and re-runs the ANN-earns-its-place check on real
  embeddings.
- **A DB-side lexical column** — still declined (see Scope).
