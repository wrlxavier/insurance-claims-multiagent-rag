# Embeddings

The dense-retrieval embedding model for [M3-02], its version pin, and the
distance / normalisation / prefix decisions that [M3-04] and the embedding
pipeline both depend on. The contract lives in code at
`app/src/infrastructure/rag/embedding_config.py`; this document is the
rationale and the evidence.

**Scope.** This issue *chooses* a model with real Portuguese support and
records why. It does **not** compare dense models against each other — the
lexical-only / dense-only / hybrid / hybrid+rerank matrix is [M3-08]'s, run
on `golden-set-v1`. The numbers below establish "this model handles
Brazilian Portuguese", not "this model is the best one".

## The model: `Alibaba-NLP/gte-multilingual-base`

mGTE's base multilingual embedding model (Zhang et al., *mGTE: Generalized
Long-Context Text Representation and Reranking Models for Multilingual Text
Retrieval*, EMNLP 2024 industry track, [arXiv:2407.19669](https://arxiv.org/abs/2407.19669)).

| Property | Value | Source |
| --- | --- | --- |
| Parameters | 305M, encoder-only | model card, "Model Information" |
| Output dimension | 768 | model card |
| Max input | 8192 tokens | model card |
| Languages | 75, Portuguese (`pt`) included | model card `language:` list |
| Prefix / instruction | none | model card usage examples (see below) |
| Similarity | cosine over L2-normalised vectors | model card usage examples |
| License | Apache-2.0 | model card |

It ships a custom `NewModel` architecture, so `sentence-transformers` /
`transformers` load it with `trust_remote_code=True`
(`EMBEDDING_TRUST_REMOTE_CODE` in `embedding_config.py`). That is acceptable
here only because the code is fetched at a pinned revision, not a branch —
see the next section.

## Why: real Brazilian-Portuguese support (evidence, not assumption)

The corpus is Brazilian-Portuguese SUSEP regulatory text, so the evidence
that matters most is a Brazilian-Portuguese benchmark, not a multilingual
average.

- **MTEB-PT** — *Beyond Multilingual Averages: MTEB-PT, a Benchmark for
  Portuguese Sentence Encoders*
  ([arXiv:2607.04071](https://arxiv.org/abs/2607.04071)). A Brazilian-Portuguese
  benchmark of 14 datasets over STS, classification, retrieval and reranking,
  evaluating 17 open- and closed-source models. On the **retrieval** family
  (`WebFAQRetrieval`, `WikipediaRetrievalMultilingual`, `MultiLongDocRetrieval`,
  all Portuguese), Table 3 reports **nDCG@10 76.9** for
  `gte-multilingual-base` — the highest of any peer-reviewed open model
  without dataset-coverage caveats, ahead of `multilingual-e5-large` (74.5)
  and `multilingual-e5-large-instruct` (70.1). This is a direct measurement
  on the corpus's exact language and task shape (asymmetric passage
  retrieval).
- **mGTE paper** — Portuguese is an explicit target language of the model's
  training and evaluation, not incidental coverage. The paper evaluates on
  MIRACL (18 languages, avg nDCG@10 62.1 for the dense model) and MLDR
  (13 languages *including Portuguese*, avg nDCG@10 56.6 for the dense
  model).
- **Model card** — `pt` appears in the model's declared `language:` list;
  the card describes SOTA multilingual retrieval "compared to models of
  similar size".

### No prefix / instruction

The model card's retrieval usage examples (`transformers` and
`sentence-transformers`) embed queries and documents **directly**, with no
`query:` / `passage:` string prepended — unlike the E5 family, whose model
card mandates those prefixes and whose scores degrade without them. So for
this model the [M3-02] "input prefix" DoD item resolves to: *no prefix on
either side*. It is still encoded in one place —
`embedding_config.QUERY_PREFIX` / `PASSAGE_PREFIX`, both `""`, applied only
through `format_query()` / `format_passage()` — so a later swap to a
prefix-requiring model has a single edit point and the drift test
(`tests/unit/infrastructure/rag/test_embedding_config.py`) forces the index
and query sides to change together.

## Version pin

`EMBEDDING_MODEL_REVISION = "9bbca17d9273fd0d03d5725c7a4b0f6b45142062"` — the
Hugging Face Hub commit (last modified 2025-07-05), **not** the floating
`main` alias. A provider-side re-upload under `main` would silently change
the vectors behind an already-published Recall@k number; pinning the commit
makes that impossible without an explicit, reviewable bump. The revision
also pins the `trust_remote_code` modelling files (`scripts/gte_embedding.py`
in the repo), which are fetched at the same revision.

The pin is a code constant, not a `.env` value, by [M1-09]'s per-constant
rule: the model id, revision, dimensionality, metric, normalisation and
prefix together determine the exact vectors, so they are part of the
experimental design — like `SEED` / `SAMPLE_SIZE` in
`scripts/sample_parsing_quality.py`. `.env`'s `EMBEDDING_MODEL` stays as the
human-facing name and is cross-checked against `EMBEDDING_MODEL_ID` by a
test.

## Maximum input length vs. [M3-01]'s chunk rules

The model truncates input past **8192 tokens** (special tokens included).
[M3-01] caps a chunk body at `CHUNK_MAX_CHAR_COUNT=3000` characters, plus the
prepended ancestor-path breadcrumb; the single largest chunk in the built
corpus is 3,300 characters (`docs/CHUNKING_REPORT.md`).

`make check-embedding-input-length`
(`scripts/check_embedding_input_length.py`) settles the fit empirically
rather than by a chars-per-token estimate: it loads the pinned tokenizer and
tokenises every row of `build/chunks.jsonl` exactly as it would be embedded
(`format_passage`, special tokens included).

Result over the 4,540-chunk corpus (`clause_segmentation_version=v7`,
`chunking_version=v1`):

| chunks | min | p50 | p90 | p99 | max | over 8192 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4540 | 16 | 236 | 494 | 731 | 1015 | 0 |

The largest chunk is **1,015 tokens** — about one eighth of the window.
[M3-01]'s chunk-length rules fit with wide margin; **no truncation is
possible and no constraint is reported back to [M3-01]**. Portuguese legal
prose tokenises at roughly 3 characters per token here, so even a
hypothetical chunk at the 3,000-character cap would land near 1,000 tokens.

The check exits non-zero if any chunk ever exceeds the limit — that is the
signal to lower `CHUNK_MAX_CHAR_COUNT` in [M3-01], not to embed truncated
text.

## Distance metric and normalisation

Fixed once, in `embedding_config.py`, and used identically on both sides:

- `DISTANCE_METRIC = DistanceMetric.COSINE`
- `NORMALIZE_EMBEDDINGS = True` — both stored chunk vectors and query
  vectors are L2-normalised, matching the model card's own usage
  (`F.normalize(embeddings, p=2, dim=1)` / `normalize_embeddings=True`).
  With unit vectors, cosine distance and `1 - inner_product` coincide.

This is recorded here for [M3-04] rather than in a comment. Concretely it
means:

- **Index side:** the `embedding halfvec(768)` column. Ordering queries use
  `ChunkRow.embedding.cosine_distance(...)` — the `<=>` (cosine distance)
  operator. The HNSW ANN index (`halfvec_cosine_ops`) is defined in
  `infrastructure.rag.ann_index` — **not** an Alembic migration; the measured
  default retrieval path is exact `<=>` over the metadata-filtered partition
  (see "ANN index vs. exact search" below).
- **Query side** ([M3-04]): the retriever orders by the same `<=>`.

`halfvec` (not `vector`) for storage follows [M0-08]'s decision in
`docs/DATABASE.md`; at 768 dimensions the column is within the plain-`vector`
index limit too, but half-precision storage is the project default.

## The embedding pipeline

`infrastructure.rag.embedding_pipeline.embed_missing_chunks` fills
`chunk.embedding` for every chunk that has no vector yet.

- **Batched.** Chunks are embedded `EMBEDDING_BATCH_SIZE` at a time (64 by
  default; `.env`-backed via `EmbeddingSettings` — see the per-constant table
  below); each chunk's `embedded_text` is run through `format_passage` first
  (identity for this model), so the string embedded on the index side is exactly
  the one `check_embedding_input_length` tokenised.
- **Resumable cursor.** The cursor is `WHERE embedding IS NULL`
  (`chunk_repository.fetch_chunks_missing_embedding`), and each finished batch
  is committed. A run killed part-way keeps every batch it completed; re-running
  embeds exactly the remainder — nothing duplicated, nothing skipped
  (`tests/integration/test_chunk_embedding.py::test_interrupt_mid_corpus_then_resume`).
  `upsert_chunks` deliberately never writes `embedding`, so a metadata refresh
  does not reset the cursor.
- **Retry policy — reused, with a caveat.** Each embed call is wrapped in the
  shared 3-attempt / 5s policy (`application.use_cases.llm_retry_defaults`),
  re-raising on exhaustion. The [M3-02] DoD allows a different policy "if
  embedding rate limits justify" one: they do not, because the pinned model
  runs **in-process** — there is no API and no rate limit. The policy is kept
  unchanged anyway, for uniformity with every other batch job and as thin cover
  for a transient local failure (a first-call model load, an OOM that clears),
  and so nothing changes if a remote embedding API is swapped in later. What
  actually makes an interrupted run safe is the resumable cursor, not the
  retry.
- **No live model in tests.** Unit and integration tests drive the pipeline
  with a deterministic `FakeEmbedder`, per the [M1-05b]/[M1-04d] precedent — no
  live model call anywhere in the suite.

### The real embedder

`infrastructure.rag.sentence_transformer_embedder.SentenceTransformerEmbedder`
loads the pinned model with `sentence-transformers` and embeds in-process:
`SentenceTransformer(EMBEDDING_MODEL_ID, revision=EMBEDDING_MODEL_REVISION,
trust_remote_code=True)`, then `model.encode(..., normalize_embeddings=True)`.

`sentence-transformers` (and its `torch` / `transformers` dependencies) is the
optional **`embed`** dependency group in `pyproject.toml`, deliberately kept out
of `[tool.uv] default-groups` so a plain `uv sync` and CI stay torch-free. The
module imports without the group installed (the heavy import is deferred to the
constructor); only `make embed-chunks` needs it, and runs `uv run --group embed`.

Two Make targets, composable into [M3-08]'s `make build-index`:

- `make load-chunks` upserts `build/chunks.jsonl` into the `chunk` table
  (`upsert_chunks` — idempotent on the deterministic `chunk_id`, and it never
  writes `embedding`).
- `make embed-chunks` (depends on `load-chunks`) composes
  `CachingEmbedder(SentenceTransformerEmbedder(...))`, runs `embed_missing_chunks`,
  and writes the cost report below. If no chunk is missing a vector it exits
  before loading the model, so chaining it into `build-index` is cheap.

**`trust_remote_code` and the `transformers` pin.** On the first run the model's
`configuration.py` / `modeling.py` are fetched and executed from
`Alibaba-NLP/new-impl` — and HF's `repo--module` auto-map form loads that at
*its* `main`, **not** at `EMBEDDING_MODEL_REVISION` (which pins only the
`gte-multilingual-base` weights). That code's RoPE path raises `IndexError` on
`transformers >= 5` / `sentence-transformers >= 4`; the `embed` group holds both
to the tested 4.4x / 3.x line, and the lockfile is the exact pin. An offline
machine needs a warm Hugging Face cache to reproduce at all.

## Corpus embedding cost (2026-08-28, AMD Ryzen 5 5600H)

**Dollar cost: $0.00 as of 2026-08-28.** `Alibaba-NLP/gte-multilingual-base`
runs locally via `sentence-transformers` — no API, no per-token charge — so
[M3-02] introduces **no price constant**, and per [M1-09]'s stale-pricing lesson
there is nothing to date-stamp beyond this sentence. The cost that answers "what
does it take to reproduce the index" is machine time.

Cold run (empty `data/cache/embeddings/`, `embed_chunks.py --device cpu`, single
process, `EMBEDDING_BATCH_SIZE=64`), figures from
`eval/runs/embedding_cost_report.{md,json}`:

| measurement | value |
| --- | --- |
| chunks | 4,540 (71 batches of 64) |
| total passage tokens | 1,240,900 (`format_passage`, special tokens included; max 1,015 in one chunk) |
| wall-clock, cold | **41.2 min** (2,472 s) — `sentence-transformers` 3.4.1 / `torch` 2.13.0, CPU |
| throughput | ~500 passage tokens/s |
| forward passes | 4,524 (16 chunks share an `embedded_text` with an earlier chunk and hit the in-run cache) |
| one-time model download | ~1.2 GB weights + the `trust_remote_code` files, network-bound, not counted above |
| warm re-run | 0 forward passes — every vector served from `cache.jsonl` |
| **dollar cost** | **$0.00** (local model) |

On the RTX 3050 Laptop GPU the same pass is a few minutes; CPU is the
conservative headline since a reproducer is not assumed to have a GPU. The
report file also records the resolved library / pgvector / Postgres versions.

## The embedding cache

`infrastructure.rag.embedding_cache.CachingEmbedder` wraps any `Embedder` with a
content-addressed, on-disk cache, so re-running the pipeline over an unchanged
corpus costs nothing. It mirrors the two caches the [M3-02] DoD points at
(`CachingClauseClassifier`, `CachingBoundaryVisionReviewer`): an in-memory dict
loaded from a JSON Lines file, appended on every miss.

### The key covers the whole contract, not just the text

Each entry is keyed by `sha256(fingerprint · "\x00" · text)`, where `fingerprint`
is `embedding_config.config_fingerprint()` — a digest of the model id, revision,
dimensionality, distance metric, normalisation flag and both prefixes. The DoD
calls an incomplete key "the most expensive failure mode available in this
issue": a cache built under one model configuration returning its vectors to a
different one, with no error at all. Folding every contract value into the key
makes that impossible — swap the model, bump the revision, change the
normalisation or add a prefix and every text is a fresh miss.
`tests/unit/infrastructure/rag/test_embedding_config.py` asserts each of those
fields moves the fingerprint; `test_embedding_cache.py` asserts the cache
re-embeds when it does.

### Location and format: the convention, not a departure

`data/cache/embeddings/cache.jsonl`, one `{"key": ..., "vector": [...]}` per
line — the same location and format as
`data/cache/llm_classification/cache.jsonl` and
`data/cache/boundary_escalation/cache.jsonl`. The DoD allows departing "if this
cache's volume justifies" it; it does not, quite:

- Every `data/cache/*` subdirectory is gitignored (each has its own
  `.gitignore` line), so the file never enters version control whatever its
  size.
- The full 4,540-chunk corpus is 74 MB on disk (measured) and loads once per run
  into a 4,540-entry dict (~110 MB RAM). That is well within the existing
  cache footprint — `data/cache/boundary_escalation_pages/` is 257 MB — and it
  is a batch-script cost, not a service cost.
- One format across every cache (greppable, inspectable, no new dependency) is
  worth more than the disk saving here. A ~10× larger corpus would tip this
  toward a packed binary format; at this scale it does not.

### Wiring

`CachingEmbedder` is applied at composition time, not inside the pipeline:
`scripts/embed_chunks.py` passes `CachingEmbedder(SentenceTransformerEmbedder(...))`
to `embed_missing_chunks`, exactly as `scripts/build_corpus.py` wraps
`LangchainClauseClassifier` in `CachingClauseClassifier`. On a database reset
(`make migrate` from scratch, then `make load-chunks`) every vector is `NULL`
again, but the on-disk cache still holds them — so a re-`embed-chunks` refills
`chunk.embedding` with **zero** model forward passes (measured: 2.6 s from the
74 MB `cache.jsonl`, against 41 min cold).

### What it does *not* fix

A chunk whose `embedded_text` changed between runs keeps its stale vector: the
pipeline's cursor is `WHERE embedding IS NULL`, and `upsert_chunks` never
touches `embedding`, so that row is never handed to the embedder — cache or no
cache. Closing that gap needs a pipeline/schema change (a stored content hash,
or nulling the vector when the text changes), and it stays deferred.

## [M1-09] per-constant decisions for [M3-02]

Every constant this issue introduces, classified per [M1-09]'s rule: a value
stays a **code constant** when it changes the vectors or a published retrieval
number (it is experimental design, like `SEED` / `SAMPLE_SIZE` in
`scripts/sample_parsing_quality.py`); it moves to **`.env`** only when it is a
pure operational lever with no effect on any published number. Operational test:
anything folded into `config_fingerprint()` is design and stays in code.

| constant | location | decision | why |
| --- | --- | --- | --- |
| `EMBEDDING_MODEL_ID`, `EMBEDDING_MODEL_REVISION`, `EMBEDDING_DIMENSIONS`, `DISTANCE_METRIC`, `NORMALIZE_EMBEDDINGS`, `QUERY_PREFIX` / `PASSAGE_PREFIX` | `embedding_config.py` | code constant | in the fingerprint; determine the exact vectors |
| `EMBEDDING_MAX_INPUT_TOKENS` | `embedding_config.py` | code constant | model capability fact; drives the input-length check |
| `EMBEDDING_TRUST_REMOTE_CODE` | `embedding_config.py` | code constant | security-sensitive, bound to the revision — must not be `.env`-flippable |
| **`EMBEDDING_BATCH_SIZE`** | was `embedding_pipeline.py`; now **`.env`** via `EmbeddingSettings` (default 64) | **`.env` knob** | pure throughput / memory lever, excluded from the fingerprint; the direct analog of `LLM_CLASSIFICATION_MAX_WORKERS` |
| embedding worker count | — | no knob | the pipeline is single-threaded in-process; `sentence-transformers` owns its own intra-op parallelism. The only levers are `EMBEDDING_BATCH_SIZE` and the device (`SentenceTransformerEmbedder(device=...)`, auto cuda/cpu) |
| `EMBEDDING_RETRY_MAX_ATTEMPTS` / `_DELAY_SECONDS` | alias of `llm_retry_defaults` | code constant (shared) | repo-wide shared policy; an in-process model has no rate limit to tune against |
| `EMBEDDING_CACHE_PATH` | `embedding_cache.py` | code constant | path convention, gitignored, not environment-varying |
| `HNSW_M`, `HNSW_EF_CONSTRUCTION` | `ann_index.py` | code constant | change the index and therefore any published Recall@k; [M3-08]'s to tune |
| `HNSW_EF_SEARCH`, `HNSW_ITERATIVE_SCAN` | `ann_index.py` | code constant | change what a filtered search *returns*; `strict_order` is a correctness default that must not be weakened from `.env` |
| `INDEX_NAME` | `ann_index.py` | code constant | an identifier, not a tunable |

Net: exactly one constant moved (`EMBEDDING_BATCH_SIZE`). `.env` and `.env.example`
carry that one new key in exact parity (a `skipif`-guarded parity test in
`tests/unit/infrastructure/config/test_settings.py` checks it locally; CI has no
`.env`).

## The HNSW ANN index over `chunk.embedding`

`infrastructure.rag.ann_index` holds the index definition — `CREATE INDEX ...
USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 64)` —
plus `HNSW_EF_SEARCH = 40` and the `hnsw.iterative_scan` default, and the
`create_hnsw_index` / `drop_hnsw_index` / `apply_ann_search_gucs` helpers.

**It is deliberately not an Alembic migration.** An ANN index is a
retrieval-tuning artifact, not schema: exact `<=>` cosine search works without
it, its `m` / `ef_construction` are [M3-08]'s to tune, and the autouse
`migrated_database` fixture would rebuild it on every integration test. The
measurement below shows it does not earn a place in the default schema at this
corpus size. [M3-08]'s `make build-index` calls `create_hnsw_index` *iff* the
verdict here says to; [M3-04]'s retriever calls `apply_ann_search_gucs` on any
query that does route through the index.

`make benchmark-ann-index` (`scripts/benchmark_ann_index.py`) builds the index
against `TEST_DATABASE_URL` and measures it. `tests/integration/test_ann_index.py`
is the committed proof of the filtered-search behaviour below.

## Does the ANN index earn its place at 4,540 chunks? (2026-08-28, synthetic vectors)

**What synthetic vectors establish and do not.** `make benchmark-ann-index`
fills `chunk.embedding` with deterministic pseudo-random unit vectors rather than
running the real embedder (which now exists — `make embed-chunks`), so the
benchmark stays a one-command, DB-only run. Build time, index size and *latency*
are structural — they depend on the row count, dimensionality, `halfvec`
storage and the index parameters, not on what the vectors mean — so they
transfer to real embeddings. ANN **recall vs. exact does not** transfer (random
vectors have no cluster structure, which is close to the worst case for HNSW);
the real recall number is [M3-08]'s to measure on real embeddings, and the
verdict here rests on latency and cost, not recall.

**Pre-registered prediction** (written in the plan before the benchmark ran,
left unedited): exact full-corpus scan low-single-digit ms; single-partition
sub-millisecond; HNSW build < 0.5 s; index ~8–16 MB; verdict — exact search is
competitive, HNSW does not earn its place at this scale.

**Outcome** (`eval/runs/ann_index_benchmark.{md,json}`, regenerable; figures
below from the 2026-08-28 run, stable to ~5% across re-runs):

| measurement | value |
| --- | --- |
| `CREATE INDEX` wall time | ~0.8 s *(predicted < 0.5 s — missed, still trivial)* |
| index size | ~9 MB (0.69× the table) *(predicted 8–16 MB — held)* |
| exact `<=>`, full corpus, no index | p50 **~11.4 ms**, p95 ~12.4 ms (seq scan) *(predicted low-single-digit — missed)* |
| exact `<=>`, single partition, no index | p50 **~0.6 ms** (btree bitmap scan) *(predicted sub-ms — held)* |
| HNSW, full corpus | p50 **~0.8 ms** (index scan) |
| HNSW available, single partition | p50 **~0.6 ms** — planner still chose the **btree bitmap scan**, not the HNSW index |
| plain HNSW recall@10 vs. exact | ~0.45 — synthetic, does **not** transfer; [M3-08] measures the real one |

50 query vectors × 10 iterations; end-to-end from the Python client on a shared
host, so read the millisecond figures as directional, not a tuned benchmark.

**Verdict: the HNSW index does not earn its place at 4,540 chunks.** [M3-04]'s
default retrieval path filters by SUSEP process + CNPJ — one document, 19–726
chunks — and there the planner reads the partition by btree and sorts it
exactly in ~0.6 ms; it does not touch the HNSW index even when the index
exists. The index only changes the *unfiltered* full-corpus number (11.4 → 0.8
ms), which is not the default path and is already far inside any budget the
downstream LLM call dominates. Against that: 9 MB, a 0.8 s build, and a recall
penalty that has to be measured and tuned. The definition stays in
`infrastructure.rag.ann_index` so [M3-08] can A/B it on real embeddings and so it
is one call away when the corpus grows (~10× is where exact scan starts to
hurt); it does not go into the schema now.

## Filtered search and the fewer-than-`k` question

Whether a metadata-filtered vector search can return fewer than `k` rows, and
how that is handled — settled here so [M3-04] does not meet it as an
unexplained recall regression. Measured counts (`k = 10`, smallest partition =
19 chunks):

| filter | exact | planner default | HNSW forced, no iter. scan | HNSW forced, `strict_order` |
| --- | ---: | ---: | ---: | ---: |
| SUSEP process + CNPJ (19 chunks) | 10 | 10 (btree) | **0** (HNSW) | 10 |
| + `clause_type = 'exclusion'` (2 chunks) | 2 | 2 (btree) | 0 (HNSW) | 2 |

1. **Exact search + the default filter cannot return fewer than `k`.** It
   returns exactly `min(k, |partition|)`. Every SUSEP-process partition in the
   corpus has ≥ 19 chunks, so the default path always fills `k`.
2. **Exact search + a *stacked* filter can return fewer than `k`** — the
   `clause_type = 'exclusion'` row above returns 2 because the partition holds
   only 2 such chunks. That is a true insufficient-context signal, not an
   artifact: the `Retriever` contract already returns "up to `k`", the shortfall
   is surfaced rather than padded, and [M3-07]'s gate is the downstream handler.
3. **An HNSW index scan under any selective filter can silently return fewer
   than `k`** — even 0 — because pgvector filters the `hnsw.ef_search` (40)
   candidate list *after* the index scan, and a filter admitting a few percent
   of rows discards all of them. The "HNSW forced" column (btree filter indexes
   dropped, seq scan disabled) shows 0 rows for a partition that has 19.
4. **The planner does not choose the HNSW scan for a filtered query at this
   scale** — with the metadata btree indexes present it always reads the
   partition by btree and sorts exactly (the "planner default" column). So case
   3 is latent, not active, on the default path today — but it must not be left
   to the planner's cost model.
5. **Mitigation: `SET LOCAL hnsw.iterative_scan = strict_order`**
   (`apply_ann_search_gucs`, pgvector ≥ 0.8.0, pinned 0.8.6). The index keeps
   scanning past `ef_search` until `k` post-filter matches are found, in exact
   distance order — the "HNSW forced, `strict_order`" column returns the full
   10 (and the honest 2 for the genuinely-2-row filter). `hnsw.max_scan_tuples`
   (20000) exceeds the whole corpus, so at this scale an iterative scan
   degrades gracefully to a full filtered scan. Residual: it can still
   under-return if `max_scan_tuples` is exhausted first — unreachable here, a
   knob for a ~10×-larger corpus.

**Decision for [M3-04] / [M3-08]:** the default filtered retrieval path uses
exact `<=>` over the pre-filtered partition (sub-millisecond, always returns
`k` when `k` rows exist). If [M3-08] benchmarks an HNSW configuration, it MUST
set `hnsw.iterative_scan` for every filtered run, or the recall number is an
index artifact rather than a property of the retriever. See
`docs/DATABASE.md`'s "Minimum pgvector version" section for the version gate.

## Still deferred in [M3-02]

Two items remain open:

- **Stale vector on a changed `embedded_text`** ("What it does *not* fix" above)
  — a chunk whose text changes under a stable `chunk_id` keeps its old vector.
  Needs a stored content hash or a null-on-change step.
- **The ANN-earns-its-place check on *real* embeddings** — the verdict above
  rests on latency and synthetic vectors. [M3-08]'s benchmark matrix re-runs it
  on the real embeddings and settles whether `make build-index` should call
  `create_hnsw_index`.
