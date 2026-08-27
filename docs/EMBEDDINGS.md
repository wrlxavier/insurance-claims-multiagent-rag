# Embeddings

The dense-retrieval embedding model for [M3-02], its version pin, and the
distance / normalisation / prefix decisions that [M3-04] and the deferred
embedding-pipeline PR both depend on. The contract lives in code at
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

- **Index side** (deferred embedding-pipeline PR): the `embedding
  halfvec(768)` column's ANN index takes `halfvec_cosine_ops`, and any
  ordering query uses the `<=>` (cosine distance) operator.
- **Query side** ([M3-04]): the retriever orders by the same `<=>`.

`halfvec` (not `vector`) for storage follows [M0-08]'s decision in
`docs/DATABASE.md`; at 768 dimensions the column is within the plain-`vector`
index limit too, but half-precision storage is the project default.

## Deferred to the embedding-pipeline PR

Still open in [M3-02] after this change: the `embedding halfvec(768)` column
and its migration, `uv add pgvector`, batched embedding with retry and a
resumable cursor, the content-hash embedding cache, the ANN index plus its
build-time/size record, the ANN-vs-exact-search measurement, the
filtered-search-returns-fewer-than-k question, the corpus embedding cost
report, and the `make` target composable into [M3-08]'s `make build-index`.
