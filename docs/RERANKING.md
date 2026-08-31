# Cross-encoder reranking

The [M3-05] reranking stage: a model that reads the query and each candidate
clause *together* re-scores the fused hybrid candidate set, so the answer clause
is lifted toward rank 1. The contract lives in code at
`app/src/infrastructure/rag/reranker_config.py`; this document is the rationale,
the evidence, and the numbers.

**Scope.** This issue builds the reranker, wires it behind the existing
`retrieve(question, *, k, metadata_filter)` interface, and tunes its **candidate
depth** on `golden-set-v1` — with the reranker sitting on the [M3-04] hybrid RRF
+ SUSEP-process + CNPJ config, which `docs/HYBRID_RETRIEVAL.md` already fixed as
the base. It does **not**:

- produce the committed lexical-only / dense-only / hybrid / hybrid+rerank
  benchmark matrix or `make build-index` — that is [M3-08]'s, which also
  re-opens the RRF-vs-weighted fusion call with the reranker in the loop;
- re-tune BM25 `k1`/`b` or the RRF constant — those stay the pinned
  `docs/LEXICAL_RETRIEVAL.md` / `docs/HYBRID_RETRIEVAL.md` values ([M3-08]'s);
- add exclusion co-retrieval ([M3-06]) or the insufficient-context gate
  ([M3-07]).

---

## Method

### The pipeline

`RerankingRetriever` (`reranking_retriever.py`) wraps the hybrid retriever behind
the **same** `retrieve(question, *, k, metadata_filter=None) -> list[str]` M4's
graph node will call:

1. ask the base retriever (hybrid RRF, `--filter default`) for
   `RERANK_CANDIDATE_DEPTH` clause ids over the question's SUSEP process + CNPJ;
2. look up each candidate's passage text;
3. score every `(question, passage)` pair with the cross-encoder;
4. return the `k` highest-scoring clause ids. Equal scores keep the base
   retriever's (fused) order — a stable sort — so the output is deterministic.

The reranker can only **reorder** the candidate set: it never introduces a
clause the base retriever did not return, and it never pads. This is the reason
`docs/HYBRID_RETRIEVAL.md` kept RRF (higher Recall@10) over weighted fusion
(higher MRR): "the reranker fixes ordering, it cannot recover a clause that was
never retrieved", so feeding it the higher-recall candidate set is the better
pipeline. (The Outcome below shows the reranker lifts RRF's MRR from 78.8 to
80.6 — real, but short of the 83.1 weighted fusion had. [M3-08] re-runs the
RRF-vs-weighted call with the reranker in the loop; M3-05 fixes the depth on the
`docs/HYBRID_RETRIEVAL.md` base.)

### The model — `Alibaba-NLP/gte-multilingual-reranker-base`

The reranking half of the same mGTE work as the pinned embedder
(`Alibaba-NLP/gte-multilingual-base`). The paper `docs/EMBEDDINGS.md` already
cites — *"mGTE: Generalized Long-Context Text Representation and Reranking
Models for Multilingual Text Retrieval"* (arXiv 2407.19669) — **is this model's
paper**; Portuguese is one of its explicit target languages, evaluated in MLDR,
not incidental.

| property | value | source |
| --- | --- | --- |
| parameters | ~306M | model card |
| context window | 8192 tokens | `config.json` `max_position_embeddings` |
| output | single relevance score (`num_labels = 1`) | `config.json` |
| architecture | `NewForSequenceClassification`, `trust_remote_code` via `Alibaba-NLP/new-impl` | `config.json` `auto_map` — the **same** custom code path as the embedder |
| revision pinned | `8215cf04918ba6f7b6a62bb44238ce2953d8831c` (2025-07-05) | `reranker_config.RERANKER_MODEL_REVISION` |

**Why this over `BAAI/bge-reranker-v2-m3`** (the other credible multilingual
option): one family, one paper, one evidence story with the embedder; and the
`embed` group's `transformers>=4.39,<4.46` / `sentence-transformers>=3.0,<3.5`
pins **already exist** for exactly this `Alibaba-NLP/new-impl` RoPE code path
(`config.json` declares `transformers_version 4.39.1`, same as the embedder), so
the reranker adds **no dependency and no new pin**. `sentence-transformers`
3.4.1 already ships `CrossEncoder`.

The local placeholder `RERANKER_MODEL=BAAI/bge-reranker-base` (which predated
this issue) was **not** viable: `bge-reranker-base` is English/Chinese only.

### Passage representation

The reranker scores `(question, passage)` where `passage` is built from the
**chunk corpus** (`build/chunks.jsonl`): for each candidate clause id, the
breadcrumb-prefixed `ChunkRecord.text` of its chunks, rejoined in `chunk_index`
order (`eval_retrieval.build_clause_text_map`). This is the same field BM25
indexes and the embedder embeds, so all three legs judge the candidate on the
same representation — the reasoning `docs/LEXICAL_RETRIEVAL.md` gives for
indexing `text` not `display_text` (the exact term is often only in an ancestor
heading).

### Caching and device

`CachingReranker` (`reranker_cache.py`) mirrors `CachingEmbedder`: a gitignored
`data/cache/reranker/cache.jsonl`, keyed by
`sha256(config_fingerprint · query · passage)`, so a re-run of
`make eval-retrieval-rerank` over the 117 golden queries costs nothing. The
fingerprint folds the model id, revision, input cap and candidate depth, so a
cache built under one configuration can never serve another.

**Device split.** The cross-encoder's scores are fp32 and device-independent, so
every Recall / MRR / nDCG / exclusion number is identical on CPU and GPU. The
*scoring* pass therefore runs on the dev GPU when one is present (`_load_reranker`
defaults to auto; `_load_query_embedder` is lazy and every golden query vector is
already cached, so the card is free) — ~10× faster than CPU. The **latency**
number the trade-off below turns on is still a **CPU** number by design: there is
no GPU serving infra, a reranker in an interactive path is the realistic
deployment, and CPU is reproducible on any machine. `make tune-reranking`
measures it on a separate `device="cpu"` reranker, as a probe over
`LATENCY_PROBE_QUESTIONS` questions at each candidate depth.

### The `[M1-09]` per-constant decision

Every value this issue introduces changes the reranked order (and so the
published MRR / Recall@10), so per [M1-09]'s rule they are **all module
constants** in `reranker_config.py`, folded into `config_fingerprint()`.
**Zero new `.env` keys** — the analog of `docs/LEXICAL_RETRIEVAL.md`'s "zero"
and `docs/HYBRID_RETRIEVAL.md`'s "zero".

| constant | value | why a code constant |
| --- | --- | --- |
| `RERANKER_MODEL_ID` | `Alibaba-NLP/gte-multilingual-reranker-base` | determines the scores; `RERANKER_MODEL` in `.env` is the human-facing name, cross-checked against this by a test — the `EMBEDDING_MODEL` arrangement |
| `RERANKER_MODEL_REVISION` | pinned Hub commit | a re-upload must not change a published number |
| `RERANKER_MAX_INPUT_TOKENS` | 8192 | truncation length changes what the model sees for a long clause |
| `RERANKER_TRUST_REMOTE_CODE` | `True` | security-sensitive — never `.env`-flippable, per `EMBEDDING_TRUST_REMOTE_CODE` |
| `RERANK_CANDIDATE_DEPTH` | `10` (set from the sweep below) | changes the reranked ranking the golden-set numbers are measured on |
| `RERANK_BATCH_SIZE` (`cross_encoder_reranker.py`) | 1 | pure throughput / memory lever, no effect on scores (padding is attention-masked) — **not** in the fingerprint (the `EMBEDDING_BATCH_SIZE` analog, minus the `.env` promotion since no operator has asked for it). Held at 1: a batch pads to its longest member, and one 8192-token clause already fills the activation budget of the 4 GB dev GPU the scoring pass uses (batch 4 CUDA-OOMs; batch 64 needs >12 GB, past the 14 GB CPU box too). Per-pair cost is the model forward, so 1 barely changes throughput (~45 ms/pair on the GPU) and bounds the peak on any hardware. The reported CPU latency is therefore a batch-1 number — the conservative direction; a batched interactive reranker would do better. M4 picks its own value for its serving hardware. |

### The harness

`scripts/eval_retrieval.py` gains a `--rerank` flag (rejected only with
`--retriever random`); `RetrievalRunConfig` is bumped to `v4` with the reranker
model id/revision, candidate depth and config fingerprint (all optional).
`make eval-retrieval-rerank` runs `--retriever hybrid --filter default --rerank`
and writes `eval/runs/retrieval_eval_hybrid_rrf_rerank_filter-default.{md,json}`.
`make tune-reranking` (`scripts/tune_reranking.py`) sweeps the candidate depth
and writes `eval/runs/rerank_tuning.{md,json}`. Both run under
`uv run --group embed` and need Postgres with loaded + embedded chunks.

---

## Pre-registered prediction

Written in the implementation plan **before** the reranker was run, reproduced
here unedited:

> - Reranking lifts MRR and R@1 on the filtered hybrid config (the top-rank
>   precision weighted fusion bought: MRR toward ~83%+, R@1 toward ~68%+) and
>   nDCG@10 with them.
> - Recall@10 moves little at shallow candidate depth (k≈10–20: pure reorder)
>   and can gain a point or two at deeper k as the cross-encoder rescues a
>   relevant clause the fusion ranked 11–30; it does not drop below the 91.5%
>   baseline at the chosen k.
> - Exclusion-clause recall does not fall below 92.6% (25/27) at the chosen
>   `(k, n)` — the DoD's named regression does not occur; `coverage_with_exclusion`
>   stays the weakest question type (exclusion co-retrieval is [M3-06]).
> - The curve has a knee: metrics rise from k=10 to ~k=30–50 then flatten while
>   latency keeps climbing roughly linearly in k. Chosen k at the knee.
> - Added latency is tens to low-hundreds of ms/query on CPU at the chosen k —
>   material for a future interactive path, negligible for the batch eval.

---

## Outcome (2026-08-29)

`make tune-reranking` (the candidate-depth sweep and the CPU latency probe) and
`make eval-retrieval-rerank` (the full breakdown at the chosen depth) →
`eval/runs/rerank_tuning.{md,json}` and
`eval/runs/retrieval_eval_hybrid_rrf_rerank_filter-default.{md,json}`
(regenerable; gitignored). Reranker `Alibaba-NLP/gte-multilingual-reranker-base`
@ `8215cf04…` (config fingerprint `777c0503f1073d52`) on the [M3-04] hybrid RRF +
SUSEP-process + CNPJ base (`279ed8ee0a668227`). 117 scorable questions (the 23
`unanswerable` are excluded — empty reference set); all CASCO / CARTA VERDE, all
`text` extraction. Scoring ran on an RTX 3050 (the scores are device-independent);
the latency column is a CPU probe on a Ryzen 5 5600H.

### Curve — candidate depth sweep

The `n` dimension (final context cut) is the Recall@1 / @5 / @10 columns:
n = 1 / 5 / 10.

| candidate depth | R@1 | R@5 | R@10 | MRR | nDCG@10 | exclusion recall | CPU rerank ms/query (p50 / mean) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid RRF (no rerank) | 64.5% | 83.6% | 91.5% | 78.8% | 80.3% | 92.6% (25/27) | — |
| **10** ← chosen | 65.8% | 86.9% | 91.5% | 80.6% | 82.0% | 92.6% (25/27) | 9 750 / 11 000 |
| 20 | 64.1% | 86.1% | 92.3% | 79.1% | 81.1% | 92.6% (25/27) | 19 200 / 22 500 |
| 30 | 64.1% | 85.7% | 91.6% | 78.8% | 80.5% | 85.2% (23/27) | 29 000 / 34 100 |
| 50 | 63.2% | 84.9% | 90.7% | 78.2% | 79.8% | 85.2% (23/27) | 38 900 / 47 100 |

The curve has **no rising limb** — every metric peaks at depth 10 and declines
monotonically after. Depth 10 is a pure reorder of the hybrid top-10 (R@10 cannot
move, and doesn't). Depth 20 is the only depth that lifts R@10, by 0.8 pt (one
question), at the cost of 1.5 pt MRR. From depth 30 the cross-encoder starts
promoting deep, plausible-but-wrong candidates: R@10 falls **below** the no-rerank
baseline (90.7% at depth 50) and **exclusion-clause recall drops to 85.2%
(23/27)** — two exclusion clauses pushed out of the top-10, the DoD's named
regression.

### At the chosen depth (10) — by question type

(→ is no-rerank → +rerank at depth 10, both on the `--filter default` base.)

| question_type | n | R@1 | R@5 | R@10 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_lookup` | 64 | 69.5 → **74.2** | 89.5 → **93.2** | 95.3 → 95.3 | 80.7 → **85.1** | 83.9 → **87.2** |
| `definition` | 18 | 77.8 → 77.8 | 83.3 → 83.3 | 88.9 → 88.9 | 80.4 → **81.5** | 82.4 → **83.3** |
| `cross_document` | 16 | 75.0 → **68.8** | 87.5 → **93.8** | 93.8 → 93.8 | 82.1 → **80.2** | 85.0 → **83.7** |
| `coverage_with_exclusion` | 19 | 26.3 → **23.7** | 60.5 → **63.2** | 78.9 → 78.9 | 68.2 → **64.8** | 62.1 → 61.8 |
| overall | 117 | 64.5 → **65.8** | 83.6 → **86.9** | 91.5 → 91.5 | 78.8 → **80.6** | 80.3 → **82.0** |

The lift is concentrated in `direct_lookup` (55% of the scored set): +4.7 pt R@1,
+3.7 pt R@5, +4.4 pt MRR — exactly where reading the query and the clause together
disambiguates an exact-term match. `definition` gains slightly. `cross_document`
and `coverage_with_exclusion` give back a little top-rank precision: the
cross-encoder scores a fluent coverage clause above the terse exclusion that
limits it, and can demote the right clause on a question that spans two
documents. But **exclusion-*clause* recall** — the DoD metric, "is the exclusion
retrieved in the top-10 at all" — is **unchanged at 92.6% (25/27)**: the same two
clauses miss as in the baseline, none new.

### Prediction vs actual

| prediction | actual | call |
| --- | --- | --- |
| Reranking lifts MRR and R@1 (MRR → ~83%+, R@1 → ~68%+) and nDCG@10 | MRR 78.8 → 80.6, R@1 64.5 → 65.8, nDCG 80.3 → 82.0 — direction right, ~half the predicted magnitude | **partly missed** — the lift is real but does not recover the MRR 83.1 weighted fusion bought |
| R@10 moves little at shallow depth, gains a point or two at deeper k, does not drop below 91.5% at the chosen k | depth 10: R@10 = 91.5% exactly (pure reorder); at deeper k it *drops* (90.7% at depth 50), it does not gain | **missed** — the "gains at deeper k" half is backwards; "not below baseline at the chosen k" held (chose depth 10) |
| Exclusion-clause recall ≥ 92.6% at the chosen (k,n); the regression does not occur; `coverage_with_exclusion` stays weakest | depth 10: 92.6% (25/27), unchanged — regression does not occur. It *does* occur at depth ≥ 30 (85.2%). `coverage_with_exclusion` stays weakest (78.9% R@10) | **held** at the chosen depth — and the sweep shows where it breaks |
| Curve has a knee: metrics rise from k=10 to ~k=30–50 then flatten; chosen k at the knee | metrics do not rise — they peak at k=10 and fall after | **missed** — chose k=10, for the opposite reason to the one predicted |
| Added latency is tens to low-hundreds of ms/query on CPU at the chosen k | ~9 750 ms/query (p50) at depth 10 on the Ryzen 5 5600H | **missed, badly** — ~50–100× the prediction; see Latency trade-off |

---

## Latency trade-off

At the chosen depth 10, batch 1, `RERANKER_MAX_INPUT_TOKENS` = 8192:

> **+~9 750 ms/query (p50; mean ~11 000, p95 ~34 000) on CPU (Ryzen 5 5600H)
> buys +1.8 pt MRR / +1.3 pt R@1 / +3.3 pt R@5 / +1.7 pt nDCG@10 / ±0 pt R@10 /
> ±0 pt exclusion-clause recall.**

The **metric** side of the trade is worth taking: a ~2 pt MRR / ~3 pt R@5 lift,
and at eval time it is nearly free — the scoring pass runs on the RTX 3050 at
~45 ms/pair (~2 min for the whole golden set), and the
`data/cache/reranker/cache.jsonl` (keyed by `config_fingerprint · query ·
passage`) makes every re-run after the first a dict lookup.

The **latency** side rules the CPU cross-encoder out as a live component: ~10 s
added to every claim (p50; ~34 s at p95, climbing to ~39 s p50 at depth 50) is
not an interactive path. Per `docs/EMBEDDINGS.md`'s "does it earn its place"
framing:

- **Batch evaluation** (the [M3-08] matrix, golden-set re-measurement): the
  reranker earns its place — GPU scoring, cached re-runs.
- **M4's live graph / M5's API**: the CPU reranker as written does **not**. The
  levers, all M4's to pull, sit behind the one `RerankingRetriever` interface:
  run the ~306 M cross-encoder on a GPU (the RTX 3050's 4 GB holds it alone — the
  eval already does this for scoring), drop `RERANK_CANDIDATE_DEPTH` further, make
  reranking optional, or swap a smaller / quantized / API reranker. `_load_reranker`
  pins CPU for the **eval harness only**; its docstring says M4 picks its device.

---

## Verdict

**Keep reranking, at `RERANK_CANDIDATE_DEPTH = 10`.** Reordering the hybrid RRF
top-10 with the cross-encoder lifts overall MRR 78.8 → 80.6, R@1 64.5 → 65.8, R@5
83.6 → 86.9 and nDCG@10 80.3 → 82.0, with R@10 unchanged at 91.5% and
exclusion-clause recall unchanged at 92.6% (25/27). Both M3 exit values
(`MILESTONES.md`: R@10 ≥ 0.85, MRR ≥ 0.60) still clear their bars. The gain is
concentrated in `direct_lookup`; `cross_document` and `coverage_with_exclusion`
give back a little top-rank precision but their R@10 and the exclusion-clause
metric are untouched.

**DoD item 4 — reranking must not push exclusion clauses out of the final
context.** At depth 10: confirmed — exclusion-clause recall 92.6% (25/27) before
and after, the same two misses. The sweep shows the regression is real at
depth ≥ 30 (85.2%, 23/27): the cross-encoder ranks a fluent coverage clause above
the terse exclusion. That is a second, independent reason the chosen depth is
shallow.

**Why not depth 20** (the only depth that lifts R@10, by 0.8 pt): 0.8 pt is one
question, and it costs 1.5 pt MRR and 1.7 pt R@1. Reranking's job is top-rank
precision; depth 10 delivers it with no R@10 cost. [M3-08] re-opens the depth
(and the RRF-vs-weighted call) with the full matrix and the reranker in the loop.

**Deviation from the pre-registered prediction, stated per `MILESTONES.md`.**
Three of the five bullets missed: the metric lift is real but ~half the predicted
magnitude; the curve has no rising limb (metrics peak at the shallowest depth,
not at k ≈ 30–50), so "chose k at the knee" holds only trivially; and the CPU
latency is ~50–100× the "tens to low-hundreds of ms" predicted, which makes the
CPU cross-encoder a batch-eval / GPU-serving tool, not a live-path component. The
predictions are left unedited above; the curve and the by-type table are the
evidence. A dated `[M3-05]` note is added to `MILESTONES.md`'s M3 section.

---

## Limitations

- **Product-line coverage stays 2 of 5.** The 117 scorable questions reference
  only CASCO (114) and CARTA VERDE (3); RCF-A, ASSIST and GAR.EST appear only
  among the excluded `unanswerable` questions — the same `golden-set-v1` limit
  `docs/LEXICAL_RETRIEVAL.md`, `docs/HYBRID_RETRIEVAL.md` and
  `docs/EVALUATION.md` record.
- **The `n` (final context) dimension is tuned against the top-10 harness.** The
  Recall@1/@5/@10 columns give the n = 1/5/10 view, but M4's real context budget
  — how many reranked clauses actually reach the assessment LLM — is M4's call
  with its own constraints; `RERANK_CANDIDATE_DEPTH` is the knob M3-05 fixes.
- **The dense leg reads the dev database** and the models load locally; not
  hermetic the way the file-based lexical eval is. The config fingerprints pin
  the contract.
- **Latency is a CPU, batch-1 number** on one machine (Ryzen 5 5600H). It is the
  conservative number for the interactive-path question — a batched or GPU
  reranker does better — not a throughput ceiling. The metrics are
  device-independent.

---

## Deferred / handed to later issues

- **The lexical / dense / hybrid / hybrid+rerank benchmark matrix and
  `make build-index`** — done: [M3-08], `docs/RETRIEVAL_BENCHMARK.md`. It also
  re-opened the RRF-vs-weighted fusion call with the reranker in the loop (RRF
  wins every metric — 92.3 vs 89.3 Recall@10) and confirmed at depth 10 the
  reranked+co-retrieval best config clears both M3 bars (Recall@10 92.3%, MRR
  80.6%).
- **Exclusion co-retrieval** (pull the exclusion linked to every retrieved
  coverage clause) — [M3-06]. `coverage_with_exclusion` is the weakest question
  type and reranking does not fix it — it can only reorder what retrieval found.
- **The insufficient-context gate** on the retrieval signals — [M3-07] (done:
  `docs/INSUFFICIENT_CONTEXT_GATE.md`; the gate reads the rank-1 reranked score
  this stage produces).
- **BM25 `k1`/`b` and RRF `k` tuning with the reranker in the loop** — not
  re-run by [M3-08]: the M3 bars are cleared and each value was tuned on this
  same golden set. The `make tune-*` targets remain for a changed corpus or
  golden-set-v2.
