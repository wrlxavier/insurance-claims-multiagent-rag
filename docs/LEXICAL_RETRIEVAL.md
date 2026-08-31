# Lexical retrieval

The BM25 lexical retriever for [M3-03], its Portuguese analyzer, its
stemming-exception list, and the standalone measurement of what it does on
`golden-set-v1`. The contract lives in code at
`app/src/infrastructure/rag/lexical_config.py`; this document is the rationale,
the evidence, and the numbers.

**Scope.** This issue *builds* the lexical leg and measures it **alone, with no
metadata filter**. It does **not** compare lexical against dense — the
lexical-only / dense-only / hybrid / hybrid+rerank matrix is [M3-08]'s, run on
`golden-set-v1`. "Which question types lexical retrieval wins outright" (the
M3-03 DoD) is answered below from lexical's own by-`question_type` profile plus
a pre-registered prediction; the cross-retriever comparison is out of scope
here, the same way [M3-02] deferred the real-embedding ANN A/B to [M3-08].

The metadata pre-filter (SUSEP process + insurer CNPJ) is [M3-04]'s. Its
absence is the single biggest factor in the numbers below, and a
one-off simulation (["What the [M3-04] filter recovers"](#what-the-m3-04-filter-recovers))
quantifies exactly how much.

---

## Method

### The retriever

Hand-rolled Okapi BM25 over the **chunk corpus** (`build/chunks.jsonl`, 4,540
chunks — [M3-01]'s output), not the raw clause corpus. "Clause corpus" in the
DoD is read as "the clause content, as chunked": most clauses map 1:1 onto a
chunk, and `domain/chunk.py` records `source_clause_ids` (anchor + every short
clause merged into the chunk) precisely so a chunk-level hit rolls back up to
clause-id granularity for the [M2-06] `Retriever` contract. Indexing chunks
keeps the lexical and dense sides on the same unit for [M3-04]'s rank fusion.

Every chunk hit is rolled up to its `source_clause_ids`, de-duplicated keeping
the best (lowest) rank, and the top *k* clause ids are returned. Verified
against the artifacts: all 140 `golden-set-v1` reference clause ids are
reachable this way (4 of them are merged, non-anchor clauses, so the roll-up
**must** expand `source_clause_ids`, not just `clause_id`); the 123 zero-content
clauses dropped in chunking are not golden references, so there is **no recall
ceiling** on the current golden set.

### BM25, hand-rolled

`app/src/infrastructure/rag/bm25.py` — an inverted index plus per-term IDF, ~60
lines, no third-party dependency. `rank-bm25` / `bm25s` would each drag `numpy`
(`bm25s` also `scipy`) into the default environment, which is deliberately
numpy/torch-free; this mirrors how `infrastructure/evaluation/retrieval_metrics.py`
hand-rolls nDCG rather than pulling an IR library.

- `k1 = 1.5`, `b = 0.75` — standard defaults. Tuning the curve is [M3-05]/[M3-08]'s.
- **Non-negative IDF**: `idf = ln(1 + (N − df + 0.5) / (df + 0.5))` (the
  Lucene / BM25+ form), always `> 0`. The classic Okapi IDF goes **negative**
  for any term in more than half the documents. With the no-stopword decision
  below, ubiquitous Portuguese function words would then carry negative weight
  and actively *penalise* the true clause under a verbose analyst question.
  This form asymptotes to ~0 for ubiquitous terms instead — harmless, not
  harmful. `IDF_VARIANT = "lucene_plus_one"` in `lexical_config.py`.
- Ties broken by `doc_id` ascending, so a committed Recall@k is reproducible
  run to run.

### Indexed text: `text`, not `display_text`

`ChunkRecord` carries two strings: `text` (what the embedder sees — the
clause body with its ancestor-path breadcrumb prepended) and `display_text`
(the breadcrumb stripped). BM25 indexes **`text`**:

- Parity with the dense side (`chunk_repository._row_values` maps
  `ChunkRecord.text` → the `embedded_text` column), so [M3-04]'s RRF fuses two
  rankings over the same representation.
- The DoD's exact terms often live *only* in an ancestor heading. Example:
  `1:clausulas-adicionais/1/1.4/1.4.3` is titled "Cancelamento de Cláusula" and
  its body never says "franquia" or "vidros" — but its `parent_path` is
  `CLÁUSULAS ADICIONAIS > 1. COBERTURA A VIDROS, FARÓIS E LANTERNAS E
  RETROVISORES > 1.4 FRANQUIA A VIDROS, RETROVISORES, FARÓIS E LANTERNAS`.
  `display_text` would make that clause unreachable for "quando a franquia de
  vidros é cancelada?".

The cost — heading terms repeat across sibling chunks and their document
frequency inflates — is bounded: IDF self-corrects, and the owning chunk still
outscores its siblings because it has the term in the breadcrumb *and* the
body. `LEXICAL_INDEX_TEXT_FIELD` is a constant so [M3-08] can A/B `display_text`
in one line.

### The Portuguese analyzer

`app/src/infrastructure/rag/lexical_analyzer.py` — one analysis path, used
identically on the index side and (from [M3-04]) the query side, so the tokens
a chunk is indexed with and the tokens a query is scored against cannot drift
apart (the lexical analog of `embedding_config.format_passage` / `format_query`).

```
raw → NFKC-normalise → lowercase → Unicode word-split (re: \w+)
for each token t:
    folded = NFKD(t) → drop non-ASCII        # e.g. "carência" → "carencia"
    if folded in the stemming-exception set:  emit folded verbatim
    else:                                     emit fold_accents(stem(t))
```

**Stem the accented token, then fold — order is load-bearing.**
snowballstemmer's Portuguese algorithm is defined over accented text: it maps
`indenização → indeniz` (the useful stem), whereas the pre-folded `indenizacao`
stems to the useless `indenizaca`. The accent-fold happens *after* the stemmer,
never before.

**Stemmer: `snowballstemmer` (Portuguese).** Pure Python, zero transitive
dependencies, ~100 KB, BSD-3 — a **core** dependency, because its unit tests
run in CI's `pytest -m unit` and `LexicalRetriever` is a runtime component
[M3-04]'s hybrid retriever will import. It is the same Snowball family
Postgres' built-in `portuguese` text-search config uses, so a future DB-side
lexical path stays consistent. Rejected: **nltk RSLP** (Portuguese-specific but
deliberately aggressive — it over-stems, which for precise domain vocabulary
means a *larger* exception list to maintain; also pulls in `nltk` + a data
download); **PyStemmer** (C extension — build fragility for a speed win
irrelevant at 4,540 docs).

**No stopword removal — deliberate.** A generic Portuguese stoplist drops
`não` / `nem` / `sem` / `salvo` / `exceto` — exactly the words that flip a
coverage clause into an exclusion, which is the hard case the whole M3
milestone targets. BM25 term saturation (`k1`) plus the near-zero IDF the
`lucene_plus_one` variant assigns ubiquitous terms already neutralise function
words. See the negation-word finding in the next section — it is the concrete
reason this matters.

### Stemming-exception list — committed as data

`data/rag/lexical_stemming_exceptions.csv` (header `term,note`; one token per
row — BM25 is bag-of-words, so `sinistro parcial` is just `sinistro` +
`parcial` and cannot be protected as a unit). Loaded by
`app/src/infrastructure/rag/lexical_stemming_exceptions.py`, mirroring
`infrastructure/parsing/rules_loader.py`.

Per the DoD — "worth confirming rather than assuming" — the list was **seeded
minimally and grown only from measured evidence**, not from a list of terms
that *look* fragile.

| term | why | evidence |
| --- | --- | --- |
| `nao` | Negation. Snowball maps `não → na`, colliding with the ubiquitous preposition `na` (`em`+`a`). In the corpus the stem `na` covers `não` ×6656 and `na` ×4481, so `não` inherits a near-zero IDF and negation drops out of BM25 scoring. Kept verbatim so `bens NÃO compreendidos` / `prejuízos NÃO indenizáveis` stay retrievable. | corpus stem-cluster scan; **added after measurement** |
| `casco`, `rcf`, `vmr`, `vd` | SUSEP product-line / indemnity-regime identifiers. Join keys, not lexical targets — per the repo's own convention ("SUSEP identifiers … are join keys, not interface text"). Retrieval-neutral on `golden-set-v1` (the stems have no colliding corpus term); kept on principle. | — |

**Terms checked and deliberately *not* protected** (the "confirm" half of the
DoD):

- `franquia` — stems to `franqu`; the only corpus surface forms are
  `franquia` / `franquias` (clean plural unification). `franquear` /
  `franqueado` appear **0 times** in the corpus, so there is nothing to
  collide with.
- `carência` — stems to `carenc`; only `carência` / `carências` in the corpus.
- `referenciado` — `referenc`, clean gender/number unification
  (`referenciada` / `referenciados` / …).
- `perda` — stems to `perd`, which *does* also cover the verb `perder` /
  `perdera` / `perdido` (~180 occurrences). Protecting it would cost the
  `perda` / `perdas` unification (1,132 / 314) for a marginal precision gain
  and **no measured miss**. Revisit if a "perda total" question shows a
  precision miss.
- `determinado` — `determin`, which also covers the verb `determinar` and its
  14 inflected forms. Same trade-off as `perda`; no measured miss.

**Under-stemming the exception list cannot fix** (documented as a limitation,
not addressed here — protecting a token stops it being stemmed, it does not
make two tokens unify): `Brasil` → `brasil` vs `brasileiro` → `brasileir`;
`território` → `territori` vs `territorial` → `territorial`; `exclusão` →
`exclusa` vs `excluído` → `excluid`; `total` vs `totais` → `tot`; `parcial`
vs `parciais` → `parc`.

### The `[M1-09]` per-constant decision

Every constant this issue introduces changes the published Recall@k, so per
[M1-09]'s rule (a value that moves a published number is experimental design,
like `SEED` / `SAMPLE_SIZE`) they are **all module constants** in
`lexical_config.py`. **Zero new `.env` keys** — the analog of
[M3-02]'s "exactly one constant moved" (here, none).

| constant | location | decision | why |
| --- | --- | --- | --- |
| `LEXICAL_ANALYZER_VERSION`, `STEMMER_LANGUAGE`, `BM25_K1`, `BM25_B`, `IDF_VARIANT`, `LEXICAL_INDEX_TEXT_FIELD` | `lexical_config.py` | code constant | each changes the tokens indexed or the score, therefore the published Recall@k |
| the stemming-exception list | `data/rag/…csv` (committed data) | committed data | it is content, curated and reviewed like `data/parsing/clause_type_mapping.csv`; its sha256 is folded into `config_fingerprint()` |

`config_fingerprint()` (sha256[:16] over the constants + the sorted
exception-token set) stamps every run's `RetrievalRunConfig` and is reserved
for [M3-04]'s query-side cache key — mirroring `embedding_config.config_fingerprint()`.

---

## Pre-registered prediction

Written in the implementation plan **before** the retriever was built or run,
reproduced here unedited:

> `definition` and `direct_lookup` Recall@10 ≥ 0.70; `coverage_with_exclusion`
> and `cross_document` ≤ 0.45; overall Recall@10 in 0.45–0.65 (below M3's 0.85
> target — expected for lexical-only, and what hybrid closes); exclusion-clause
> recall ≤ overall.

---

## Outcome (2026-08-28)

`make eval-retrieval-lexical` → `eval/runs/retrieval_eval_lexical.{md,json}`
(regenerable; gitignored). Analyzer `v1`, BM25 `k1=1.5 b=0.75`, IDF
`lucene_plus_one`, indexed field `text`, 5 stemming exceptions, config
fingerprint `ef0a2dd0c1dfb4e4`. 117 scorable questions (the 23 `unanswerable`
are excluded — empty reference set); all CASCO / CARTA VERDE, all `text`
extraction mode (`golden-set-v1`'s known coverage limit, see `docs/EVALUATION.md`).

### Overall

| metric | value |
| --- | ---: |
| Recall@1 | 24.5% |
| Recall@5 | 47.7% |
| Recall@10 | **58.7%** |
| MRR | **38.8%** |
| nDCG@10 | 41.9% |
| Exclusion-clause recall (top-10, pooled) | **51.9%** (14/27) |

### By question type — predicted vs actual

| question_type | n | predicted R@10 | actual R@10 | actual MRR | actual nDCG@10 | call |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| `direct_lookup` | 64 | ≥ 0.70 | **69.8%** | 48.5% | 53.8% | held |
| `coverage_with_exclusion` | 19 | ≤ 0.45 | **42.1%** | 44.5% | 33.7% | held |
| `cross_document` | 16 | ≤ 0.45 | **62.5%** | 18.8% | 28.7% | **missed** (better than predicted; but MRR 18.8% — the right clause is retrieved deep, not near the top) |
| `definition` | 18 | ≥ 0.70 | **33.3%** | 15.8% | 20.1% | **missed** (worse than predicted — see below) |

Overall Recall@10 58.7% landed inside the predicted 0.45–0.65 band;
exclusion-clause recall (51.9%) ≤ overall (58.7%), as predicted.

### Why `definition` missed the prediction

The assumption was "definition questions name exact terms, so lexical wins".
It does not, because every `definition` question references a **single
whole-glossary clause** (`{doc}:glossario`) and, with no metadata filter, BM25
retrieves *another* document's glossary — the CASCO glossaries share almost all
their definitional vocabulary. It is the cross-document-leakage failure, not a
vocabulary failure: under a perfect document filter `definition` Recall@10 is
**88.9%** (below).

---

## Verdict: which question types lexical retrieval wins outright

**`direct_lookup` is the clear winner** — Recall@10 69.8%, MRR 48.5%,
nDCG@10 53.8%, every metric well above the other three types. These are the
exact-term factual lookups ("qual é o percentual…", "existe limite de
valor…"), where a precise Portuguese term in the question matches the same term
in one clause. The DoD's hypothesis ("expected to be the exact-term ones") is
**confirmed for `direct_lookup`** and **refuted for `definition`** — the latter
loses not on vocabulary but on the missing document filter.

`coverage_with_exclusion` is the weakest by nDCG (33.7%) and the one lexical
structurally cannot solve alone: the reference set is two clauses (a coverage
clause *and* the exclusion that limits it), the question names neither, and the
exclusion clause shares little vocabulary with either. This is the case
[M3-06]'s exclusion co-retrieval exists for.

Implication for [M3-04] fusion: weight lexical up for exact-term / short
factual queries; do not expect it to carry `coverage_with_exclusion` or,
until the filter lands, `definition`.

---

## What the [M3-04] filter recovers

Not part of the committed harness — a one-off simulation (index only the
question's own document, i.e. a perfect `susep_process`+`cnpj` filter). It
isolates lexical's vocabulary-matching capability from the cross-document
leakage that [M3-04]'s pre-filter will remove:

| question_type | unfiltered R@10 | filtered R@10 |
| --- | ---: | ---: |
| `direct_lookup` | 69.8% | 92.2% |
| `definition` | 33.3% | 88.9% |
| `coverage_with_exclusion` | 42.1% | 100.0% |
| `cross_document` | 62.5% | 93.8% |
| **overall** | **58.7%** | **93.2%** |
| exclusion-clause recall | 51.9% | 85.2% |

The entire ~35-point Recall@10 gap is cross-document leakage. Within a
document, lexical BM25 alone already clears M3's Recall@10 ≥ 0.85 target on
`golden-set-v1`. This is the strongest single argument that [M3-04]'s
metadata pre-filter is the highest-leverage next step, and that lexical is not
the bottleneck.

---

## Deferred / handed to later issues

- **A Postgres `tsvector` column + GIN index on `chunk`.** [M3-02]'s scope note
  reserved "the lexical column and its index" for M3-03; the finalised M3-03
  DoD ("BM25", "standalone baseline") governs instead. Postgres' native
  `ts_rank` / `ts_rank_cd` is *not* BM25 (it is a length-normalised
  term-frequency weight), and real in-database BM25 needs a heavy extension
  (ParadeDB `pg_search`, `rum`). The eval harness is file-based and nothing
  queries a lexical column until [M3-04], which owns the DB-side retrieval
  interface and can decide persistence with these numbers in hand.
- **Query-side analysis + a token cache.** [M3-04] adds the query side; it
  reuses this module's `TextAnalyzer` unchanged and may key a cache on
  `config_fingerprint()`. No cache in this issue — the index builds in ~3 s.
- **Hybrid fusion (RRF), metadata pre-filtering** — [M3-04].
- **The lexical-vs-dense-vs-hybrid benchmark matrix** and **`make build-index`**
  — done: [M3-08], `docs/RETRIEVAL_BENCHMARK.md`. Filtered lexical Recall@10 is
  87.2% in the matrix (the committed baseline here is `--filter none`, 58.7%).

## Limitations

- **No metadata filter** (above) — the dominant effect on every number.
- **Cross-document leakage is in play** and `cross_document` questions
  intentionally exercise it; a filter-less lexical retriever is not a realistic
  production configuration.
- **117 / 140 scorable, 2 of 5 product lines** — `golden-set-v1`'s coverage
  limit, not a lexical issue (`docs/EVALUATION.md`).
- **OCR hyphenation noise** survives tokenisation: an OCR artefact like
  `RE- FERENCIADA` becomes the tokens `re` + `ferenciada`. A corpus-quality
  issue, out of scope here.
- **Protected-token plurals**: a protected token bypasses the stemmer, so its
  plural must also be protected to still match its singular. Not currently an
  issue (`nao` has no plural; the SUSEP identifiers are invariant).
- **Merged-clause MRR**: when a chunk's `source_clause_ids` lists the anchor
  before a non-anchor golden reference, the anchor is emitted first and can
  cost a fraction of an MRR point. The merged set is small; accepted.
